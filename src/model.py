"""
Forward building thermal model: 2R2C sensible heat only.

Deliberately simplified relative to the UCF model (no humidity state) so
that the RC parameter space is small and identifiable, and the inverse
problem is clean for the benchmark study.

State vector (2 components):
    x = [T_z,  T_w]^T
    T_z : zone air temperature               [°C]
    T_w : lumped envelope (wall) temperature [°C]

Continuous-time dynamics
------------------------
    C_z dT_z/dt = (T_w - T_z)/R_zw
                + (T_oa - T_z)/R_inf
                + alpha_win * A_eff_win * I_sol
                + Q_int(t)
                - Q_hvac(T_z)

    C_w dT_w/dt = (T_z - T_w)/R_zw
                + (T_oa + alpha_sa*I_sol - T_w)/R_wo
                + (1 - alpha_win) * A_eff_win * I_sol

Closed-loop HVAC (proportional, hard-clipped):
    Q_hvac = clip(K_T * (T_z - T_z_sp), 0, Q_max)

Observation model (what meters see):
    Q_cw  = Q_hvac                          [W]  chilled-water meter
    P_elec = Q_cw / COP_eff + P_aux         [W]  HVAC electricity

Ground-truth parameters are stored in BuildingParams and are the *known*
values the inverse problem will attempt to recover from noisy Q_cw (and
optionally T_z) observations.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


# ============================================================================
# Ground-truth parameter bundle
# ============================================================================
@dataclass
class BuildingParams:
    """Physical parameters for the synthetic 2R2C benchmark building.

    These are the GROUND TRUTH values.  The inverse problem estimates them
    from noisy observations.  They represent a generic ~3 000 m² commercial
    building in a warm climate.
    """
    # -- thermal capacitances
    C_z: float = 3.5e7        # J/K   zone air + light interior mass
    C_w: float = 2.0e8        # J/K   lumped envelope mass

    # -- thermal resistances
    R_zw:  float = 4.5e-4     # K/W   zone-to-envelope coupling
    R_wo:  float = 2.8e-3     # K/W   envelope-to-outdoor
    R_inf: float = 3.5e-4     # K/W   infiltration / outdoor air

    # -- solar
    A_eff_win: float = 60.0   # m²    effective window aperture (A_win × SHGC)
    alpha_win: float = 0.55   # –     fraction of window solar → zone air
    alpha_sa:  float = 0.10   # K/(W/m²)  sol-air coefficient

    # -- HVAC control
    T_z_sp:  float = 23.0     # °C    cooling setpoint
    K_T:     float = 2.0e5    # W/K   proportional gain
    Q_max:   float = 4.0e5    # W     sensible cooling capacity

    # -- electricity observation
    COP_eff: float = 3.5      # –     building-side AHU efficiency
    P_aux:   float = 4000.0   # W     always-on auxiliaries

    def as_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Internal-gain schedule (generic commercial building)
# ============================================================================
def internal_gains_schedule(
    timestamps: pd.DatetimeIndex,
    Q_int_peak_W: float = 50_000.0,
) -> np.ndarray:
    """Generate ground-truth Q_int(t) for a generic commercial occupancy.

    Weekday: sigmoid ramp 07-09, plateau 09-18, ramp down 18-21.
    Weekend: 15% of weekday peak.
    Baseline (overnight): 20% of weekday peak (servers, signage, etc.).

    Returns float32 array of shape [len(timestamps)].
    """
    hod = timestamps.hour + timestamps.minute / 60.0
    dow = timestamps.dayofweek

    def _sig(x: np.ndarray, center: float, k: float) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-k * (x - center)))

    weekday_curve = _sig(hod, 8.0, 1.5) - _sig(hod, 19.0, 1.5)
    weekday_curve = np.clip(weekday_curve, 0.0, 1.0)

    occ = np.where(dow < 5, weekday_curve, 0.15 * weekday_curve)
    occ_total = 0.20 + 0.80 * occ      # never below 20% of peak

    return (Q_int_peak_W * occ_total).astype(np.float32)


# ============================================================================
# Interpolator helper
# ============================================================================
def _make_interp(t_sec: np.ndarray, vals: np.ndarray) -> Callable:
    return interp1d(
        t_sec, vals, kind="linear", bounds_error=False,
        fill_value=(vals[0], vals[-1]), assume_sorted=True,
    )


# ============================================================================
# ODE right-hand side
# ============================================================================
def building_rhs(
    t: float,
    x: list,
    p: BuildingParams,
    T_oa_fn: Callable,
    I_sol_fn: Callable,
    Q_int_fn: Callable,
) -> list:
    """2-state thermal ODE RHS.  t in seconds."""
    T_z, T_w = x

    T_oa  = float(T_oa_fn(t))
    I_sol = float(I_sol_fn(t))
    Q_int = float(Q_int_fn(t))

    Q_hvac = np.clip(p.K_T * (T_z - p.T_z_sp), 0.0, p.Q_max)

    Q_sol_zone = p.alpha_win * p.A_eff_win * I_sol
    Q_sol_wall = (1.0 - p.alpha_win) * p.A_eff_win * I_sol
    T_sa       = T_oa + p.alpha_sa * I_sol

    dTz = (
        (T_w - T_z) / p.R_zw
        + (T_oa - T_z) / p.R_inf
        + Q_sol_zone
        + Q_int
        - Q_hvac
    ) / p.C_z

    dTw = (
        (T_z - T_w) / p.R_zw
        + (T_sa - T_w) / p.R_wo
        + Q_sol_wall
    ) / p.C_w

    return [dTz, dTw]


# ============================================================================
# Observation model
# ============================================================================
def observe(
    T_z: np.ndarray,
    Q_int: np.ndarray,
    p: BuildingParams,
) -> pd.DataFrame:
    """Compute noise-free meter signals from the state trajectory."""
    Q_hvac = np.clip(p.K_T * (T_z - p.T_z_sp), 0.0, p.Q_max)
    Q_cw   = Q_hvac
    P_elec = Q_cw / p.COP_eff + p.P_aux
    return pd.DataFrame({
        "Q_hvac_W":  Q_hvac,
        "Q_cw_W":    Q_cw,
        "P_elec_W":  P_elec,
    })


# ============================================================================
# Integrator wrapper
# ============================================================================
def simulate(
    weather: pd.DataFrame,
    params: BuildingParams,
    Q_int_peak_W: float = 50_000.0,
    x0: Optional[np.ndarray] = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> pd.DataFrame:
    """Simulate the 2R2C building over the timestamps in `weather`.

    Returns a DataFrame with columns:
        timestamp, T_z, T_w          (state)
        Q_hvac_W, Q_cw_W, P_elec_W  (observations, noise-free)
        Q_int_W                      (ground-truth internal gain)
    """
    ts    = pd.DatetimeIndex(weather["timestamp"])
    t_sec = (ts - ts[0]).total_seconds().to_numpy()

    Q_int = internal_gains_schedule(ts, Q_int_peak_W=Q_int_peak_W)
    T_oa  = weather["T_oa_C"].to_numpy()
    I_sol = weather["I_sol_Wm2"].to_numpy()

    T_oa_fn  = _make_interp(t_sec, T_oa)
    I_sol_fn = _make_interp(t_sec, I_sol)
    Q_int_fn = _make_interp(t_sec, Q_int)

    if x0 is None:
        x0 = np.array([params.T_z_sp + 0.5,
                       0.5 * (params.T_z_sp + T_oa[0])])

    sol = solve_ivp(
        fun=lambda t, x: building_rhs(t, x, params, T_oa_fn, I_sol_fn, Q_int_fn),
        t_span=(t_sec[0], t_sec[-1]),
        y0=x0,
        t_eval=t_sec,
        method="RK45",
        rtol=rtol, atol=atol,
        max_step=900.0,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    T_z = sol.y[0]
    T_w = sol.y[1]

    obs = observe(T_z, Q_int, params)
    out = pd.DataFrame({"timestamp": ts, "T_z": T_z, "T_w": T_w})
    out = pd.concat([out, obs], axis=1)
    out["Q_int_W"] = Q_int
    return out


# ============================================================================
# Quick self-test
# ============================================================================
if __name__ == "__main__":
    from weather import synthetic_weather
    wx = synthetic_weather(hours=24 * 7, seed=0)
    p  = BuildingParams()
    df = simulate(wx, p)

    print(df[["timestamp", "T_z", "T_w", "Q_cw_W", "P_elec_W", "Q_int_W"]].head(6).to_string(index=False))
    print(f"\nT_z   : {df.T_z.min():.2f} – {df.T_z.max():.2f} °C")
    print(f"T_w   : {df.T_w.min():.2f} – {df.T_w.max():.2f} °C")
    print(f"Q_cw  : {df.Q_cw_W.mean()/1e3:.1f} kW mean  |  {df.Q_cw_W.max()/1e3:.1f} kW peak")
    print(f"Q_int : {df.Q_int_W.mean()/1e3:.1f} kW mean  |  {df.Q_int_W.max()/1e3:.1f} kW peak")
