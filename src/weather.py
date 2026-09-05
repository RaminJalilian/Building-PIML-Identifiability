"""
Synthetic weather generator for the ACDM benchmark study.

Produces a realistic diurnal/seasonal weather signal for a warm climate
(loosely based on a hot-humid zone) without reference to any specific city
or real dataset.  All parameters are adjustable so experiments can be
reproduced exactly from a seed.

Outputs a DataFrame with columns:
    timestamp   : pandas Timestamp, hourly
    T_oa_C      : outdoor dry-bulb temperature  [°C]
    T_dp_C      : dew-point temperature         [°C]
    W_oa        : humidity ratio                [kg_w / kg_da]
    I_sol_Wm2   : global horizontal irradiance  [W/m²]
    P_Pa        : atmospheric pressure          [Pa]  (constant)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────
P_ATM = 101_325.0   # Pa  (sea-level standard)


def humidity_ratio_from_dewpoint(T_dp_C: np.ndarray,
                                  P_Pa: np.ndarray | float = P_ATM) -> np.ndarray:
    """ASHRAE-standard humidity ratio from dew-point temperature."""
    T_dp_K = T_dp_C + 273.15
    p_ws   = 611.657 * np.exp(17.2694 * T_dp_C / (T_dp_K - 35.85))   # Pa
    p_w    = p_ws
    return 0.621945 * p_w / (P_Pa - p_w)


def synthetic_weather(
    start: str = "2023-01-01 00:00",
    hours: int = 24 * 90,
    T_annual_mean_C: float = 24.0,
    T_annual_amp_C:  float = 5.0,    # seasonal swing ± around mean
    T_diurnal_amp_C: float = 6.0,    # daily swing ± around daily mean
    RH_mean: float = 0.65,           # mean relative humidity (fraction)
    RH_amp:  float = 0.10,           # diurnal RH amplitude
    I_sol_peak_Wm2: float = 900.0,   # clear-sky peak GHI
    noise_T: float = 0.4,            # std of Gaussian noise on T_oa [°C]
    noise_I: float = 20.0,           # std of Gaussian noise on I_sol [W/m²]
    seed: int = 0,
) -> pd.DataFrame:
    """Generate hourly synthetic weather for `hours` timesteps.

    Parameters
    ----------
    start : str
        ISO datetime string for the first timestamp.
    hours : int
        Number of hourly timesteps to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns: timestamp, T_oa_C, T_dp_C, W_oa,
                                I_sol_Wm2, P_Pa
    """
    rng = np.random.default_rng(seed)
    ts  = pd.date_range(start=start, periods=hours, freq="h")

    # fractional day-of-year for seasonal cycle
    doy   = ts.dayofyear.to_numpy(dtype=float)
    hour  = ts.hour.to_numpy(dtype=float)

    # ── Temperature ──────────────────────────────────────────────────────────
    # seasonal: peak in summer (day ~200), trough in winter (day ~20)
    T_seasonal = T_annual_amp_C * np.cos(2 * np.pi * (doy - 200) / 365.0)
    # diurnal: peak at 15:00, trough at 03:00 (hour-of-day mean trough falls in 02:00-05:00)
    T_diurnal  = T_diurnal_amp_C * np.cos(2 * np.pi * (hour - 15.0) / 24.0)
    T_oa       = (T_annual_mean_C + T_seasonal + T_diurnal
                  + rng.normal(0, noise_T, hours))

    # ── Relative humidity (anticorrelated with temperature) ───────────────────
    RH = np.clip(RH_mean - RH_amp * (T_diurnal / T_diurnal_amp_C)
                 + rng.normal(0, 0.02, hours), 0.10, 0.99)

    # dew-point from T_oa and RH
    T_K    = T_oa + 273.15
    p_ws   = 611.657 * np.exp(17.2694 * T_oa / (T_K - 35.85))
    p_w    = RH * p_ws
    T_dp_C = (np.log(p_w / 611.657) * (243.04)
              / (17.625 - np.log(p_w / 611.657)))   # Magnus approximation

    W_oa   = humidity_ratio_from_dewpoint(T_dp_C, P_ATM)

    # ── Solar irradiance (daytime trapezoid + noise) ──────────────────────────
    # sunrise ~06:00, sunset ~18:30 (approximate for mid-latitude summer)
    rise, set_ = 6.0, 18.5
    frac       = np.clip((hour - rise) / (set_ - rise), 0.0, 1.0)
    # triangle: peaks at solar noon (~12:15)
    solar_base = I_sol_peak_Wm2 * np.sin(np.pi * frac) ** 1.5
    solar_base = np.where((hour >= rise) & (hour <= set_), solar_base, 0.0)
    # seasonal scaling: more solar in summer
    solar_seas = 1.0 + 0.25 * np.cos(2 * np.pi * (doy - 172) / 365.0)
    I_sol      = np.clip(solar_base * solar_seas
                         + rng.normal(0, noise_I, hours), 0.0, None)

    return pd.DataFrame({
        "timestamp": ts,
        "T_oa_C":    T_oa.astype(np.float32),
        "T_dp_C":    T_dp_C.astype(np.float32),
        "W_oa":      W_oa.astype(np.float32),
        "I_sol_Wm2": I_sol.astype(np.float32),
        "P_Pa":      np.full(hours, P_ATM, dtype=np.float32),
    })


if __name__ == "__main__":
    wx = synthetic_weather(hours=24 * 14, seed=0)
    print(wx[["timestamp", "T_oa_C", "T_dp_C", "W_oa", "I_sol_Wm2"]].head(10).to_string(index=False))
    print(f"\nT_oa : {wx.T_oa_C.min():.1f} – {wx.T_oa_C.max():.1f} °C")
    print(f"W_oa : {wx.W_oa.min()*1e3:.1f} – {wx.W_oa.max()*1e3:.1f} g/kg")
    print(f"I_sol: {wx.I_sol_Wm2.max():.0f} W/m² peak")
