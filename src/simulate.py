"""
Synthetic measurement dataset for the ACDM constraint-ablation benchmark.

Builds a controlled 'measured' dataset from the noise-free forward simulation
by adding:
  - Heteroscedastic Gaussian noise on Q_cw and T_z meters
  - Random timestep dropout on each meter
  - A constant unknown bias on Q_cw (mimics calibration drift)
  - Three sensing configurations used in the experiments:
        S1 : Q_cw only            (energy-only metering)
        S2 : Q_cw + sparse T_z    (10% of T_z timesteps observed)
        S3 : Q_cw + full T_z      (all T_z timesteps observed)

Outputs (written to data/):
  - synthetic_dataset.csv      full record with truth + noisy signals + masks
  - sensing_S1.csv / S2 / S3  experiment-ready views (NaN where unobserved)
  - diagnostic.png             multi-panel summary plot

Train / test split:
  - Training : first 80% of timesteps
  - Test     : last 20% of timesteps  (never used during training)
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from weather import synthetic_weather
from model import BuildingParams, simulate

# Repo root, so outputs land in the AUTHORITATIVE data/ and figures/ regardless of CWD
# (experiment.py reads this same data/). Resolves the historical src/data vs data hazard.
ROOT = Path(__file__).resolve().parents[1]


# ============================================================================
# Measurement layer
# ============================================================================
def add_measurement_layer(
    df: pd.DataFrame,
    *,
    # Q_cw noise
    sigma_cw_rel: float = 0.03,
    sigma_cw_abs_W: float = 2000.0,
    cw_bias_W: float = 3000.0,          # UNKNOWN to inverse model
    dropout_cw: float = 0.01,
    # T_z noise
    sigma_Tz_C: float = 0.3,
    dropout_Tz: float = 0.005,
    seed: int = 42,
) -> pd.DataFrame:
    """Add sensor noise, bias, and random dropout to truth columns.

    The chilled-water bias is deliberately hidden from the inverse model
    to test whether identifiability analysis can detect it.
    """
    rng = np.random.default_rng(seed)
    n   = len(df)
    df  = df.copy()

    # ── Q_cw ─────────────────────────────────────────────────────────────────
    Q_true   = df["Q_cw_W"].to_numpy()
    sigma_cw = np.sqrt(sigma_cw_abs_W**2 + (sigma_cw_rel * Q_true)**2)
    Q_meas   = Q_true + cw_bias_W + rng.normal(0, sigma_cw, n)
    mask_cw  = rng.random(n) < dropout_cw
    Q_meas[mask_cw] = np.nan

    # ── T_z ──────────────────────────────────────────────────────────────────
    Tz_true  = df["T_z"].to_numpy()
    Tz_meas  = Tz_true + rng.normal(0, sigma_Tz_C, n)
    mask_Tz  = rng.random(n) < dropout_Tz
    Tz_meas[mask_Tz] = np.nan

    df["Q_cw_meas_W"]   = Q_meas
    df["T_z_meas_C"]    = Tz_meas
    df["sigma_cw_W"]    = sigma_cw
    df["cw_dropout"]    = mask_cw
    df["Tz_dropout"]    = mask_Tz
    df["cw_bias_W"]     = cw_bias_W   # stored for analysis; inverse model does NOT see this
    return df


# ============================================================================
# Sensing configuration masks
# ============================================================================
def make_sensing_configs(
    df: pd.DataFrame,
    sparse_frac: float = 0.10,
    seed: int = 99,
) -> dict[str, pd.DataFrame]:
    """Return three experiment-ready DataFrames with NaN where unobserved.

    S1 : Q_cw_meas only   → T_z_obs = NaN everywhere
    S2 : Q_cw_meas + sparse T_z  (sparse_frac of timesteps)
    S3 : Q_cw_meas + full T_z
    """
    rng  = np.random.default_rng(seed)
    n    = len(df)

    base_cols = ["timestamp", "T_oa_C", "I_sol_Wm2", "Q_int_W",
                 "Q_cw_W", "Q_cw_meas_W", "T_z", "T_z_meas_C",
                 "sigma_cw_W", "split"]

    configs = {}

    for name in ("S1", "S2", "S3"):
        d = df[base_cols].copy()

        if name == "S1":
            d["T_z_obs"] = np.nan

        elif name == "S2":
            sparse_mask = rng.random(n) < sparse_frac
            d["T_z_obs"] = np.where(sparse_mask, df["T_z_meas_C"].values, np.nan)

        else:   # S3
            d["T_z_obs"] = df["T_z_meas_C"].values

        configs[name] = d

    return configs


# ============================================================================
# Diagnostic plot
# ============================================================================
def make_diagnostic_plot(df: pd.DataFrame, fname: str) -> None:
    ts  = pd.DatetimeIndex(df["timestamp"])
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    ax.plot(ts, df["T_oa_C"], color="firebrick", lw=0.9, label="T_oa")
    ax.set_ylabel("T_oa [°C]"); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    ax.set_title("Synthetic benchmark dataset — overview", fontsize=11)

    ax = axes[1]
    ax.plot(ts, df["T_z"],       color="black",     lw=1.0, label="T_z truth")
    ax.plot(ts, df["T_w"],       color="gray",      lw=0.8, label="T_w truth")
    ax.plot(ts, df["T_z_meas_C"], color="steelblue", lw=0, marker=".", ms=1.5,
            alpha=0.5, label="T_z measured")
    ax.axhline(BuildingParams().T_z_sp, color="green", ls="--", lw=0.8,
               label=f"setpoint {BuildingParams().T_z_sp:.1f}°C")
    ax.set_ylabel("Temperature [°C]"); ax.legend(fontsize=7, ncol=2)

    ax = axes[2]
    ax.plot(ts, df["Q_int_W"]  / 1e3, color="purple",  lw=0.9, label="Q_int truth")
    ax.set_ylabel("Q_int [kW]"); ax.legend(fontsize=8); ax.grid(alpha=0.2)

    ax = axes[3]
    ax.plot(ts, df["Q_cw_W"]      / 1e3, color="black",  lw=0.9, alpha=0.5, label="Q_cw truth")
    ax.plot(ts, df["Q_cw_meas_W"] / 1e3, color="tomato", lw=0, marker=".", ms=1.5,
            label="Q_cw measured (biased+noisy)")
    ax.set_ylabel("Q_cw [kW]"); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    ax.set_xlabel("Time")

    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"  wrote {fname}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    DATA_DIR = ROOT / "data"
    FIG_DIR  = ROOT / "figures"
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR,  exist_ok=True)

    # ── Generate forward simulation ───────────────────────────────────────────
    # Start in July so T_oa stays above T_sp = 23°C (cooling-dominated).
    # The quasi-static approximation breaks when T_oa < T_sp — winter data
    # produces negative Q_ext terms that bias parameter recovery.
    wx = synthetic_weather(start="2023-07-01 00:00", hours=24 * 90, seed=0)
    p  = BuildingParams()

    frac_cooling = (wx["T_oa_C"] > p.T_z_sp).mean()
    print(f"Fraction T_oa > T_sp ({p.T_z_sp}C): {frac_cooling*100:.1f}%  "
          f"(target >80% for valid QS approximation)")

    df = simulate(wx, p, Q_int_peak_W=50_000.0)

    # attach outdoor weather columns needed downstream
    df["T_oa_C"]    = wx["T_oa_C"].values
    df["I_sol_Wm2"] = wx["I_sol_Wm2"].values

    # ── Add measurement noise ─────────────────────────────────────────────────
    df = add_measurement_layer(df, seed=42)

    # ── Train / test split (80 / 20) ──────────────────────────────────────────
    split_idx = int(0.80 * len(df))
    df["split"] = "train"
    df.loc[df.index >= split_idx, "split"] = "test"

    n_train = (df.split == "train").sum()
    n_test  = (df.split == "test").sum()
    n_drop  = df["cw_dropout"].sum()
    bias    = df["cw_bias_W"].iloc[0]
    print(f"Dataset : {len(df)} rows  |  train {n_train}  test {n_test}")
    print(f"CW dropout : {n_drop} timesteps ({n_drop/len(df)*100:.1f}%)")
    print(f"CW bias    : +{bias/df['Q_cw_W'].mean()*100:.1f}% of mean  (hidden from model)")

    df.to_csv(DATA_DIR / "synthetic_dataset.csv", index=False)
    print(f"  wrote {DATA_DIR / 'synthetic_dataset.csv'}")

    # ── Sensing configurations ────────────────────────────────────────────────
    configs = make_sensing_configs(df, sparse_frac=0.10, seed=99)
    for name, d in configs.items():
        path = DATA_DIR / f"sensing_{name}.csv"
        d.to_csv(path, index=False)
        n_obs_Tz = d["T_z_obs"].notna().sum()
        print(f"  wrote {path}  (T_z observed: {n_obs_Tz}/{len(d)} = {n_obs_Tz/len(d)*100:.1f}%)")

    # ── Diagnostic plot ───────────────────────────────────────────────────────
    make_diagnostic_plot(df, str(FIG_DIR / "diagnostic.png"))

    # ── Ground-truth parameter summary ───────────────────────────────────────
    print("\nGround-truth BuildingParams:")
    for k, v in p.as_dict().items():
        print(f"  {k:<14} = {v:.4g}")
