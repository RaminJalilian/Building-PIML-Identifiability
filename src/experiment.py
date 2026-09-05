"""
experiment.py — Run the full ACDM ablation study.

Grid:
  6 variants × 3 sensing configs × 3 ADDITIONAL-noise levels × 5 seeds = 270 executions.
  These 270 executions cover only 54 nominal (variant, sensing, noise) cells and, because
  only V4 uses T_z, just 24 distinct variant × effective-sensing × noise OBJECTIVES; with
  V3/V4F deterministic at 0 added noise there are 112 distinct scientific outputs in total.
  Duplicate sensing/seed rows are NOT independent replication (see README N1 / audit D04).

Additional-noise levels (0/5/10) are ADDED ON TOP of the base measurement layer; they do not
re-scale it. The forward simulation is run only once.

Outputs (written to results/):
  results_all.json      all 270 raw VariantResult objects (serialised; each carries its seed)
  results_summary.csv   legacy seed-0 flat table
  results_summary_seeds.csv / results_aggregated.csv   per-seed and 5-seed-mean tables

Usage:
  cd src
  python experiment.py              # full grid (270 executions)
  python experiment.py --quick      # V4+V5 only, S1+S3 only, noise 0%+10%
  python experiment.py --reaggregate  # rebuild identifiable metrics from existing seeds CSV

Variants: V1, V2, V3, V4, V4F, V5 (see variants.py)
Note: V4D (dynamic ODE training) is implemented in variants.py but not
      included in the paper benchmark grid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from variants import run_variant, VariantResult
from model import BuildingParams

# ── Ground-truth identifiable combinations ───────────────────────────────────
# The quasi-static operator depends on R_zw and R_wo only through their series sum,
# so only R_s = R_zw+R_wo (G_s = 1/R_s) and R_inf (G_inf = 1/R_inf) are identifiable.
# The R_zw/R_wo split and any three-resistance mean are non-identifiable.
_GT        = BuildingParams()
R_S_TRUE   = _GT.R_zw + _GT.R_wo
G_S_TRUE   = 1.0 / R_S_TRUE
R_INF_TRUE = _GT.R_inf
G_INF_TRUE = 1.0 / _GT.R_inf

# ── Output paths (anchored to repo root; CWD-independent) ──────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
os.makedirs(ROOT / "results", exist_ok=True)
os.makedirs(ROOT / "figures", exist_ok=True)

RESULTS_JSON = str(ROOT / "results/results_all.json")
RESULTS_CSV  = str(ROOT / "results/results_summary.csv")

# ── Experiment grid ───────────────────────────────────────────────────────────
ALL_VARIANTS  = ["V1", "V2", "V3", "V4", "V4F", "V5"]
ALL_SENSING   = ["S1", "S2", "S3"]
ALL_NOISE     = [0.00, 0.05, 0.10]   # relative noise levels (0%, 5%, 10%)
ALL_SEEDS     = [0, 1, 2, 3, 4]      # repeated seeds for mean +/- std reporting

QUICK_VARIANTS = ["V4", "V4F", "V5"]
QUICK_SENSING  = ["S1", "S3"]
QUICK_NOISE    = [0.00, 0.10]
QUICK_SEEDS    = [0, 1]


# ============================================================================
# Noise injection
# ============================================================================

def inject_noise(df: pd.DataFrame, noise_level: float, seed: int = 0) -> pd.DataFrame:
    """Add EXTRA Gaussian noise on top of the base measurement layer (ADDITIONAL-noise level).

    This does NOT re-scale the base noise. At noise_level=0.00 the measured signal is the base
    measurement layer unchanged: truth + 3 kW bias + base heteroscedastic noise + dropout
    (base residual std ~2.98 kW) -- it is NOT truth+bias only. At noise_level>0 an extra
    zero-mean Gaussian term with std = noise_level*Q_cw_W is added on top (e.g. total residual
    std ~4.86 kW at 5% and ~7.78 kW at 10% for the checked seed). The dropout mask is preserved.
    """
    if noise_level == 0.0:
        return df.copy()

    rng    = np.random.default_rng(seed)
    df     = df.copy()
    Q_true = df["Q_cw_W"].to_numpy()
    extra  = rng.normal(0, noise_level * Q_true, len(df))
    # only add noise where measurement exists
    valid  = df["Q_cw_meas_W"].notna()
    df.loc[valid, "Q_cw_meas_W"] = df.loc[valid, "Q_cw_meas_W"] + extra[valid]
    return df


# ============================================================================
# Serialisation helpers
# ============================================================================

def result_to_dict(r: VariantResult, seed: int = 0) -> dict:
    """Convert VariantResult to a JSON-serialisable dict (includes its seed)."""
    return dict(
        variant        = r.variant,
        sensing        = r.sensing,
        seed           = seed,
        noise_level    = r.noise_level,
        metrics_train  = r.metrics_train,
        metrics_test   = r.metrics_test,
        param_errors   = r.param_errors,
        recovered_rc   = r.recovered_rc,
        train_time_s   = r.train_time_s,
        epochs_trained = r.epochs_trained,
        train_losses   = r.train_losses,
        # arrays: store as lists
        q_int_pred = r.q_int_pred.tolist() if r.q_int_pred is not None else None,
        q_cw_pred  = r.q_cw_pred.tolist()  if r.q_cw_pred  is not None else None,
    )


def result_to_row(r: VariantResult, seed: int = 0) -> dict:
    """Flat dict for the summary CSV."""
    row = dict(
        variant        = r.variant,
        sensing        = r.sensing,
        noise_pct      = round(r.noise_level * 100),
        seed           = seed,
        epochs         = r.epochs_trained,
        train_time_s   = round(r.train_time_s, 1),
    )
    for k, v in r.metrics_test.items():
        row[f"test_{k}"] = round(v, 4) if not np.isnan(v) else np.nan
    for k, v in r.metrics_train.items():
        row[f"train_{k}"] = round(v, 4) if not np.isnan(v) else np.nan
    for k, v in r.param_errors.items():
        row[k] = round(v, 2) if not np.isnan(v) else np.nan
    for k, v in r.recovered_rc.items():
        row[f"rec_{k}"] = v
    return row


# ============================================================================
# Main experiment runner
# ============================================================================

def run_experiments(variants, sensing_names, noise_levels, seeds, data_dir=str(DATA_DIR)):
    """Run the full grid over seeds and return list of VariantResult objects."""

    # Load sensing DataFrames once
    sensing_dfs = {}
    for s in sensing_names:
        path = os.path.join(data_dir, f"sensing_{s}.csv")
        sensing_dfs[s] = pd.read_csv(path, parse_dates=["timestamp"])

    total = len(variants) * len(sensing_names) * len(noise_levels) * len(seeds)
    print(f"\n{'='*60}")
    print(f"ACDM Ablation Study — {total} runs")
    print(f"Variants : {variants}")
    print(f"Sensing  : {sensing_names}")
    print(f"Noise    : {[f'{n*100:.0f}%' for n in noise_levels]}")
    print(f"Seeds    : {seeds}")
    print(f"{'='*60}\n")

    results = []
    run_idx = 0
    t_start = time.time()

    for seed in seeds:
        for s_name in sensing_names:
            df_base = sensing_dfs[s_name]
            df_train_base = df_base[df_base["split"] == "train"].reset_index(drop=True)
            df_test_base  = df_base[df_base["split"] == "test"].reset_index(drop=True)

            for noise in noise_levels:
                # Vary noise realization per seed so both noise and init differ
                df_train = inject_noise(df_train_base, noise, seed=int(noise*100) + 1000*seed)
                df_test  = inject_noise(df_test_base,  noise, seed=int(noise*100) + 1000*seed + 1)

                for v_name in variants:
                    run_idx += 1
                    print(f"[{run_idx}/{total}]", end=" ")
                    result = run_variant(v_name, df_train, df_test, noise, s_name, seed=seed)
                    results.append((result, seed))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"All {total} runs complete in {elapsed/60:.1f} min")
    print(f"{'='*60}\n")
    return results


# ============================================================================
# Save results
# ============================================================================

RESULTS_SEEDS_CSV = str(ROOT / "results/results_summary_seeds.csv")
RESULTS_AGG_CSV   = str(ROOT / "results/results_aggregated.csv")


def add_identifiable_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the identifiable-combination recovery errors from the per-run recovered
    resistances. R_zw and R_wo enter the operator only through R_s = R_zw+R_wo, so the
    identifiable metrics are R_s / G_s (=1/R_s) and R_inf / G_inf (=1/R_inf). This only
    DERIVES columns from existing recovered values — it recomputes no run value.
    """
    df = df.copy()
    if {"rec_R_zw", "rec_R_wo"}.issubset(df.columns):
        R_s = df["rec_R_zw"] + df["rec_R_wo"]
        df["rec_R_s"] = R_s
        df["err_R_s"] = (R_s - R_S_TRUE).abs() / R_S_TRUE * 100.0
        df["err_G_s"] = ((1.0 / R_s) - G_S_TRUE).abs() / G_S_TRUE * 100.0
    if "rec_R_inf" in df.columns:
        df["err_G_inf"] = ((1.0 / df["rec_R_inf"]) - G_INF_TRUE).abs() / G_INF_TRUE * 100.0
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across seeds. PRIMARY recovery metrics are the identifiable combinations
    (err_R_s, err_G_s, err_R_inf, err_G_inf) plus their two-combination mean (ident_mean_err).
    err_R_zw and err_R_wo are retained ONLY as non-identifiable split diagnostics, never as
    recovery results. Capacitance errors are excluded (C_z, C_w do not enter the operator).
    """
    df = df.copy()
    ident_cols = [c for c in ["err_R_s", "err_G_s", "err_R_inf", "err_G_inf"] if c in df.columns]
    if {"err_R_s", "err_R_inf"}.issubset(df.columns):
        df["ident_mean_err"] = df[["err_R_s", "err_R_inf"]].mean(axis=1)
        ident_cols = ident_cols + ["ident_mean_err"]
    split_diag_cols = [c for c in ["err_R_zw", "err_R_wo"] if c in df.columns]  # NON-identifiable
    metric_cols = [c for c in (["test_cv_rmse"] + ident_cols + split_diag_cols) if c in df.columns]
    grp = df.groupby(["variant", "sensing", "noise_pct"])
    agg = grp[metric_cols].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    return agg


def _print_summary(df: pd.DataFrame) -> None:
    print("\nTest CV-RMSE mean +/- std (%), S1:")
    s1 = df[df["sensing"] == "S1"]
    print(s1.groupby(["variant", "noise_pct"])["test_cv_rmse"].agg(["mean", "std"]).to_string())
    if "ident_mean_err" in df.columns:
        print("\nIdentifiable recovery error [mean of R_s, R_inf] mean +/- std (%), S1:")
        s1r = s1[s1["ident_mean_err"].notna()]
        print(s1r.groupby(["variant", "noise_pct"])["ident_mean_err"].agg(["mean", "std"]).to_string())


def save_results(results):
    """results is a list of (VariantResult, seed) tuples."""
    # JSON — full results including loss curves and predictions (each carries its seed)
    with open(RESULTS_JSON, "w") as f:
        json.dump([result_to_dict(r, seed) for r, seed in results], f, indent=2)
    print(f"Saved -> {RESULTS_JSON}")

    # Per-seed CSV — every run, tagged with seed (+ derived identifiable columns)
    rows = [result_to_row(r, seed) for r, seed in results]
    df   = add_identifiable_metrics(pd.DataFrame(rows))
    df.to_csv(RESULTS_SEEDS_CSV, index=False)
    print(f"Saved -> {RESULTS_SEEDS_CSV}")

    # Also keep the legacy single-file name (seed 0 only) for backward compat
    df[df["seed"] == df["seed"].min()].to_csv(RESULTS_CSV, index=False)
    print(f"Saved -> {RESULTS_CSV}  (seed={df['seed'].min()} only, legacy)")

    agg = aggregate(df)
    agg.to_csv(RESULTS_AGG_CSV, index=False)
    print(f"Saved -> {RESULTS_AGG_CSV}")
    _print_summary(df)


def reaggregate():
    """Rebuild the identifiable-metric columns of the seeds + aggregated CSVs from the
    EXISTING per-run values (no variant is re-run). Migrates the reported recovery
    combination to identifiable-only while leaving every computed run value untouched.
    """
    df = add_identifiable_metrics(pd.read_csv(RESULTS_SEEDS_CSV))
    df.to_csv(RESULTS_SEEDS_CSV, index=False)
    print(f"Rewrote -> {RESULTS_SEEDS_CSV}  (added identifiable columns; run values unchanged)")
    aggregate(df).to_csv(RESULTS_AGG_CSV, index=False)
    print(f"Rewrote -> {RESULTS_AGG_CSV}  (identifiable primary metrics)")
    _print_summary(df)


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Run reduced grid for quick testing")
    parser.add_argument("--reaggregate", action="store_true",
                        help="Rebuild identifiable metrics + aggregate from existing seeds CSV "
                             "(no variants re-run; run values untouched)")
    args = parser.parse_args()

    if args.reaggregate:
        reaggregate()
        sys.exit(0)

    if args.quick:
        variants     = QUICK_VARIANTS
        sensing_list = QUICK_SENSING
        noise_list   = QUICK_NOISE
        seed_list    = QUICK_SEEDS
        print("QUICK MODE: reduced grid")
    else:
        variants     = ALL_VARIANTS
        sensing_list = ALL_SENSING
        noise_list   = ALL_NOISE
        seed_list    = ALL_SEEDS

    results = run_experiments(variants, sensing_list, noise_list, seed_list)
    save_results(results)
