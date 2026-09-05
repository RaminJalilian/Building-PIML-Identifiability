"""
analysis_common.py — shared helpers for the read-only diagnostic scripts.

Provides:
  provenance(*files)      -> a one-line "PROVENANCE  git=<sha>  <file>=sha256:<hash> ..."
                             string so every analysis run records the code + input state.
  fitted_ratios(variant)  -> resistance ratios (recovered / truth), averaged over seeds,
                             read from results_summary_seeds.csv at runtime. NEVER hardcoded.

These keep the five analysis scripts free of hardcoded result literals: any number that
came from a training run is loaded from the current results CSVs instead.
"""
from __future__ import annotations
import os
import hashlib
import subprocess
import numpy as np
import pandas as pd

from model import BuildingParams

# macOS Accelerate (vecLib) spuriously raises divide/overflow/invalid FP flags from
# inside BLAS matmul for some array shapes, even when the result is finite and correct
# (verified: all downstream values print finite). These are not numerical errors in the
# analysis; silence just these three so the diagnostic output is clean. Real algorithmic
# issues would surface as NaN/inf in the printed results, which are checked by eye.
np.seterr(divide="ignore", over="ignore", invalid="ignore")

P  = BuildingParams()
RS = P.R_zw + P.R_wo


def provenance(*files: str) -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        sha = "nogit"
    parts = [f"git={sha}"]
    for f in files:
        try:
            with open(f, "rb") as fh:
                parts.append(f"{os.path.basename(f)}=sha256:"
                             f"{hashlib.sha256(fh.read()).hexdigest()[:12]}")
        except FileNotFoundError:
            parts.append(f"{os.path.basename(f)}=MISSING")
    return "PROVENANCE  " + "  ".join(parts)


def fitted_ratios(variant: str,
                  sensing: str = "S1",
                  noise_pct: int = 0,
                  seeds_csv: str = "../results/results_summary_seeds.csv") -> dict | None:
    """Fitted resistance ratios (recovered / truth) for `variant`, averaged over seeds.

    Returns {'R_zw', 'R_wo', 'R_inf', 'R_s'} as multiples of truth, or None if the
    variant/cell is absent. Read from the current results — no literals baked in.
    """
    s = pd.read_csv(seeds_csv)
    sub = s[(s.variant == variant) & (s.sensing == sensing) & (s.noise_pct == noise_pct)]
    if len(sub) == 0:
        return None
    rzw, rwo, rinf = sub.rec_R_zw.mean(), sub.rec_R_wo.mean(), sub.rec_R_inf.mean()
    return dict(R_zw=rzw / P.R_zw, R_wo=rwo / P.R_wo,
                R_inf=rinf / P.R_inf, R_s=(rzw + rwo) / RS)
