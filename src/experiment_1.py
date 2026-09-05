"""
experiment_1.py — the hourly quasi-static RECOVERY CEILING.

S1 corrected data, KNOWN true forcing (Q_int from ground truth), no neural net. Fit only the
identifiable conductances (G_s, G_inf) [+ intercept b where stated] to four load targets, and
ask: is truth (G_s*, G_inf*) the optimum of the data objective, or is the hourly static
operator's own optimum displaced from truth even under ideal (clean, noiseless) data?

Operator (variants.quasi_static_Q_cw, reduced wall, no envelope-solar term):
    Q_cw_qs = max( Q_int + Swin + dT_sa*G_s + dT_oa*G_inf [+ b], 0 )
    Swin  = alpha_win*A_eff*I_sol ;  dT_sa = (T_oa - T_sp) + alpha_sa*I_sol ;  dT_oa = T_oa - T_sp
    G_s = 1/(R_zw+R_wo) ,  G_inf = 1/R_inf     (only these are reported — never the R split)

Targets:
    1a  true Q_cw,                 no intercept   -> absolute ceiling
    1b  measured - 3 kW (bias off), no intercept  -> ceiling with clean bias removal
    1c  measured (biased+noisy),   intercept b    -> practical case
    1d  true Q_cw,                 intercept b    -> does b distort a clean fit?

Stable LS only (lstsq; covariance via SVD). Read-only. Run from src/:  python experiment_1.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.optimize import minimize
from model import BuildingParams
from analysis_common import provenance

p = BuildingParams()
GS_T = 1.0 / (p.R_zw + p.R_wo)      # G_s*  = 307.69 W/K
GI_T = 1.0 / p.R_inf                # G_inf* = 2857.14 W/K
RS_T = p.R_zw + p.R_wo
BIAS_TRUE = 3000.0
DATA = "../data/sensing_S1.csv"
LO, HI = 0.1, 5.0                   # conductance box, x truth (same as variants)


def ols(y, X):
    """Unbounded OLS via lstsq + stable SVD covariance. Returns beta, cov, cond."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = (r @ r) / dof
    cov = sigma2 * (Vt.T * (1.0 / S**2)) @ Vt
    return beta, cov, S[0] / S[-1]


def clipped_bounded(y, dT_sa, dT_oa, Qint, Swin):
    """Bounded solve of the TRUE clipped objective. Box matches the pipeline: R_s and R_inf
    in [0.1,5]x truth (variants bound each resistance there), i.e. G in [GS*/5, GS*/0.1]."""
    def loss(g):
        z = Qint + Swin + dT_sa * g[0] + dT_oa * g[1]
        return np.mean((np.maximum(z, 0.0) - y) ** 2) / np.mean(y) ** 2
    # R in [0.1,5]x truth  ->  G in [G*/5, G*/0.1]
    bnds = [(GS_T / HI, GS_T / LO), (GI_T / HI, GI_T / LO)]
    starts = np.geomspace(GS_T / HI, GS_T / LO, 5)
    starts_i = np.geomspace(GI_T / HI, GI_T / LO, 5)
    best = None
    for gs in starts:
        for gi in starts_i:
            r = minimize(loss, [gs, gi], bounds=bnds, method="L-BFGS-B")
            if best is None or r.fun < best.fun:
                best = r
    at = (np.isclose(best.x[0], bnds[0][0], rtol=2e-3) or np.isclose(best.x[0], bnds[0][1], rtol=2e-3)
          or np.isclose(best.x[1], bnds[1][0], rtol=2e-3) or np.isclose(best.x[1], bnds[1][1], rtol=2e-3))
    return best.x, at


if __name__ == "__main__":
    print(provenance(DATA))
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    tr = df[df.split == "train"].copy()

    I     = tr.I_sol_Wm2.to_numpy(float)
    dT_oa = tr.T_oa_C.to_numpy(float) - p.T_z_sp
    dT_sa = dT_oa + p.alpha_sa * I
    Swin  = p.alpha_win * p.A_eff_win * I
    Qint  = tr.Q_int_W.to_numpy(float)
    Qtrue = tr.Q_cw_W.to_numpy(float)
    Qmeas = tr.Q_cw_meas_W.to_numpy(float)

    print(f"\nGround truth: G_s* = {GS_T:.2f} W/K   G_inf* = {GI_T:.2f} W/K   true bias = {BIAS_TRUE:.0f} W")
    print(f"Operator: Q_cw = max(Q_int + Swin + dT_sa*G_s + dT_oa*G_inf [+b], 0)   (reduced wall)\n")

    targets = [
        ("1a true, no b     ", Qtrue, False),
        ("1b meas-3kW, no b  ", Qmeas - BIAS_TRUE, False),
        ("1c meas, +b        ", Qmeas, True),
        ("1d true, +b        ", Qtrue, True),
    ]

    hdr = (f"{'target':20s} {'G_s/G*':>8s} {'G_inf/G*':>9s} {'R_s/R*':>8s} {'R_inf/R*':>9s} "
           f"{'b [W]':>9s} {'cond(X)':>8s} {'corr':>7s} {'n':>6s}  clip")
    print("="*len(hdr)); print("UNBOUNDED OLS  (the affine data optimum — shows operator displacement)")
    print("="*len(hdr)); print(hdr); print("-"*len(hdr))

    for name, ytar, use_b in targets:
        m = np.isfinite(ytar)
        y = (ytar - Qint - Swin)[m]
        cols = [dT_sa[m], dT_oa[m]] + ([np.ones(m.sum())] if use_b else [])
        X = np.column_stack(cols)
        beta, cov, cond = ols(y, X)
        gs, gi = beta[0], beta[1]
        b = beta[2] if use_b else 0.0
        corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        # clip diagnostic at this solution
        z = Qint[m] + Swin[m] + dT_sa[m]*gs + dT_oa[m]*gi + b
        nclip = int((z <= 0).sum())
        print(f"{name:20s} {gs/GS_T:8.3f} {gi/GI_T:9.3f} {GS_T/gs if gs!=0 else np.nan:8.3f} "
              f"{GI_T/gi:9.3f} {b:9.0f} {cond:8.1f} {corr:+7.3f} {m.sum():6d}  {nclip}")

    print(f"\n  DISPLACEMENT under IDEAL data (1a, clean+noiseless): read row 1a above.")
    print(f"  If G_s/G* is not ~1.0 there, the hourly static operator's own optimum is displaced")
    print(f"  from truth even with perfect data -> that displacement IS the ceiling.\n")

    print("="*72); print("BOUNDED CLIPPED SOLVE  (R_s,R_inf in [0.1,5]x truth) — 1a & 1b"); print("="*72)
    for name, ytar, _ in targets[:2]:
        m = np.isfinite(ytar); y = ytar[m]
        g, at = clipped_bounded(y, dT_sa[m], dT_oa[m], Qint[m], Swin[m])
        print(f"{name:20s} G_s/G* {g[0]/GS_T:6.3f}  G_inf/G* {g[1]/GI_T:6.3f}  "
              f"R_s/R* {GS_T/g[0]:6.3f}  R_inf/R* {GI_T/g[1]:6.3f}   "
              f"{'*** AT BOUND ***' if at else 'interior'}")
    print("\n  (If the bounded solve sits AT BOUND, the reported R_s/R_inf is a bound, not an estimate.)")
