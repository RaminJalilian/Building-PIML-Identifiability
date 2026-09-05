"""
Six constraint variants for the Building-PIML-Identifiability ablation study.

Each variant is a self-contained class that:
  1. Takes a sensing configuration DataFrame (from simulate.py)
  2. Trains on the training split
  3. Returns predictions and recovered RC parameters on the test split

Variants
--------
V1  Black-box NN
    Pure data-driven: MLP maps weather/time features → Q_cw.
    No physics whatsoever.

V2  Oracle-parameter reference
    Same MLP learns Q_int(t); Q_cw is computed via the reduced quasi-static
    operator with RC parameters FIXED AT TRUTH. The only regularizer is a Q_int
    smoothness penalty (0.01*mean(diff(Q_int)^2)); there is NO ODE-residual loss.
    Not a deployable inverse method -- it is the oracle-parameter reference showing
    the accuracy attainable when the physics parameters are known exactly.

V3  Hard architectural constraint (explicit RC layer, no NN forcing)
    RC parameters are learnable scalars with positivity bounds enforced
    via softplus.  Q_int is a fixed sinusoidal schedule (wrong assumption).
    Tests whether architectural constraints alone are sufficient.

V4  Neural forcing + correct bounds  [proposed method]
    Learnable RC parameters (softplus-bounded) + NN learns Q_int(t).
    Quasi-static training formulation.

V4F Neural forcing + known forcing function  [oracle forcing]
    Same as V4 but Q_int is fixed to the ground-truth schedule.
    Only the three resistances are learned. Isolates whether RC
    resistances can be recovered when forcing is independently known.

V5  Neural forcing + mis-specified constraints  [wrong physics]
    Same as V4 but with deliberately wrong RC bounds. Primary mis-specification:
    R_inf bounded far too small (1e-6 to 0.1x truth), which makes the infiltration
    CONDUCTANCE 1/R_inf excessively large -- an excessive infiltration heat flow,
    not a removed pathway (Q_inf = (T_oa-T_sp)/R_inf grows as R_inf shrinks). C_z
    bound is also wrong but does not affect the quasi-static training loss. Tests
    whether mis-specified physical constraints are more damaging than no physics.

All variants use the same:
  - ForcingNet architecture (where applicable)
  - Training loop (Adam, cosine LR, early stopping)
  - Evaluation metrics
  - Quasi-static observation model for training (where applicable)

Note: V4D (dynamic ODE training) is also implemented below but is NOT
included in the paper benchmark grid. It is retained as experimental
code for future work on dynamic identifiability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model import BuildingParams

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ── Ground-truth parameter values (for recovery error computation) ────────────
GT = BuildingParams()
GT_RC = {
    "C_z":   GT.C_z,
    "C_w":   GT.C_w,
    "R_zw":  GT.R_zw,
    "R_wo":  GT.R_wo,
    "R_inf": GT.R_inf,
}

# ── Fixed physics constants ───────────────────────────────────────────────────
ALPHA_WIN  = GT.alpha_win
ALPHA_SA   = GT.alpha_sa
T_SP       = GT.T_z_sp
K_T_FIXED  = GT.K_T        # used in quasi-static model
Q_MAX      = GT.Q_max
A_EFF_WIN  = GT.A_eff_win

# ── Training hyperparameters ──────────────────────────────────────────────────
N_EPOCHS  = 600
LR_START  = 2e-3
LR_MIN    = 1e-5
PATIENCE  = 80
GRAD_CLIP = 1.0


# ============================================================================
# Shared utilities
# ============================================================================

def metrics(pred: np.ndarray, meas: np.ndarray) -> dict:
    """CV-RMSE, NMBE, R², MAE (kW) on finite pairs."""
    mask = np.isfinite(pred) & np.isfinite(meas)
    if mask.sum() < 2:
        return dict(cv_rmse=np.nan, nmbe=np.nan, r2=np.nan, mae_kW=np.nan, n=0)
    p, m   = pred[mask], meas[mask]
    resid  = p - m
    mu     = m.mean()
    rmse   = np.sqrt((resid**2).mean())
    ss_res = (resid**2).sum()
    ss_tot = ((m - mu)**2).sum()
    return dict(
        cv_rmse = rmse / max(mu, 1e-8) * 100.0,
        nmbe    = resid.mean() / max(mu, 1e-8) * 100.0,
        r2      = 1.0 - ss_res / max(ss_tot, 1e-8),
        mae_kW  = float(np.abs(resid).mean() / 1e3),
        n       = int(mask.sum()),
    )


def param_recovery_error(recovered: dict) -> dict:
    """Relative error |recovered - truth| / truth for each RC parameter."""
    errors = {}
    for k, gt_val in GT_RC.items():
        rec_val = recovered.get(k, np.nan)
        errors[f"err_{k}"] = abs(rec_val - gt_val) / abs(gt_val) * 100.0
    return errors


def build_features(df: pd.DataFrame,
                   T_oa_min: float, T_oa_range: float) -> torch.Tensor:
    """6 features: sin/cos hour, sin/cos dow, T_oa_norm, I_sol_norm."""
    ts   = pd.DatetimeIndex(df["timestamp"])
    h    = ts.hour.to_numpy(dtype=np.float32)
    dow  = ts.dayofweek.to_numpy(dtype=np.float32)
    T_oa = df["T_oa_C"].to_numpy(dtype=np.float32)
    I_sol = df["I_sol_Wm2"].to_numpy(dtype=np.float32)
    feat = np.stack([
        np.sin(2 * np.pi * h   / 24.0),
        np.cos(2 * np.pi * h   / 24.0),
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
        (T_oa  - T_oa_min)  / (T_oa_range  + 1e-8),
        I_sol  / 1000.0,
    ], axis=1)
    return torch.tensor(feat, dtype=torch.float32, device=DEVICE)


def quasi_static_Q_cw(Q_int: torch.Tensor,
                      T_oa: np.ndarray,
                      I_sol: np.ndarray,
                      rc: dict) -> torch.Tensor:
    """Quasi-static Q_cw assuming T_z = T_sp.

    Q_cw_qs = clamp(Q_int + Q_ext_qs, 0)
    where Q_ext_qs = solar + infiltration + wall conduction (all at T_sp).

    RC values in `rc` may be torch.Tensor (with grad) or plain float.
    All arithmetic stays in torch so gradients flow to learnable RC params.
    """
    T_oa_t  = torch.tensor(T_oa,  dtype=torch.float32, device=DEVICE)
    I_sol_t = torch.tensor(I_sol, dtype=torch.float32, device=DEVICE)

    def _t(v):
        """Ensure value is a scalar tensor on DEVICE."""
        if isinstance(v, torch.Tensor):
            return v.squeeze()
        return torch.tensor(float(v), dtype=torch.float32, device=DEVICE)

    R_zw    = _t(rc["R_zw"])
    R_wo    = _t(rc["R_wo"])
    R_inf   = _t(rc["R_inf"])
    A_eff   = _t(rc["A_eff_win"])

    T_sa    = T_oa_t + ALPHA_SA * I_sol_t
    inv_zw  = 1.0 / R_zw
    inv_wo  = 1.0 / R_wo
    T_w_qs  = (T_SP * inv_zw + T_sa * inv_wo) / (inv_zw + inv_wo)

    Q_sol   = ALPHA_WIN * A_eff * I_sol_t
    Q_inf   = (T_oa_t - T_SP) / R_inf
    Q_wall  = (T_w_qs - T_SP) / R_zw

    Q_ext   = Q_sol + Q_inf + Q_wall
    return torch.clamp(Q_int + Q_ext, min=0.0)


def softplus_param(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Map unconstrained raw → (lo, hi) via softplus."""
    return lo + (hi - lo) * torch.sigmoid(raw)


# ============================================================================
# Shared MLP (ForcingNet)
# ============================================================================

class ForcingNet(nn.Module):
    """Maps weather/calendar features → Q_int(t) ∈ [0, Q_int_max]."""

    def __init__(self, q_int_max: float, n_features: int = 6, hidden: int = 64):
        super().__init__()
        self.q_int_max = float(q_int_max)
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_int_max * torch.sigmoid(self.net(x)).squeeze(-1)


class BlackBoxNet(nn.Module):
    """Direct Q_cw predictor: features → Q_cw (no physics)."""

    def __init__(self, q_cw_max: float, n_features: int = 6, hidden: int = 64):
        super().__init__()
        self.q_cw_max = float(q_cw_max)
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_cw_max * torch.sigmoid(self.net(x)).squeeze(-1)


# ============================================================================
# Result container
# ============================================================================

@dataclass
class VariantResult:
    variant:        str
    sensing:        str
    noise_level:    float
    metrics_train:  dict
    metrics_test:   dict
    param_errors:   dict          # relative RC recovery errors
    recovered_rc:   dict          # actual recovered values
    train_losses:   list
    train_time_s:   float
    epochs_trained: int
    q_int_pred:     Optional[np.ndarray] = None   # test-period Q_int (if available)
    q_cw_pred:      Optional[np.ndarray] = None   # test-period Q_cw prediction


# ============================================================================
# Training loop (shared)
# ============================================================================

def _train_loop(model: nn.Module,
                loss_fn,
                n_epochs: int = N_EPOCHS,
                lr: float = LR_START,
                patience: int = PATIENCE) -> tuple[list, float, int]:
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=LR_MIN
    )
    best_loss  = np.inf
    best_state = None
    patience_ct = 0
    losses = []
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()

        lv = loss.item()
        losses.append(lv)

        if lv < best_loss:
            best_loss   = lv
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return losses, time.time() - t0, len(losses)


# ============================================================================
# V1 — Black-box NN
# ============================================================================

def run_V1(df_train: pd.DataFrame,
           df_test:  pd.DataFrame,
           noise_level: float,
           sensing: str) -> VariantResult:
    """Pure black-box MLP: features → Q_cw.  No physics."""
    q_cw_max = float(df_train["Q_cw_meas_W"].dropna().max()) * 1.1

    T_oa_min   = float(df_train["T_oa_C"].min())
    T_oa_range = float(df_train["T_oa_C"].max() - df_train["T_oa_C"].min())

    feats_tr = build_features(df_train, T_oa_min, T_oa_range)
    feats_te = build_features(df_test,  T_oa_min, T_oa_range)

    Q_cw_tr  = torch.tensor(df_train["Q_cw_meas_W"].fillna(0).to_numpy(dtype=np.float32),
                             device=DEVICE)
    mask_tr  = torch.tensor(df_train["Q_cw_meas_W"].notna().to_numpy(), device=DEVICE)
    mu_tr    = Q_cw_tr[mask_tr].mean()

    model = BlackBoxNet(q_cw_max=q_cw_max).to(DEVICE)

    def loss_fn(m):
        pred = m(feats_tr)
        return torch.mean((pred[mask_tr] - Q_cw_tr[mask_tr])**2) / (mu_tr**2 + 1e-8)

    losses, t_train, epochs = _train_loop(model, loss_fn)

    model.eval()
    with torch.no_grad():
        q_cw_pred_tr = model(feats_tr).cpu().numpy()
        q_cw_pred_te = model(feats_te).cpu().numpy()

    return VariantResult(
        variant="V1_blackbox",
        sensing=sensing,
        noise_level=noise_level,
        metrics_train  = metrics(q_cw_pred_tr, df_train["Q_cw_meas_W"].to_numpy()),
        metrics_test   = metrics(q_cw_pred_te, df_test["Q_cw_meas_W"].to_numpy()),
        param_errors   = {f"err_{k}": np.nan for k in GT_RC},
        recovered_rc   = {},
        train_losses   = losses,
        train_time_s   = t_train,
        epochs_trained = epochs,
        q_int_pred     = None,
        q_cw_pred      = q_cw_pred_te,
    )


# ============================================================================
# V2 — Soft physics (ODE residual regularizer)
# ============================================================================

def run_V2(df_train: pd.DataFrame,
           df_test:  pd.DataFrame,
           noise_level: float,
           sensing: str) -> VariantResult:
    """Oracle-parameter reference: NN predicts Q_int; Q_cw via the reduced quasi-static
    operator with RC parameters FIXED AT TRUTH. The only regularizer is a Q_int
    smoothness penalty -- there is NO ODE-residual loss."""
    q_int_max  = 80_000.0
    T_oa_min   = float(df_train["T_oa_C"].min())
    T_oa_range = float(df_train["T_oa_C"].max() - df_train["T_oa_C"].min())

    feats_tr = build_features(df_train, T_oa_min, T_oa_range)
    feats_te = build_features(df_test,  T_oa_min, T_oa_range)

    T_oa_tr  = df_train["T_oa_C"].to_numpy(dtype=np.float32)
    I_sol_tr = df_train["I_sol_Wm2"].to_numpy(dtype=np.float32)
    Q_cw_tr  = torch.tensor(df_train["Q_cw_meas_W"].fillna(0).to_numpy(dtype=np.float32),
                             device=DEVICE)
    mask_tr  = torch.tensor(df_train["Q_cw_meas_W"].notna().to_numpy(), device=DEVICE)
    mu_tr    = Q_cw_tr[mask_tr].mean()

    # RC parameters FIXED AT TRUTH (oracle-parameter reference; not RC recovery)
    rc_fixed = {
        "C_z": GT.C_z, "C_w": GT.C_w,
        "R_zw": GT.R_zw, "R_wo": GT.R_wo, "R_inf": GT.R_inf,
        "A_eff_win": GT.A_eff_win,
    }

    model = ForcingNet(q_int_max=q_int_max).to(DEVICE)
    W_PHYS = 0.01   # soft physics weight

    def loss_fn(m):
        Q_int = m(feats_tr)
        Q_cw_qs = quasi_static_Q_cw(Q_int, T_oa_tr, I_sol_tr, rc_fixed)

        L_data = torch.mean((Q_cw_qs[mask_tr] - Q_cw_tr[mask_tr])**2) / (mu_tr**2 + 1e-8)

        # Soft ODE residual: dQ_int/dt should be smooth (proxy physics penalty)
        L_phys = torch.mean(torch.diff(Q_int)**2) / (q_int_max**2 + 1e-8)

        return L_data + W_PHYS * L_phys

    losses, t_train, epochs = _train_loop(model, loss_fn)

    model.eval()
    with torch.no_grad():
        Q_int_te = model(feats_te)
        T_oa_te  = df_test["T_oa_C"].to_numpy(dtype=np.float32)
        I_sol_te = df_test["I_sol_Wm2"].to_numpy(dtype=np.float32)
        q_cw_qs_te = quasi_static_Q_cw(Q_int_te, T_oa_te, I_sol_te, rc_fixed)
        q_cw_pred_te = q_cw_qs_te.cpu().numpy()

        Q_int_tr = model(feats_tr)
        q_cw_qs_tr = quasi_static_Q_cw(Q_int_tr, T_oa_tr, I_sol_tr, rc_fixed)
        q_cw_pred_tr = q_cw_qs_tr.cpu().numpy()

    return VariantResult(
        variant="V2_soft_physics",
        sensing=sensing,
        noise_level=noise_level,
        metrics_train  = metrics(q_cw_pred_tr, df_train["Q_cw_meas_W"].to_numpy()),
        metrics_test   = metrics(q_cw_pred_te, df_test["Q_cw_meas_W"].to_numpy()),
        param_errors   = {f"err_{k}": np.nan for k in GT_RC},
        recovered_rc   = {},
        train_losses   = losses,
        train_time_s   = t_train,
        epochs_trained = epochs,
        q_int_pred     = Q_int_te.cpu().numpy(),
        q_cw_pred      = q_cw_pred_te,
    )


# ============================================================================
# V3 — Hard architectural constraint, fixed sinusoidal Q_int
# ============================================================================

def run_V3(df_train: pd.DataFrame,
           df_test:  pd.DataFrame,
           noise_level: float,
           sensing: str) -> VariantResult:
    """Learnable RC parameters (softplus-bounded) + fixed sinusoidal Q_int.

    Tests whether architectural constraints alone (without a learned forcing)
    are sufficient for RC parameter recovery.
    """
    T_oa_tr  = df_train["T_oa_C"].to_numpy(dtype=np.float32)
    I_sol_tr = df_train["I_sol_Wm2"].to_numpy(dtype=np.float32)
    T_oa_te  = df_test["T_oa_C"].to_numpy(dtype=np.float32)
    I_sol_te = df_test["I_sol_Wm2"].to_numpy(dtype=np.float32)

    Q_cw_tr  = torch.tensor(df_train["Q_cw_meas_W"].fillna(0).to_numpy(dtype=np.float32),
                             device=DEVICE)
    mask_tr  = torch.tensor(df_train["Q_cw_meas_W"].notna().to_numpy(), device=DEVICE)
    mu_tr    = Q_cw_tr[mask_tr].mean()

    # Fixed sinusoidal Q_int (wrong assumption — no NN)
    ts_tr = pd.DatetimeIndex(df_train["timestamp"])
    ts_te = pd.DatetimeIndex(df_test["timestamp"])
    hod_tr = ts_tr.hour.to_numpy(dtype=np.float32)
    hod_te = ts_te.hour.to_numpy(dtype=np.float32)
    Q_int_fixed_tr = torch.tensor(
        25000.0 * (0.5 + 0.5 * np.sin(2 * np.pi * hod_tr / 24.0 - np.pi / 2)),
        dtype=torch.float32, device=DEVICE
    )
    Q_int_fixed_te = torch.tensor(
        25000.0 * (0.5 + 0.5 * np.sin(2 * np.pi * hod_te / 24.0 - np.pi / 2)),
        dtype=torch.float32, device=DEVICE
    )

    # Learnable RC parameters in unconstrained space
    # Bounds: capacitances 0.2x-3.0x truth; resistances 0.1x-5.0x truth
    # (sigmoid-midpoint initialisation therefore starts C at 1.6x and R at 2.55x truth)
    bounds = {
        "C_z":   (GT.C_z  * 0.2, GT.C_z  * 3.0),
        "C_w":   (GT.C_w  * 0.2, GT.C_w  * 3.0),
        "R_zw":  (GT.R_zw * 0.1, GT.R_zw * 5.0),
        "R_wo":  (GT.R_wo * 0.1, GT.R_wo * 5.0),
        "R_inf": (GT.R_inf* 0.1, GT.R_inf* 5.0),
    }
    raw = nn.ParameterDict({
        k: nn.Parameter(torch.zeros(1, device=DEVICE)) for k in bounds
    })

    def get_rc(raw_params):
        rc = {}
        for k, (lo, hi) in bounds.items():
            rc[k] = softplus_param(raw_params[k], lo, hi).squeeze()
        rc["A_eff_win"] = torch.tensor(A_EFF_WIN, device=DEVICE)
        return rc

    opt   = torch.optim.Adam(raw.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=LR_MIN)
    best_loss, best_state, patience_ct = np.inf, None, 0
    losses = []
    t0 = time.time()

    for epoch in range(N_EPOCHS):
        raw.train()
        opt.zero_grad()
        rc = get_rc(raw)   # dict of tensors — gradients intact
        Q_cw_qs = quasi_static_Q_cw(Q_int_fixed_tr, T_oa_tr, I_sol_tr, rc)
        loss = torch.mean((Q_cw_qs[mask_tr] - Q_cw_tr[mask_tr])**2) / (mu_tr**2 + 1e-8)
        loss.backward()
        opt.step(); sched.step()
        lv = loss.item(); losses.append(lv)
        if lv < best_loss:
            best_loss = lv
            best_state = {k: v.clone() for k, v in raw.state_dict().items()}
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE: break

    if best_state: raw.load_state_dict(best_state)

    with torch.no_grad():
        rc_final = get_rc(raw)
        Q_cw_qs_tr = quasi_static_Q_cw(Q_int_fixed_tr, T_oa_tr, I_sol_tr, rc_final)
        Q_cw_qs_te = quasi_static_Q_cw(Q_int_fixed_te, T_oa_te, I_sol_te, rc_final)

    recovered = {k: rc_final[k].item() for k in GT_RC}

    return VariantResult(
        variant="V3_hard_arch",
        sensing=sensing,
        noise_level=noise_level,
        metrics_train  = metrics(Q_cw_qs_tr.cpu().numpy(), df_train["Q_cw_meas_W"].to_numpy()),
        metrics_test   = metrics(Q_cw_qs_te.cpu().numpy(), df_test["Q_cw_meas_W"].to_numpy()),
        param_errors   = param_recovery_error(recovered),
        recovered_rc   = recovered,
        train_losses   = losses,
        train_time_s   = time.time() - t0,
        epochs_trained = len(losses),
        q_int_pred     = None,
        q_cw_pred      = Q_cw_qs_te.cpu().numpy(),
    )


# ============================================================================
# V4 — Neural forcing + correct bounds  [proposed]
# ============================================================================

def run_V4(df_train: pd.DataFrame,
           df_test:  pd.DataFrame,
           noise_level: float,
           sensing: str) -> VariantResult:
    """Learnable resistance parameters + ForcingNet learns Q_int(t).

    Capacitances C_z and C_w are fixed at physics-derived priors because
    they are NOT identifiable from quasi-static Q_cw observations alone —
    storage terms only appear in the dynamic (time-derivative) terms of the
    ODE, which vanish at quasi-steady state.  This is a documented design
    choice and a key finding of the paper.

    Only identifiable COMBINATIONS enter Q_ext_qs, not the three resistances
    individually. Algebraically the wall term reduces to
    Q_wall = (T_sa - T_sp)/(R_zw + R_wo), so R_zw and R_wo appear only through
    their series sum R_s = R_zw + R_wo and can never be separated by this
    objective. The identifiable quantities are therefore R_s (equivalently
    G_s = 1/R_s) and R_inf (equivalently G_inf = 1/R_inf); the R_zw/R_wo split
    reported per-run is prior/initialisation-selected, not data-identified.
    """
    q_int_max  = 80_000.0
    T_oa_min   = float(df_train["T_oa_C"].min())
    T_oa_range = float(df_train["T_oa_C"].max() - df_train["T_oa_C"].min())

    feats_tr = build_features(df_train, T_oa_min, T_oa_range)
    feats_te = build_features(df_test,  T_oa_min, T_oa_range)

    T_oa_tr  = df_train["T_oa_C"].to_numpy(dtype=np.float32)
    I_sol_tr = df_train["I_sol_Wm2"].to_numpy(dtype=np.float32)
    T_oa_te  = df_test["T_oa_C"].to_numpy(dtype=np.float32)
    I_sol_te = df_test["I_sol_Wm2"].to_numpy(dtype=np.float32)

    Q_cw_tr  = torch.tensor(df_train["Q_cw_meas_W"].fillna(0).to_numpy(dtype=np.float32),
                             device=DEVICE)
    mask_tr  = torch.tensor(df_train["Q_cw_meas_W"].notna().to_numpy(), device=DEVICE)
    mu_tr    = Q_cw_tr[mask_tr].mean()

    # Only resistances are learnable — capacitances fixed at physics prior
    bounds = {
        "R_zw":  (GT.R_zw * 0.1, GT.R_zw * 5.0),
        "R_wo":  (GT.R_wo * 0.1, GT.R_wo * 5.0),
        "R_inf": (GT.R_inf* 0.1, GT.R_inf* 5.0),
    }
    prior_centers = {k: torch.tensor((lo + hi) / 2.0, dtype=torch.float32, device=DEVICE)
                     for k, (lo, hi) in bounds.items()}

    forcing_net = ForcingNet(q_int_max=q_int_max).to(DEVICE)
    raw_rc = nn.ParameterDict({
        k: nn.Parameter(torch.zeros(1, device=DEVICE)) for k in bounds
    })

    all_params = list(forcing_net.parameters()) + list(raw_rc.parameters())
    opt   = torch.optim.Adam(all_params, lr=LR_START)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=LR_MIN)
    best_loss, best_state, patience_ct = np.inf, None, 0
    losses = []
    t0 = time.time()

    # T_z observations (available for S2, S3)
    T_z_obs_tr = df_train["T_z_obs"].to_numpy(dtype=np.float32) \
        if "T_z_obs" in df_train.columns else None
    has_Tz = T_z_obs_tr is not None and np.isfinite(T_z_obs_tr).any()
    if has_Tz:
        T_z_obs_t  = torch.tensor(np.nan_to_num(T_z_obs_tr, nan=T_SP),
                                  dtype=torch.float32, device=DEVICE)
        mask_Tz_tr = torch.tensor(np.isfinite(T_z_obs_tr), device=DEVICE)
        mu_Tz      = T_z_obs_t[mask_Tz_tr].mean()
        W_TZ       = 1.0
    else:
        W_TZ = 0.0

    W_PRIOR = 0.005

    for epoch in range(N_EPOCHS):
        forcing_net.train()
        opt.zero_grad()

        rc_t = {}
        for k, (lo, hi) in bounds.items():
            rc_t[k] = softplus_param(raw_rc[k], lo, hi).squeeze()
        # Fixed capacitances — not learnable
        rc_t["C_z"]      = torch.tensor(GT.C_z,      dtype=torch.float32, device=DEVICE)
        rc_t["C_w"]      = torch.tensor(GT.C_w,      dtype=torch.float32, device=DEVICE)
        rc_t["A_eff_win"] = torch.tensor(A_EFF_WIN,  dtype=torch.float32, device=DEVICE)

        Q_int    = forcing_net(feats_tr)
        Q_cw_qs  = quasi_static_Q_cw(Q_int, T_oa_tr, I_sol_tr, rc_t)
        L_data   = torch.mean((Q_cw_qs[mask_tr] - Q_cw_tr[mask_tr])**2) / (mu_tr**2 + 1e-8)
        L_smooth = torch.mean(torch.diff(Q_int)**2) / (q_int_max**2 + 1e-8)
        L_prior  = sum(((rc_t[k] - prior_centers[k]) / prior_centers[k])**2
                       for k in bounds) / len(bounds)

        if has_Tz and W_TZ > 0:
            T_z_pred = T_SP + Q_int / K_T_FIXED
            L_tz = torch.mean((T_z_pred[mask_Tz_tr] - T_z_obs_t[mask_Tz_tr])**2) \
                   / (mu_Tz**2 + 1e-8)
        else:
            L_tz = torch.tensor(0.0, device=DEVICE)

        loss = L_data + 0.005 * L_smooth + W_PRIOR * L_prior + W_TZ * L_tz
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)
        opt.step(); sched.step()

        lv = loss.item(); losses.append(lv)
        if lv < best_loss:
            best_loss = lv
            best_state = {
                "forcing": {k: v.clone() for k, v in forcing_net.state_dict().items()},
                "rc":      {k: v.clone() for k, v in raw_rc.state_dict().items()},
            }
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE: break

    if best_state:
        forcing_net.load_state_dict(best_state["forcing"])
        raw_rc.load_state_dict(best_state["rc"])

    with torch.no_grad():
        rc_final = {}
        for k, (lo, hi) in bounds.items():
            rc_final[k] = softplus_param(raw_rc[k], lo, hi).squeeze()
        rc_final["A_eff_win"] = torch.tensor(A_EFF_WIN, device=DEVICE)
        rc_final["C_z"] = torch.tensor(GT.C_z, device=DEVICE)
        rc_final["C_w"] = torch.tensor(GT.C_w, device=DEVICE)

        Q_int_te   = forcing_net(feats_te)
        Q_cw_qs_te = quasi_static_Q_cw(Q_int_te, T_oa_te, I_sol_te, rc_final)
        Q_int_tr_  = forcing_net(feats_tr)
        Q_cw_qs_tr = quasi_static_Q_cw(Q_int_tr_, T_oa_tr, I_sol_tr, rc_final)

    # Report recovery for the three learnable resistances
    recovered = {k: rc_final[k].item() for k in ["R_zw", "R_wo", "R_inf"]}
    # Capacitances reported as fixed-at-prior (0% error by construction)
    recovered["C_z"] = GT.C_z
    recovered["C_w"] = GT.C_w

    return VariantResult(
        variant="V4_neural_forcing",
        sensing=sensing,
        noise_level=noise_level,
        metrics_train  = metrics(Q_cw_qs_tr.cpu().numpy(), df_train["Q_cw_meas_W"].to_numpy()),
        metrics_test   = metrics(Q_cw_qs_te.cpu().numpy(), df_test["Q_cw_meas_W"].to_numpy()),
        param_errors   = param_recovery_error(recovered),
        recovered_rc   = recovered,
        train_losses   = losses,
        train_time_s   = time.time() - t0,
        epochs_trained = len(losses),
        q_int_pred     = Q_int_te.cpu().numpy(),
        q_cw_pred      = Q_cw_qs_te.cpu().numpy(),
    )


# ============================================================================
# V5 — Neural forcing + misspecified constraints  [wrong physics]
# ============================================================================

def run_V5(df_train: pd.DataFrame,
           df_test:  pd.DataFrame,
           noise_level: float,
           sensing: str) -> VariantResult:
    """Same as V4 but with deliberately mis-specified RC parameter bounds.

    Misspecifications:
      - R_inf bounded far too small (1e-6 to 0.1x truth), so the infiltration
        CONDUCTANCE 1/R_inf is excessively large -- an excessive infiltration heat
        flow (Q_inf = (T_oa-T_sp)/R_inf), NOT a removed pathway (primary mis-spec)
      - C_z upper bound set too low  (true C_z unreachable; does not affect the
        quasi-static loss since C_z does not enter the observation equation)
      - R_zw, R_wo bounds narrowed relative to V4

    Because C_z does not enter the quasi-static training loss, the wrong C_z
    bound does not affect the training outcome; it is included only for
    architectural consistency. The dominant mis-specification causing training
    collapse is the excessive infiltration conductance from the too-small R_inf,
    which forces the external cooling pathway into a structurally wrong regime.

    Tests whether wrong physical constraints are more damaging than no physics.
    """
    q_int_max  = 80_000.0
    T_oa_min   = float(df_train["T_oa_C"].min())
    T_oa_range = float(df_train["T_oa_C"].max() - df_train["T_oa_C"].min())

    feats_tr = build_features(df_train, T_oa_min, T_oa_range)
    feats_te = build_features(df_test,  T_oa_min, T_oa_range)

    T_oa_tr  = df_train["T_oa_C"].to_numpy(dtype=np.float32)
    I_sol_tr = df_train["I_sol_Wm2"].to_numpy(dtype=np.float32)
    T_oa_te  = df_test["T_oa_C"].to_numpy(dtype=np.float32)
    I_sol_te = df_test["I_sol_Wm2"].to_numpy(dtype=np.float32)

    Q_cw_tr  = torch.tensor(df_train["Q_cw_meas_W"].fillna(0).to_numpy(dtype=np.float32),
                             device=DEVICE)
    mask_tr  = torch.tensor(df_train["Q_cw_meas_W"].notna().to_numpy(), device=DEVICE)
    mu_tr    = Q_cw_tr[mask_tr].mean()

    # ── WRONG bounds ──────────────────────────────────────────────────────────
    bounds_wrong = {
        "C_z":   (GT.C_z  * 0.2, GT.C_z  * 0.6),   # true C_z unreachable
        "C_w":   (GT.C_w  * 0.2, GT.C_w  * 3.0),
        "R_zw":  (GT.R_zw * 0.1, GT.R_zw * 5.0),
        "R_wo":  (GT.R_wo * 0.1, GT.R_wo * 5.0),
        "R_inf": (1e-6,           GT.R_inf * 0.1),   # infiltration suppressed
    }

    forcing_net = ForcingNet(q_int_max=q_int_max).to(DEVICE)
    raw_rc = nn.ParameterDict({
        k: nn.Parameter(torch.zeros(1, device=DEVICE)) for k in bounds_wrong
    })

    all_params = list(forcing_net.parameters()) + list(raw_rc.parameters())
    opt   = torch.optim.Adam(all_params, lr=LR_START)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=LR_MIN)
    best_loss, best_state, patience_ct = np.inf, None, 0
    losses = []
    t0 = time.time()

    for epoch in range(N_EPOCHS):
        forcing_net.train()
        opt.zero_grad()

        rc_t = {}
        for k, (lo, hi) in bounds_wrong.items():
            rc_t[k] = softplus_param(raw_rc[k], lo, hi).squeeze()
        rc_t["A_eff_win"] = torch.tensor(A_EFF_WIN, device=DEVICE)

        Q_int    = forcing_net(feats_tr)
        Q_cw_qs  = quasi_static_Q_cw(Q_int, T_oa_tr, I_sol_tr, rc_t)
        L_data   = torch.mean((Q_cw_qs[mask_tr] - Q_cw_tr[mask_tr])**2) / (mu_tr**2 + 1e-8)
        L_smooth = torch.mean(torch.diff(Q_int)**2) / (q_int_max**2 + 1e-8)
        loss     = L_data + 0.005 * L_smooth

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)
        opt.step(); sched.step()

        lv = loss.item(); losses.append(lv)
        if lv < best_loss:
            best_loss = lv
            best_state = {
                "forcing": {k: v.clone() for k, v in forcing_net.state_dict().items()},
                "rc":      {k: v.clone() for k, v in raw_rc.state_dict().items()},
            }
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE: break

    if best_state:
        forcing_net.load_state_dict(best_state["forcing"])
        raw_rc.load_state_dict(best_state["rc"])

    with torch.no_grad():
        rc_final = {}
        for k, (lo, hi) in bounds_wrong.items():
            rc_final[k] = softplus_param(raw_rc[k], lo, hi).squeeze()
        rc_final["A_eff_win"] = torch.tensor(A_EFF_WIN, device=DEVICE)

        Q_int_te   = forcing_net(feats_te)
        Q_cw_qs_te = quasi_static_Q_cw(Q_int_te, T_oa_te, I_sol_te, rc_final)
        Q_int_tr_  = forcing_net(feats_tr)
        Q_cw_qs_tr = quasi_static_Q_cw(Q_int_tr_, T_oa_tr, I_sol_tr, rc_final)

    recovered = {k: rc_final[k].item() for k in GT_RC}

    return VariantResult(
        variant="V5_wrong_constraints",
        sensing=sensing,
        noise_level=noise_level,
        metrics_train  = metrics(Q_cw_qs_tr.cpu().numpy(), df_train["Q_cw_meas_W"].to_numpy()),
        metrics_test   = metrics(Q_cw_qs_te.cpu().numpy(), df_test["Q_cw_meas_W"].to_numpy()),
        param_errors   = param_recovery_error(recovered),
        recovered_rc   = recovered,
        train_losses   = losses,
        train_time_s   = time.time() - t0,
        epochs_trained = len(losses),
        q_int_pred     = Q_int_te.cpu().numpy(),
        q_cw_pred      = Q_cw_qs_te.cpu().numpy(),
    )


# ============================================================================
# V4D — Neural forcing + dynamic ODE training  [proposed, full version]
# ============================================================================

def implicit_step_2r2c(
    T_z: torch.Tensor,
    T_w: torch.Tensor,
    Q_int_i: torch.Tensor,
    T_oa_i: float,
    I_sol_i: float,
    rc: dict,
    dt: float = 3600.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One semi-implicit Euler step for the 2R2C thermal ODE.

    Solves a 2×2 linear system via Cramer's rule (unconditionally stable).
    RC parameters may be torch.Tensor with grad — gradients flow through.
    """
    C_z   = rc["C_z"];   C_w  = rc["C_w"]
    R_zw  = rc["R_zw"];  R_wo = rc["R_wo"];  R_inf = rc["R_inf"]
    A_eff = rc["A_eff_win"]
    K_T   = torch.tensor(GT.K_T,   dtype=torch.float32, device=DEVICE)

    T_sa       = T_oa_i + ALPHA_SA * I_sol_i
    Q_sol_zone = ALPHA_WIN * A_eff * I_sol_i
    Q_sol_wall = (1.0 - ALPHA_WIN) * A_eff * I_sol_i

    a11 = C_z / dt + 1.0 / R_zw + 1.0 / R_inf + K_T
    a12 = -1.0 / R_zw
    a22 = C_w / dt + 1.0 / R_zw + 1.0 / R_wo

    b1  = C_z / dt * T_z + T_oa_i / R_inf + Q_sol_zone + Q_int_i + K_T * T_SP
    b2  = C_w / dt * T_w + T_sa   / R_wo  + Q_sol_wall

    det     = a11 * a22 - a12 * a12
    T_z_new = (b1 * a22 - b2 * a12) / det
    T_w_new = (a11 * b2 - a12 * b1) / det
    return T_z_new, T_w_new


def integrate_2r2c(
    T_oa: np.ndarray,
    I_sol: np.ndarray,
    Q_int_vals: torch.Tensor,
    rc: dict,
    window_h: int = 168,
    dt: float = 3600.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate 2R2C ODE with multi-shooting (detach at window boundaries).

    Gradients flow within each window_h-step window; detach prevents
    gradient explosion across long horizons.
    """
    T = len(T_oa)

    # Quasi-static initial conditions
    R_zw = rc["R_zw"].item() if isinstance(rc["R_zw"], torch.Tensor) else rc["R_zw"]
    R_wo = rc["R_wo"].item()  if isinstance(rc["R_wo"],  torch.Tensor) else rc["R_wo"]
    T_sa_0  = float(T_oa[0]) + ALPHA_SA * float(I_sol[0])
    T_w_ic  = (T_SP / R_zw + T_sa_0 / R_wo) / (1.0 / R_zw + 1.0 / R_wo)

    T_z = torch.tensor(T_SP + 0.5,  dtype=torch.float32, device=DEVICE)
    T_w = torch.tensor(T_w_ic,       dtype=torch.float32, device=DEVICE)

    T_z_traj: list[torch.Tensor] = []
    T_w_traj: list[torch.Tensor] = []

    for i in range(T):
        T_z_traj.append(T_z)
        T_w_traj.append(T_w)
        if i < T - 1:
            T_z, T_w = implicit_step_2r2c(
                T_z, T_w, Q_int_vals[i],
                float(T_oa[i]), float(I_sol[i]), rc, dt,
            )
            if (i + 1) % window_h == 0:
                T_z = T_z.detach()
                T_w = T_w.detach()

    return torch.stack(T_z_traj), torch.stack(T_w_traj)


def run_V4D(df_train: pd.DataFrame,
            df_test:  pd.DataFrame,
            noise_level: float,
            sensing: str) -> VariantResult:
    """EXPERIMENTAL — NOT part of the paper benchmark grid; do not use scientifically.

    Neural forcing + ALL RC parameters learnable + dynamic ODE training. Unlike V4
    (quasi-static), this variant trains through the full ODE, which would in principle
    make capacitances C_z and C_w identifiable because they appear in the time-derivative
    terms.

    KNOWN LIMITATION (unfixed): the semi-implicit step in implicit_step_2r2c always includes
    the ACTIVE, uncapped proportional-controller term K_T(T_z - T_sp), whereas evaluation
    clips HVAC to [0, Q_max]. It therefore does not integrate the stated piecewise HVAC ODE
    when the controller is off or capacity-bound, and would need an active-set / piecewise
    implicit step before any scientific use. Retained only as a starting point for future
    dynamic-identifiability work.

    Loss = data fit on Q_cw (from ODE trajectory) + optional T_z fit + smoothness
    Uses multi-shooting with 168-h windows to keep gradients tractable.
    """
    q_int_max  = 80_000.0
    T_oa_min   = float(df_train["T_oa_C"].min())
    T_oa_range = float(df_train["T_oa_C"].max() - df_train["T_oa_C"].min())

    feats_tr = build_features(df_train, T_oa_min, T_oa_range)
    feats_te = build_features(df_test,  T_oa_min, T_oa_range)

    T_oa_tr  = df_train["T_oa_C"].to_numpy(dtype=np.float32)
    I_sol_tr = df_train["I_sol_Wm2"].to_numpy(dtype=np.float32)
    T_oa_te  = df_test["T_oa_C"].to_numpy(dtype=np.float32)
    I_sol_te = df_test["I_sol_Wm2"].to_numpy(dtype=np.float32)

    Q_cw_tr  = torch.tensor(df_train["Q_cw_meas_W"].fillna(0).to_numpy(dtype=np.float32),
                             device=DEVICE)
    mask_tr  = torch.tensor(df_train["Q_cw_meas_W"].notna().to_numpy(), device=DEVICE)
    mu_tr    = Q_cw_tr[mask_tr].mean()

    # All 5 RC parameters are learnable
    bounds = {
        "C_z":   (GT.C_z  * 0.2, GT.C_z  * 4.0),
        "C_w":   (GT.C_w  * 0.2, GT.C_w  * 4.0),
        "R_zw":  (GT.R_zw * 0.1, GT.R_zw * 5.0),
        "R_wo":  (GT.R_wo * 0.1, GT.R_wo * 5.0),
        "R_inf": (GT.R_inf* 0.1, GT.R_inf* 5.0),
    }
    prior_centers = {k: torch.tensor((lo + hi) / 2.0, dtype=torch.float32, device=DEVICE)
                     for k, (lo, hi) in bounds.items()}

    forcing_net = ForcingNet(q_int_max=q_int_max).to(DEVICE)
    raw_rc = nn.ParameterDict({
        k: nn.Parameter(torch.zeros(1, device=DEVICE)) for k in bounds
    })

    all_params = list(forcing_net.parameters()) + list(raw_rc.parameters())
    opt   = torch.optim.Adam(all_params, lr=5e-4)   # lower LR for ODE training
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-6)
    best_loss, best_state, patience_ct = np.inf, None, 0
    losses = []
    t0 = time.time()

    # T_z observations
    T_z_obs_tr = df_train["T_z_obs"].to_numpy(dtype=np.float32) \
        if "T_z_obs" in df_train.columns else None
    has_Tz = T_z_obs_tr is not None and np.isfinite(T_z_obs_tr).any()
    if has_Tz:
        T_z_obs_t  = torch.tensor(np.nan_to_num(T_z_obs_tr, nan=T_SP),
                                  dtype=torch.float32, device=DEVICE)
        mask_Tz_tr = torch.tensor(np.isfinite(T_z_obs_tr), device=DEVICE)
        mu_Tz      = T_z_obs_t[mask_Tz_tr].mean()
        W_TZ       = 2.0    # strong weight — T_z is the key signal for capacitances
    else:
        W_TZ = 0.0

    W_PRIOR = 0.01

    for epoch in range(N_EPOCHS):
        forcing_net.train()
        opt.zero_grad()

        rc_t = {}
        for k, (lo, hi) in bounds.items():
            rc_t[k] = softplus_param(raw_rc[k], lo, hi).squeeze()
        rc_t["A_eff_win"] = torch.tensor(A_EFF_WIN, dtype=torch.float32, device=DEVICE)

        Q_int = forcing_net(feats_tr)

        # Integrate ODE — gradients flow through T_z, T_w trajectories
        T_z_traj, _ = integrate_2r2c(T_oa_tr, I_sol_tr, Q_int, rc_t)

        # Q_cw from ODE trajectory
        K_T_t   = torch.tensor(GT.K_T, dtype=torch.float32, device=DEVICE)
        Q_hvac  = torch.clamp(K_T_t * (T_z_traj - T_SP), min=0.0,
                               max=torch.tensor(GT.Q_max, device=DEVICE))
        Q_cw_pred = Q_hvac

        L_data   = torch.mean((Q_cw_pred[mask_tr] - Q_cw_tr[mask_tr])**2) / (mu_tr**2 + 1e-8)
        L_smooth = torch.mean(torch.diff(Q_int)**2) / (q_int_max**2 + 1e-8)
        L_prior  = sum(((rc_t[k] - prior_centers[k]) / prior_centers[k])**2
                       for k in bounds) / len(bounds)

        if has_Tz and W_TZ > 0:
            L_tz = torch.mean((T_z_traj[mask_Tz_tr] - T_z_obs_t[mask_Tz_tr])**2) \
                   / (mu_Tz**2 + 1e-8)
        else:
            L_tz = torch.tensor(0.0, device=DEVICE)

        loss = L_data + 0.005 * L_smooth + W_PRIOR * L_prior + W_TZ * L_tz
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)
        opt.step(); sched.step()

        lv = loss.item(); losses.append(lv)
        if lv < best_loss:
            best_loss = lv
            best_state = {
                "forcing": {k: v.clone() for k, v in forcing_net.state_dict().items()},
                "rc":      {k: v.clone() for k, v in raw_rc.state_dict().items()},
            }
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE: break

    if best_state:
        forcing_net.load_state_dict(best_state["forcing"])
        raw_rc.load_state_dict(best_state["rc"])

    with torch.no_grad():
        rc_final = {}
        for k, (lo, hi) in bounds.items():
            rc_final[k] = softplus_param(raw_rc[k], lo, hi).squeeze()
        rc_final["A_eff_win"] = torch.tensor(A_EFF_WIN, device=DEVICE)

        Q_int_te    = forcing_net(feats_te)
        T_z_te, _   = integrate_2r2c(T_oa_te, I_sol_te, Q_int_te, rc_final)
        K_T_t       = torch.tensor(GT.K_T, dtype=torch.float32, device=DEVICE)
        Q_cw_qs_te  = torch.clamp(K_T_t * (T_z_te - T_SP), min=0.0,
                                   max=torch.tensor(GT.Q_max, device=DEVICE))
        Q_int_tr_   = forcing_net(feats_tr)
        T_z_tr, _   = integrate_2r2c(T_oa_tr, I_sol_tr, Q_int_tr_, rc_final)
        Q_cw_qs_tr  = torch.clamp(K_T_t * (T_z_tr - T_SP), min=0.0,
                                   max=torch.tensor(GT.Q_max, device=DEVICE))

    recovered = {k: rc_final[k].item() for k in GT_RC}

    return VariantResult(
        variant="V4D_dynamic",
        sensing=sensing,
        noise_level=noise_level,
        metrics_train  = metrics(Q_cw_qs_tr.cpu().numpy(), df_train["Q_cw_meas_W"].to_numpy()),
        metrics_test   = metrics(Q_cw_qs_te.cpu().numpy(), df_test["Q_cw_meas_W"].to_numpy()),
        param_errors   = param_recovery_error(recovered),
        recovered_rc   = recovered,
        train_losses   = losses,
        train_time_s   = time.time() - t0,
        epochs_trained = len(losses),
        q_int_pred     = Q_int_te.cpu().numpy(),
        q_cw_pred      = Q_cw_qs_te.cpu().numpy(),
    )


# ============================================================================
# V4F — Neural forcing with FIXED Q_int schedule  [identifiability baseline]
# ============================================================================

def run_V4F(df_train: pd.DataFrame,
            df_test:  pd.DataFrame,
            noise_level: float,
            sensing: str) -> VariantResult:
    """RC parameter estimation with Q_int fixed to ground-truth schedule.

    This variant answers: "Can we recover RC parameters when the forcing
    function is known?"  If yes, the failure of V4/V4D is due to the
    Q_int–RC degeneracy, not a method failure.

    In practice Q_int is never known — but in the synthetic benchmark we
    have the ground truth.  This variant is the identifiability baseline.

    Uses quasi-static training (same as V4) with learnable resistances only.
    Capacitances remain fixed because they are unidentifiable in QS regime
    even with known Q_int.
    """
    T_oa_tr  = df_train["T_oa_C"].to_numpy(dtype=np.float32)
    I_sol_tr = df_train["I_sol_Wm2"].to_numpy(dtype=np.float32)
    T_oa_te  = df_test["T_oa_C"].to_numpy(dtype=np.float32)
    I_sol_te = df_test["I_sol_Wm2"].to_numpy(dtype=np.float32)

    Q_cw_tr  = torch.tensor(df_train["Q_cw_meas_W"].fillna(0).to_numpy(dtype=np.float32),
                             device=DEVICE)
    mask_tr  = torch.tensor(df_train["Q_cw_meas_W"].notna().to_numpy(), device=DEVICE)
    mu_tr    = Q_cw_tr[mask_tr].mean()

    # Ground-truth Q_int — known because this is synthetic data
    Q_int_tr = torch.tensor(df_train["Q_int_W"].to_numpy(dtype=np.float32), device=DEVICE)
    Q_int_te = torch.tensor(df_test["Q_int_W"].to_numpy(dtype=np.float32),  device=DEVICE)

    # Only resistances learnable (same as V4)
    bounds = {
        "R_zw":  (GT.R_zw * 0.1, GT.R_zw * 5.0),
        "R_wo":  (GT.R_wo * 0.1, GT.R_wo * 5.0),
        "R_inf": (GT.R_inf* 0.1, GT.R_inf* 5.0),
    }
    prior_centers = {k: torch.tensor((lo + hi) / 2.0, dtype=torch.float32, device=DEVICE)
                     for k, (lo, hi) in bounds.items()}

    raw_rc = nn.ParameterDict({
        k: nn.Parameter(torch.zeros(1, device=DEVICE)) for k in bounds
    })

    opt   = torch.optim.Adam(raw_rc.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=LR_MIN)
    best_loss, best_state, patience_ct = np.inf, None, 0
    losses = []
    t0 = time.time()

    W_PRIOR = 0.005

    for epoch in range(N_EPOCHS):
        raw_rc.train()
        opt.zero_grad()

        rc_t = {}
        for k, (lo, hi) in bounds.items():
            rc_t[k] = softplus_param(raw_rc[k], lo, hi).squeeze()
        rc_t["A_eff_win"] = torch.tensor(A_EFF_WIN, dtype=torch.float32, device=DEVICE)

        Q_cw_qs  = quasi_static_Q_cw(Q_int_tr, T_oa_tr, I_sol_tr, rc_t)
        L_data   = torch.mean((Q_cw_qs[mask_tr] - Q_cw_tr[mask_tr])**2) / (mu_tr**2 + 1e-8)
        L_prior  = sum(((rc_t[k] - prior_centers[k]) / prior_centers[k])**2
                       for k in bounds) / len(bounds)

        loss = L_data + W_PRIOR * L_prior
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_rc.parameters(), GRAD_CLIP)
        opt.step(); sched.step()

        lv = loss.item(); losses.append(lv)
        if lv < best_loss:
            best_loss  = lv
            best_state = {k: v.clone() for k, v in raw_rc.state_dict().items()}
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE: break

    if best_state:
        raw_rc.load_state_dict(best_state)

    with torch.no_grad():
        rc_final = {}
        for k, (lo, hi) in bounds.items():
            rc_final[k] = softplus_param(raw_rc[k], lo, hi).squeeze()
        rc_final["A_eff_win"] = torch.tensor(A_EFF_WIN, device=DEVICE)

        Q_cw_qs_tr = quasi_static_Q_cw(Q_int_tr, T_oa_tr, I_sol_tr, rc_final)
        Q_cw_qs_te = quasi_static_Q_cw(Q_int_te, T_oa_te, I_sol_te, rc_final)

    recovered = {k: rc_final[k].item() for k in ["R_zw", "R_wo", "R_inf"]}
    recovered["C_z"] = GT.C_z   # fixed at prior
    recovered["C_w"] = GT.C_w   # fixed at prior

    return VariantResult(
        variant="V4F_fixed_Qint",
        sensing=sensing,
        noise_level=noise_level,
        metrics_train  = metrics(Q_cw_qs_tr.cpu().numpy(), df_train["Q_cw_meas_W"].to_numpy()),
        metrics_test   = metrics(Q_cw_qs_te.cpu().numpy(), df_test["Q_cw_meas_W"].to_numpy()),
        param_errors   = param_recovery_error(recovered),
        recovered_rc   = recovered,
        train_losses   = losses,
        train_time_s   = time.time() - t0,
        epochs_trained = len(losses),
        q_int_pred     = None,
        q_cw_pred      = Q_cw_qs_te.cpu().numpy(),
    )


# ============================================================================
# Dispatch
# ============================================================================

VARIANT_FNS = {
    "V1":  run_V1,
    "V2":  run_V2,
    "V3":  run_V3,
    "V4":  run_V4,
    "V4F": run_V4F,
    "V5":  run_V5,
}


def run_variant(name: str,
                df_train: pd.DataFrame,
                df_test:  pd.DataFrame,
                noise_level: float,
                sensing: str,
                seed: int = 0) -> VariantResult:
    """Dispatch to the appropriate variant function.

    `seed` controls PyTorch weight initialization and any numpy randomness
    inside the variant, so repeated calls with different seeds produce
    independent training runs for mean +/- std reporting.
    """
    if name not in VARIANT_FNS:
        raise ValueError(f"Unknown variant '{name}'. Choose from {list(VARIANT_FNS)}")
    # Seed all RNGs that affect weight init / training stochasticity
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"  Running {name} | sensing={sensing} | noise={noise_level:.0%} | seed={seed} ...",
          flush=True, end=" ")
    result = VARIANT_FNS[name](df_train, df_test, noise_level, sensing)
    print(f"CV-RMSE={result.metrics_test['cv_rmse']:.2f}%  "
          f"epochs={result.epochs_trained}  t={result.train_time_s:.1f}s")
    return result
