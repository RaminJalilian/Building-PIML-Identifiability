"""
make_figures.py — Publication-oriented figures for the PINN/RC benchmark paper.

Design philosophy:
- Journal-clean, not "presentation flashy"
- No large in-figure "Figure X" titles; captions in manuscript carry numbering
- Consistent variant colors across all figures
- Hatches distinguish noise levels for print/BW readability
- Direct labels and small legends, reduced clutter
- Exports PNG/PDF/SVG; writes a provenance sidecar (figures/v4/PROVENANCE.txt)

Run (CWD-independent; paths anchored to repo root):
    python src/make_figures.py

Figures are generated from results/results_aggregated.csv (5-seed mean) if present,
else results/results_summary.csv. If neither exists the script errors out rather than
falling back to any hardcoded numbers.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from analysis_common import provenance, P          # P = ground-truth BuildingParams

# -------------------------
# Global style
# -------------------------
STYLE = "muted"       # "muted" or "bw"
ROOT = Path(__file__).resolve().parents[1]      # repo root; CWD-independent
OUTDIR = ROOT / "figures/v4"
RESULTS_CSV = ROOT / "results/results_summary.csv"
AGGREGATED_CSV = ROOT / "results/results_aggregated.csv"
DATA_CSV = ROOT / "data/synthetic_dataset.csv"   # forward simulation (weather, states, meters)
RESULTS_JSON = ROOT / "results/results_all.json"  # per-run records incl. test-period q_cw_pred
SENSING_S1_CSV = ROOT / "data/sensing_S1.csv"     # S1 split + timestamps + metered Q_cw

# Representative two-week window (avoids the startup transient and extreme days);
# offset in days from the first timestamp. Shared by both time-series figures.
WINDOW_START_DAY = 28
WINDOW_DAYS = 14

# Prediction / residual figures: first PRED_WINDOW_DAYS days of the HELD-OUT test split
# (not train). Both figures slice the SAME window so they can be read side by side.
PRED_WINDOW_DAYS = 5
PRED_SEED = 0          # V3/V4F are seed-invariant here; V4 seed 0 sits on its 5-seed mean
PRED_NOISE = 0.0       # base noise = the meter signal as-is (bias + base residual, no extra)

# Variant order + line-style identity shared by fig_prediction_timeseries and
# fig_prediction_residuals: style is a SECOND encoding so the traces survive grayscale.
# zorder is set so V4 (tightest trace) is never hidden underneath V4F or V3.
PRED_VARIANTS = ["V4_neural_forcing", "V4F_known_Qint", "V3_hard_arch"]
PRED_LINESTYLE = {"V4_neural_forcing": "-", "V4F_known_Qint": "--", "V3_hard_arch": "-."}
PRED_ZORDER = {"V4_neural_forcing": 6, "V4F_known_Qint": 5, "V3_hard_arch": 4}
PRED_LW = 1.15
MEAS_COLOR = "#B4B4B4"     # neutral light grey for the metered reference

# Identical figure width + identical axes margins in both figures, so the shared x-axis
# lands on exactly the same pixels and the panels overlay cleanly.
PRED_FIGW = 7.4
PRED_LEFT, PRED_RIGHT = 0.088, 0.985

VARIANTS_ALL = [
    "V1_blackbox", "V2_oracle", "V3_hard_arch",
    "V4_neural_forcing", "V4F_known_Qint", "V5_wrong_constraints"
]
VARIANTS_RC = ["V3_hard_arch", "V4_neural_forcing", "V4F_known_Qint", "V5_wrong_constraints"]
NOISES = [0, 5, 10]
SENSING = ["S1", "S2", "S3"]

LABELS = {
    "V1_blackbox": "V1 Black-box",
    "V2_oracle": "V2 Oracle-param ref",
    "V3_hard_arch": "V3 Hard arch.",
    "V4_neural_forcing": "V4 Neural forcing",
    "V4F_known_Qint": r"V4F Known $Q_{int}$",
    "V5_wrong_constraints": "V5 Wrong constraints",
}
SHORT = {
    "V1_blackbox": "V1", "V2_oracle": "V2", "V3_hard_arch": "V3",
    "V4_neural_forcing": "V4", "V4F_known_Qint": "V4F", "V5_wrong_constraints": "V5"
}
# results_all.json stores the original run-time variant names; figures use the display names.
RUN_NAMES = {"V2_oracle": "V2_soft_physics", "V4F_known_Qint": "V4F_fixed_Qint"}

# Single source of truth for variant colors — every figure that colors by variant reads this.
# Hue assignment: V1 purple, V2 brown/ochre, V3 red, V4 blue, V4F green, V5 grey.
# Shades are lightness-staggered (not the raw hues) so the palette survives two failure modes:
#   - grayscale printing: the six print-gray levels are ~21 units apart on 0-255
#     (82, 104, 125, 147, 169, 190), so no two variants collapse. Blue and purple in
#     particular are separated by lightness, not hue alone.
#   - color-vision deficiency: minimum CIELAB dE between any pair is ~15 under simulated
#     protanopia, deuteranopia and tritanopia.
COLORS_MUTED = {
    "V1_blackbox": "#67437F",          # purple  (print gray  82)
    "V2_oracle": "#DF9B3D",            # brown   (print gray 169)
    "V3_hard_arch": "#D6504F",         # red     (print gray 125)
    "V4_neural_forcing": "#296AAD",    # blue    (print gray 104)
    "V4F_known_Qint": "#32A671",       # green   (print gray 147)
    "V5_wrong_constraints": "#BEBEBE", # grey    (print gray 190)
}
COLORS_BW = {
    "V1_blackbox": "#444444",
    "V2_oracle": "#666666",
    "V3_hard_arch": "#222222",
    "V4_neural_forcing": "#000000",
    "V4F_known_Qint": "#555555",
    "V5_wrong_constraints": "#999999",
}
COLORS = COLORS_MUTED if STYLE == "muted" else COLORS_BW

HATCHES = ["", "///", "xx"]
ALPHAS = [0.95, 0.72, 0.52]
NOISE_LABELS = ["base", "+5% noise", "+10% noise"]


def apply_style():
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "ytick.major.size": 3.0,
        "xtick.major.size": 3.0,
        "axes.grid": True,
        "grid.color": "#D7D7D7",
        "grid.linewidth": 0.45,
        "grid.alpha": 0.55,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "hatch.linewidth": 0.45,
    })


def save(fig, name):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUTDIR / f"{name}.{ext}", bbox_inches="tight")
    print(f"saved {OUTDIR / name}.[png,pdf,svg]")
    plt.close(fig)


def load_data():
    # Prefer the aggregated (5-seed mean) file if present, since the manuscript
    # tables and text report seed means, not seed-0 values.
    if AGGREGATED_CSV.exists():
        df = pd.read_csv(AGGREGATED_CSV)
        # Aggregated columns end in _mean/_std/_count; keep the _mean columns
        # and strip the suffix so get_value() reads e.g. "ident_mean_err", "err_R_s".
        rename = {}
        for c in df.columns:
            if c.endswith("_mean"):
                rename[c] = c[: -len("_mean")]
        df = df.rename(columns=rename)
        # Drop the now-redundant _std / _count columns to avoid name clashes
        drop_cols = [c for c in df.columns if c.endswith("_std") or c.endswith("_count")]
        df = df.drop(columns=drop_cols)
        name_map = {
            "V2_soft_physics": "V2_oracle",
            "V4F_fixed_Qint": "V4F_known_Qint",
        }
        df["variant"] = df["variant"].replace(name_map)
        df["noise_pct"] = df["noise_pct"].astype(int)
        print(f"Loaded seed-mean results from {AGGREGATED_CSV}")
        return df
    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
        # Normalize common old variant names
        name_map = {
            "V2_soft_physics": "V2_oracle",
            "V4F_fixed_Qint": "V4F_known_Qint",
        }
        df["variant"] = df["variant"].replace(name_map)
        df["noise_pct"] = df["noise_pct"].astype(int)
        print(f"WARNING: using {RESULTS_CSV} (seed-0 only, not seed-mean).")
        return df
    raise SystemExit(
        "No results CSV found (results/results_aggregated.csv or results_summary.csv). "
        "Run experiment.py first. (The old hardcoded fallback table was removed so figures "
        "can never silently use stale, pre-fix numbers.)")


def select_row(df, variant, sensing, noise):
    sub = df[(df.variant == variant) & (df.sensing == sensing) & (df.noise_pct == noise)]
    if len(sub) == 0:
        return None
    return sub.iloc[0]


def get_value(df, variant, sensing, noise, col):
    row = select_row(df, variant, sensing, noise)
    if row is None or col not in row or pd.isna(row[col]):
        return np.nan
    return float(row[col])


def text_color(variant, min_contrast=4.0):
    """Palette color, darkened just enough to stay legible as small text on white.

    Derived from COLORS (still the single source of truth) — it only scales luminance.
    The light end of the palette (V5 grey) prints fine as a filled patch but is too weak
    for a 8.5 pt label, so labels use this instead of the raw fill color.
    """
    rgb = np.array(matplotlib.colors.to_rgb(COLORS[variant]))
    for _ in range(40):
        lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        L = float(lin @ np.array([0.2126, 0.7152, 0.0722]))
        if (1.0 + 0.05) / (L + 0.05) >= min_contrast:
            break
        rgb *= 0.93
    return matplotlib.colors.to_hex(rgb)


def variant_legend(handles=True, variants=None, ncol=3):
    variants = variants or VARIANTS_ALL
    return [mpatches.Patch(facecolor=COLORS[v], edgecolor="#333333", linewidth=0.5, label=LABELS[v])
            for v in variants]


def noise_legend_patches(base_color="#808080"):
    return [mpatches.Patch(facecolor=base_color, edgecolor="#333333", linewidth=0.5,
                           hatch=HATCHES[i], alpha=ALPHAS[i], label=NOISE_LABELS[i])
            for i in range(3)]


def fig2_prediction_accuracy(df):
    fig, ax = plt.subplots(figsize=(7.2, 3.7), constrained_layout=True)
    x = np.arange(len(VARIANTS_ALL))
    width = 0.22
    cap = 50

    for i, noise in enumerate(NOISES):
        vals = [get_value(df, v, "S1", noise, "test_cv_rmse") for v in VARIANTS_ALL]
        plot_vals = [min(v, cap) if not np.isnan(v) else 0 for v in vals]
        bars = ax.bar(x + (i-1)*width, plot_vals, width,
                      color=[COLORS[v] for v in VARIANTS_ALL],
                      edgecolor="#333333", linewidth=0.45,
                      hatch=HATCHES[i], alpha=ALPHAS[i], zorder=3)

    # Mark truncated V5 bars — zigzag break + staggered labels
    v5_idx = VARIANTS_ALL.index("V5_wrong_constraints")
    label_offsets_v5 = [-0.04, 0.0, 0.04]   # small vertical stagger
    for i, noise in enumerate(NOISES):
        xpos = x[v5_idx] + (i-1)*width
        actual = get_value(df, "V5_wrong_constraints", "S1", noise, "test_cv_rmse")
        # All labels at the same fixed height, just above the bars
        ax.text(xpos, cap + 1.2, f"{actual:.0f}%", ha="center", va="bottom",
                fontsize=6.8, color="#333333")
        # Zigzag break mark
        zx = np.linspace(xpos-width*0.42, xpos+width*0.42, 10)
        zy = (cap - 2.0) + np.array([0,1,-1,1,-1,1,-1,1,-1,0]) * 0.85
        ax.plot(zx, zy, color="white", lw=2.5, zorder=5, solid_capstyle="butt")
        ax.plot(zx, zy, color="#333333", lw=0.6, zorder=6, solid_capstyle="butt")

    ax.set_ylabel("Test CV-RMSE [%]")
    ax.set_ylim(0, 57)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_yticklabels(["0", "10", "20", "30", "40", "≥50"])
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[v] for v in VARIANTS_ALL], fontsize=9)
    ax.set_xlabel("Model variant")
    ax.text(0.995, 0.97, "V5 bars capped at 50%; actual values shown above",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#666666")

    # Compact legends
    leg1 = ax.legend(handles=noise_legend_patches(), loc="upper left",
                     frameon=True, edgecolor="#BBBBBB", framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(handles=variant_legend(variants=VARIANTS_ALL), loc="upper center",
              bbox_to_anchor=(0.5, -0.20), ncol=3,
              frameon=True, edgecolor="#BBBBBB", framealpha=0.95)
    save(fig, "fig2_prediction_accuracy_v3")


def fig3_parameter_recovery(df):
    # Recovery is reported ONLY for identifiable combinations: R_s = R_zw+R_wo and R_inf.
    # The R_zw/R_wo split is non-identifiable under the operator and is never plotted here.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.8), constrained_layout=True)

    # (a) identifiable recovery error (mean of R_s, R_inf) across noise levels
    x = np.arange(len(VARIANTS_RC))
    width = 0.22
    for i, noise in enumerate(NOISES):
        vals = [get_value(df, v, "S1", noise, "ident_mean_err") for v in VARIANTS_RC]
        ax1.bar(x + (i-1)*width, vals, width,
                color=[COLORS[v] for v in VARIANTS_RC],
                edgecolor="#333333", linewidth=0.45,
                hatch=HATCHES[i], alpha=ALPHAS[i], zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([SHORT[v] for v in VARIANTS_RC])
    ax1.set_xlabel("Model variant")
    ax1.set_ylabel("Identifiable recovery error [%]")
    ax1.set_ylim(0, 195)
    ax1.set_title(r"(a) Identifiable recovery error (mean of $R_s$, $R_{inf}$)",
                  loc="left", fontsize=9.5)
    ax1.legend(handles=noise_legend_patches(),
               loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
               frameon=True, edgecolor="#BBBBBB", framealpha=0.95)

    # (b) per identifiable combination at base noise
    params = ["err_R_s", "err_R_inf"]
    labels = [r"$R_s\,{=}\,R_{zw}{+}R_{wo}$", r"$R_{inf}$"]
    xp = np.arange(len(params))
    w = 0.18
    offsets = np.linspace(-1.5*w, 1.5*w, len(VARIANTS_RC))
    for j, v in enumerate(VARIANTS_RC):
        vals = [get_value(df, v, "S1", 0, p) for p in params]
        ax2.bar(xp + offsets[j], vals, w,
                color=COLORS[v], alpha=0.9, edgecolor="#333333", linewidth=0.45, zorder=3)
    ax2.set_xticks(xp)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_ylabel("Recovery error [%]")
    ax2.set_ylim(0, 240)
    ax2.set_title("(b) Identifiable combinations (base case)", loc="left", fontsize=9.5)
    ax2.legend(handles=variant_legend(variants=VARIANTS_RC),
               loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
               frameon=True, edgecolor="#BBBBBB", framealpha=0.95, fontsize=7.5)
    save(fig, "fig3_parameter_recovery_v3")


def fig4_tradeoff(df):
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    points = []
    for v in VARIANTS_RC:
        cv = get_value(df, v, "S1", 0, "test_cv_rmse")
        rec = get_value(df, v, "S1", 0, "ident_mean_err")   # identifiable (R_s, R_inf) mean
        if np.isnan(cv) or np.isnan(rec):
            continue
        cv_plot = min(cv, 62)
        points.append((v, cv, cv_plot, rec))

    for v, cv, cv_plot, rec in points:
        ax.scatter(cv_plot, rec, s=210, color=COLORS[v],
                   edgecolor="#222222", linewidth=0.8, zorder=5)
        label_offsets = {
            "V3_hard_arch": (-8, 10),
            "V4_neural_forcing": (-18, -16),
            "V4F_known_Qint": (-12, 10),
            "V5_wrong_constraints": (-18, 10),
        }
        dx, dy = label_offsets.get(v, (8, 8))
        ax.annotate(SHORT[v], (cv_plot, rec), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8.5, fontweight="bold",
                    color=text_color(v))

    # Optional subtle quadrant note, no arbitrary threshold lines
    ax.text(0.02, 0.03, "Lower-left is ideal:\nlow prediction error + low recovery error",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color="#777777")

    v5_cv = get_value(df, "V5_wrong_constraints", "S1", 0, "test_cv_rmse")
    ax.annotate(f"actual CV-RMSE = {v5_cv:.0f}%\n(plotted at capped x-position)",
                xy=(62, get_value(df, "V5_wrong_constraints", "S1", 0, "ident_mean_err")),
                xytext=(40, 140), fontsize=7.3, color="#777777",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#888888"))

    ax.set_xlabel("Test CV-RMSE [%]  (lower = better prediction)")
    ax.set_ylabel(r"Identifiable recovery error [%] (mean of $R_s$, $R_{inf}$)")
    ax.set_xlim(-1, 66)
    ax.set_ylim(-5, 195)
    ax.legend(handles=variant_legend(variants=VARIANTS_RC), loc="upper center",
              bbox_to_anchor=(0.52, 1.02), ncol=2,
              frameon=True, edgecolor="#BBBBBB", framealpha=0.95)
    save(fig, "fig4_prediction_identification_tradeoff_v3")


def fig5_sensing(df):
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.8), constrained_layout=False)
    fig.subplots_adjust(top=0.80, wspace=0.35)
    sensing_labels = [r"$Q_{cw}$ only", r"$Q_{cw}$ + sparse $T_z$", r"$Q_{cw}$ + full $T_z$"]
    variants = ["V4_neural_forcing", "V4F_known_Qint"]

    for ax, v in zip(axes, variants):
        x = np.arange(len(SENSING))
        width = 0.22
        for i, noise in enumerate(NOISES):
            vals = [get_value(df, v, s, noise, "test_cv_rmse") for s in SENSING]
            ax.bar(x + (i-1)*width, vals, width,
                   color=COLORS[v], edgecolor="#333333", linewidth=0.45,
                   hatch=HATCHES[i], alpha=ALPHAS[i], zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(sensing_labels, rotation=12, ha="right")
        ax.set_ylabel("Test CV-RMSE [%]")
        ax.set_title(LABELS[v], loc="left", fontsize=10)
        ax.legend(handles=noise_legend_patches(COLORS[v]),
                  loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3,
                  frameon=True, edgecolor="#BBBBBB", framealpha=0.95)
        # Keep y-limits comparable but not too empty
        ax.set_ylim(0, 26 if v == "V4F_known_Qint" else 17)
    save(fig, "fig5_sensing_configuration_v3")


# ============================================================================
# Time-series manuscript figures (from the forward simulation, DATA_CSV)
# ============================================================================

def _window(dfd):
    """Slice the shared representative two-week window from the forward simulation."""
    ts = pd.DatetimeIndex(dfd["timestamp"])
    t0 = ts[0] + pd.Timedelta(days=WINDOW_START_DAY)
    t1 = t0 + pd.Timedelta(days=WINDOW_DAYS)
    return dfd[(ts >= t0) & (ts < t1)].copy()


def _shade_daytime(axes, wk):
    """Light daytime band (sunrise 06:00 - sunset 18:30) to show the T_oa / solar phase."""
    ts = pd.DatetimeIndex(wk["timestamp"])
    for a in axes:
        for day in pd.date_range(ts[0].normalize(), ts[-1], freq="D"):
            a.axvspan(day + pd.Timedelta(hours=6), day + pd.Timedelta(hours=18.5),
                      color="#F1C40F", alpha=0.05, lw=0)


def fig_data_verification(dfd):
    """Figure 1 — the benchmark data is physical.

    Three panels on a shared time axis over a representative two-week window:
      (a) outdoor dry-bulb temperature T_oa, with the cooling setpoint reference line;
      (b) global horizontal solar irradiance I_sol (peaks at solar noon);
      (c) chilled-water cooling load Q_cw (tracks temperature and solar).
    (The wall state T_w and its lag are not shown here; the wall-model discrepancy is discussed
    with the ceiling analysis in the study text.)
    """
    wk = _window(dfd)
    ts = pd.DatetimeIndex(wk["timestamp"])
    fig, ax = plt.subplots(3, 1, figsize=(7.4, 5.8), sharex=True, constrained_layout=True)

    ax[0].plot(ts, wk["T_oa_C"], color="#C0392B", lw=1.1)
    ax[0].axhline(P.T_z_sp, color="#2E9E6B", ls="--", lw=0.8, label=f"setpoint {P.T_z_sp:.0f} °C")
    ax[0].set_ylabel("T$_{oa}$ [°C]"); ax[0].legend(fontsize=7.5, loc="upper right")

    ax[1].plot(ts, wk["I_sol_Wm2"], color="#E67E22", lw=1.1)
    ax[1].set_ylabel("I$_{sol}$ [W/m²]")

    ax[2].plot(ts, wk["Q_cw_W"] / 1e3, color="#2B6CB0", lw=1.1)
    ax[2].set_ylabel("Q$_{cw}$ [kW]"); ax[2].set_xlabel("Time")

    _shade_daytime(ax, wk)
    fig.suptitle("Synthetic benchmark: outdoor temperature, solar irradiance, and cooling load "
                 "over a representative two-week window", fontsize=9.5, y=1.02)
    fig.autofmt_xdate(rotation=20)
    save(fig, "fig_data_verification")


def fig_measurement_layer(dfd):
    """Figure 2 — the hidden forcing and the measurement layer the inverse problem must cope with.

    Three panels on the SAME two-week window as Figure 1 (kept distinct from Figure 1, which
    shows the physical drivers/states):
      (a) internal-gain forcing Q_int(t) — the HIDDEN input the inverse problem must untangle
          from the RC parameters (never metered);
      (b) chilled-water load: ground truth vs. the biased + noisy meter the model actually sees;
      (c) zone air temperature: ground truth vs. the noisy meter, with the setpoint. T_z is
          HVAC-regulated to ~setpoint, so its TRUE signal std (~0.19 degC) is comparable to the
          sensor noise (0.30 degC): measurement SNR is near unity BY CONSTRUCTION. This low SNR
          is itself why observed zone temperature adds little identifying information (it is not
          a sensor failure) — consistent with the sensing result.
    """
    wk = _window(dfd)
    ts = pd.DatetimeIndex(wk["timestamp"])
    fig, ax = plt.subplots(3, 1, figsize=(7.4, 6.0), sharex=True, constrained_layout=True)

    ax[0].plot(ts, wk["Q_int_W"] / 1e3, color="#6A51A3", lw=1.1)
    ax[0].set_ylabel("Q$_{int}$ [kW]")
    ax[0].text(0.012, 0.92, "hidden internal-gain forcing (unmetered)",
               transform=ax[0].transAxes, fontsize=7.3, va="top", color="#555555")

    ax[1].plot(ts, wk["Q_cw_W"] / 1e3, color="#333333", lw=1.0, label="truth")
    ax[1].plot(ts, wk["Q_cw_meas_W"] / 1e3, color="#E45756", lw=0, marker=".", ms=2.2,
               alpha=0.6, label="metered (biased + noisy)")
    ax[1].set_ylabel("Q$_{cw}$ [kW]"); ax[1].legend(fontsize=7.5, loc="upper right", ncol=2)

    ax[2].plot(ts, wk["T_z"], color="#333333", lw=1.0, label="truth")
    ax[2].plot(ts, wk["T_z_meas_C"], color="#2B6CB0", lw=0, marker=".", ms=2.2,
               alpha=0.5, label="metered (noisy)")
    ax[2].axhline(P.T_z_sp, color="#2E9E6B", ls="--", lw=0.8, label=f"setpoint {P.T_z_sp:.0f} °C")
    ax[2].set_ylabel("T$_z$ [°C]"); ax[2].legend(fontsize=7.0, loc="upper right", ncol=3)
    ax[2].set_xlabel("Time")
    sig = float(dfd["T_z"].std()); noise = 0.30      # sensor sigma from simulate.add_measurement_layer
    ax[2].text(0.012, 0.05,
               f"zone HVAC-regulated to ~setpoint: true σ≈{sig:.2f}°C ≈ sensor noise {noise:.2f}°C "
               f"(SNR≈{sig/noise:.1f}, near unity by construction)\n→ observed T$_z$ carries little "
               f"identifying information — this low SNR (not sensor failure) is the sensing result",
               transform=ax[2].transAxes, fontsize=6.4, va="bottom", color="#333333",
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BBBBBB", alpha=0.85))

    fig.suptitle("Benchmark forcing and measurement layer over the same two-week window",
                 fontsize=9.5, y=1.02)
    fig.autofmt_xdate(rotation=20)
    save(fig, "fig_measurement_layer")


def _load_predictions():
    """Stored test-period predictions + the metered reference, aligned to timestamps.

    Reads results/results_all.json (no retraining, no refitting) and data/sensing_S1.csv.
    Returns a dict shared by fig_prediction_timeseries and fig_prediction_residuals so the two
    figures cannot drift apart in window, units, or CV-RMSE convention.

    The reference series is the METERED Q_cw — the biased, noisy meter signal the models were
    both trained and scored against — so the residuals drawn here are exactly the residuals that
    enter the reported CV-RMSE. Ground-truth Q_cw is deliberately not used: it is not the
    evaluation target. V5 is omitted: at ~298% CV-RMSE its trace would compress every other
    series into a flat band.
    """
    runs = json.loads(RESULTS_JSON.read_text())
    s1 = pd.read_csv(SENSING_S1_CSV, parse_dates=["timestamp"])
    test = s1[s1["split"] == "test"].reset_index(drop=True)

    ts = pd.DatetimeIndex(test["timestamp"])
    t0 = ts[0]
    t1 = t0 + pd.Timedelta(days=PRED_WINDOW_DAYS)
    win = np.asarray((ts >= t0) & (ts < t1))
    meas_kW = test["Q_cw_meas_W"].to_numpy() / 1e3   # NaN = meter dropout; kept as NaN so
                                                     # every trace breaks at the same hours

    pred, resid, cv_full, cv_win = {}, {}, {}, {}
    for v in PRED_VARIANTS:
        run_name = RUN_NAMES.get(v, v)
        rec = next(r for r in runs
                   if r["variant"] == run_name and r["sensing"] == "S1"
                   and r["noise_level"] == PRED_NOISE and r["seed"] == PRED_SEED)
        p = np.asarray(rec["q_cw_pred"], dtype=float).ravel() / 1e3
        if len(p) != len(test):
            raise SystemExit(f"{run_name}: stored q_cw_pred has {len(p)} points but the S1 "
                             f"test split has {len(test)} — cannot align to timestamps.")
        pred[v] = p
        resid[v] = p - meas_kW               # NaN wherever the meter dropped out
        # CV-RMSE recomputed from the saved arrays (same convention as variants.metrics):
        # RMSE over finite pairs, normalised by the mean of the measured series.
        cv_full[v] = _cv_rmse(p, meas_kW)
        cv_win[v] = _cv_rmse(p[win], meas_kW[win])

    return dict(test=test, ts=ts, win=win, meas_kW=meas_kW, pred=pred, resid=resid,
                cv_full=cv_full, cv_win=cv_win, stored={v: _stored_cv(runs, v) for v in PRED_VARIANTS})


def _cv_rmse(pred, meas):
    """CV-RMSE [%] on the finite pairs — same definition as variants.metrics()."""
    ok = np.isfinite(pred) & np.isfinite(meas)
    resid = pred[ok] - meas[ok]
    return float(np.sqrt((resid ** 2).mean()) / meas[ok].mean() * 100.0)


def _stored_cv(runs, variant):
    """The CV-RMSE recorded at run time, for cross-checking the recomputation."""
    run_name = RUN_NAMES.get(variant, variant)
    rec = next(r for r in runs
               if r["variant"] == run_name and r["sensing"] == "S1"
               and r["noise_level"] == PRED_NOISE and r["seed"] == PRED_SEED)
    return float(rec["metrics_test"]["cv_rmse"])


def _pred_time_axis(ax, ts_win):
    """Identical x-axis (limits, tick positions, labels) in both prediction figures."""
    import matplotlib.dates as mdates
    ax.set_xlim(ts_win[0], ts_win[-1])
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))


def fig_prediction_timeseries(D):
    """Predicted vs. metered chilled-water load over the first 5 days of the held-out test split.

    Shows what the CV-RMSE table means trace-by-trace, under S1 (energy-only metering) at base
    noise. Companion to fig_prediction_residuals, which plots the same window as residuals.
    """
    ts, win, meas_kW = D["ts"], D["win"], D["meas_kW"]

    fig, ax = plt.subplots(figsize=(PRED_FIGW, 3.5))
    fig.subplots_adjust(left=PRED_LEFT, right=PRED_RIGHT, top=0.80, bottom=0.155)

    # Metered reference in the background: thin light-grey line with faint hourly markers.
    # NaN hours stay NaN, so dropout shows as a gap rather than an interpolated segment.
    ax.plot(ts[win], meas_kW[win], color=MEAS_COLOR, lw=0.8, marker=".", ms=2.0,
            markerfacecolor=MEAS_COLOR, markeredgecolor="none", alpha=0.9, zorder=2,
            label=r"measured Q$_{cw}$ (S1 meter)")

    for v in PRED_VARIANTS:
        ax.plot(ts[win], D["pred"][v][win], color=COLORS[v], lw=PRED_LW,
                ls=PRED_LINESTYLE[v], zorder=PRED_ZORDER[v], solid_joinstyle="round",
                dash_capstyle="round", label=f"{LABELS[v]}  ({D['cv_full'][v]:.1f}%)")

    ax.set_ylabel("Q$_{cw}$ [kW]")
    ax.set_xlabel("Time")
    # Tight y-limits: span everything actually drawn, plus a 4% margin — peaks are never clipped.
    drawn = np.concatenate([s[win][np.isfinite(s[win])]
                            for s in [meas_kW] + [D["pred"][v] for v in PRED_VARIANTS]])
    pad = 0.04 * (drawn.max() - drawn.min())
    ax.set_ylim(drawn.min() - pad, drawn.max() + pad)
    _pred_time_axis(ax, ts[win])

    # Legend above the axes so it never sits on the daily peaks. The parenthesised numbers are
    # FULL-TEST CV-RMSE (all 429 finite hours), not the 5-day window — stated in the legend title.
    ax.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=4,
              frameon=True, edgecolor="#BBBBBB", framealpha=0.95, columnspacing=1.2,
              handlelength=2.4,
              title="CV-RMSE in parentheses = full held-out test period (not this window)",
              title_fontsize=7.2)

    fig.suptitle("Predicted vs. metered chilled-water load, first 5 days of the held-out test "
                 "period (S1, base noise)", fontsize=9.5, y=0.995)
    save(fig, "fig_prediction_timeseries")


def fig_prediction_residuals(D):
    """Residuals (predicted − metered Q_cw) of the same three variants.

    (a) residual traces over exactly the window drawn in fig_prediction_timeseries;
    (b) boxplots of every finite residual in the COMPLETE held-out test period.

    The reference is the metered signal, i.e. the reference behind the reported metrics — not
    noise-free truth. No +-sigma band is drawn: the measurement layer combines bias,
    heteroscedastic noise and dropout, so a single constant sigma would misrepresent it.
    """
    ts, win = D["ts"], D["win"]

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(PRED_FIGW, 5.3),
                                     gridspec_kw=dict(height_ratios=[2.05, 1.0]))
    fig.subplots_adjust(left=PRED_LEFT, right=PRED_RIGHT, top=0.945, bottom=0.115, hspace=0.42)

    # ---- (a) residual time series -----------------------------------------------------
    ax_a.axhline(0.0, color="#8C8C8C", lw=0.8, zorder=2)
    for v in PRED_VARIANTS:
        ax_a.plot(ts[win], D["resid"][v][win], color=COLORS[v], lw=PRED_LW,
                  ls=PRED_LINESTYLE[v], zorder=PRED_ZORDER[v], solid_joinstyle="round",
                  dash_capstyle="round", label=LABELS[v])
    # Symmetric about zero, 8% beyond the largest finite residual drawn: nothing is clipped.
    rmax = max(np.nanmax(np.abs(D["resid"][v][win])) for v in PRED_VARIANTS)
    ax_a.set_ylim(-1.08 * rmax, 1.08 * rmax)
    _pred_time_axis(ax_a, ts[win])
    ax_a.set_ylabel("Residual, predicted − measured [kW]")
    ax_a.set_xlabel("Time")
    ax_a.set_title("(a) Prediction residuals over the selected test interval",
                   loc="left", fontsize=9.5)
    # Legend inside the panel: the symmetric limits leave the top band empty (residuals never
    # exceed ~+16 kW), so it costs no data space and keeps the panel title clear.
    ax_a.legend(fontsize=7.5, loc="upper right", ncol=3, frameon=True,
                edgecolor="#BBBBBB", framealpha=0.95, handlelength=2.4, columnspacing=1.2)

    # ---- (b) full-test residual distributions -----------------------------------------
    # Horizontal boxplots keep panel (a)'s axes width (and therefore its x-axis geometry)
    # identical to fig_prediction_timeseries. Boxes, not violins: no bandwidth choice to read.
    ax_b.axvline(0.0, color="#8C8C8C", lw=0.8, zorder=2)
    data = [D["resid"][v][np.isfinite(D["resid"][v])] for v in PRED_VARIANTS]
    pos = [3, 2, 1]      # V4 top, then V4F, then V3 — same order as the legend above
    bp = ax_b.boxplot(data, positions=pos, vert=False, widths=0.55, whis=1.5,
                      patch_artist=True, showfliers=True, zorder=3)
    for i, v in enumerate(PRED_VARIANTS):
        edge = text_color(v)
        bp["boxes"][i].set(facecolor=COLORS[v], alpha=0.35, edgecolor=edge, linewidth=0.9,
                           linestyle=PRED_LINESTYLE[v])
        bp["medians"][i].set(color=edge, linewidth=1.5)
        for w in bp["whiskers"][2*i:2*i+2]:
            w.set(color=edge, linewidth=0.9)
        for c in bp["caps"][2*i:2*i+2]:
            c.set(color=edge, linewidth=0.9)
        bp["fliers"][i].set(marker=".", markersize=2.4, markerfacecolor=COLORS[v],
                            markeredgecolor="none", alpha=0.35)
    ax_b.set_yticks(pos)
    ax_b.set_yticklabels([SHORT[v] for v in PRED_VARIANTS])
    ax_b.set_ylim(0.4, 3.6)
    ax_b.set_xlabel("Residual, predicted − measured [kW]")
    ax_b.set_title("(b) Full-test residual distributions", loc="left", fontsize=9.5)
    n_full = int(np.isfinite(D["resid"][PRED_VARIANTS[0]]).sum())
    fig.text(0.5, 0.012,
             f"all {n_full} finite hourly residuals of the held-out test period; "
             f"box = median and IQR, whiskers = 1.5×IQR, faint points beyond",
             ha="center", va="bottom", fontsize=6.8, color="#666666")

    save(fig, "fig_prediction_residuals")


def report_prediction_figures(D):
    """Numbers behind both figures, printed so the figure claims are checkable from the log."""
    ts, win, meas_kW = D["ts"], D["win"], D["meas_kW"]
    print("\n--- prediction/residual figure validation "
          "-------------------------------------------")
    print(f"  test period  : {ts[0]} -> {ts[-1]}  ({len(ts)} h)")
    print(f"  window drawn : {ts[win][0]} -> {ts[win][-1]}  "
          f"({win.sum()} h, {PRED_WINDOW_DAYS} d, seed {PRED_SEED})")
    miss_all = ts[~np.isfinite(meas_kW)]
    miss_win = ts[win][~np.isfinite(meas_kW[win])]
    print(f"  measured dropout: {len(miss_win)} in window, {len(miss_all)} in full test "
          f"-> {', '.join(str(t) for t in miss_all)}")
    print(f"  {'':>4s}{'CV-RMSE full':>14s}{'stored':>10s}{'CV-RMSE win':>14s}")
    for v in PRED_VARIANTS:
        print(f"  {SHORT[v]:>4s}{D['cv_full'][v]:13.4f}%{D['stored'][v]:10.4f}"
              f"{D['cv_win'][v]:13.4f}%")
    hdr = ("n", "mean", "sd", "median", "MAE", "p5", "p25", "p75", "p95", "min", "max")
    print(f"  full-test residuals [kW]  {'':>2s}" + "".join(f"{h:>9s}" for h in hdr))
    for v in PRED_VARIANTS:
        r = D["resid"][v][np.isfinite(D["resid"][v])]
        q = np.percentile(r, [5, 25, 50, 75, 95])
        vals = (len(r), r.mean(), r.std(ddof=1), q[2], np.abs(r).mean(),
                q[0], q[1], q[3], q[4], r.min(), r.max())
        print(f"  {SHORT[v]:>4s}{'':>22s}" + f"{vals[0]:9d}" +
              "".join(f"{x:9.2f}" for x in vals[1:]))
    print("-" * 78)


if __name__ == "__main__":
    prov = provenance(AGGREGATED_CSV, DATA_CSV, RESULTS_JSON, SENSING_S1_CSV)
    print(prov)
    apply_style()
    df = load_data()
    fig2_prediction_accuracy(df)
    fig3_parameter_recovery(df)
    fig4_tradeoff(df)
    fig5_sensing(df)
    # time-series figures from the forward simulation
    df_data = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    fig_data_verification(df_data)
    fig_measurement_layer(df_data)
    # predicted vs. metered traces + residuals, from the stored test-period predictions.
    # Both figures read the same loader, so they cannot drift apart in window or convention.
    pred_data = _load_predictions()
    fig_prediction_timeseries(pred_data)
    fig_prediction_residuals(pred_data)
    report_prediction_figures(pred_data)
    # provenance sidecar: maps every figure in this directory to the code/input state (D14)
    import datetime
    figs = sorted({p.stem for p in OUTDIR.glob("*.png")})
    (OUTDIR / "PROVENANCE.txt").write_text(
        f"{prov}\n"
        f"generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
        f"generator: src/make_figures.py\n"
        f"figures ({len(figs)}): {', '.join(figs)}\n")
    print(f"Wrote provenance sidecar -> {OUTDIR / 'PROVENANCE.txt'}")
    print(f"Done. Figures saved to {OUTDIR.resolve()}")
