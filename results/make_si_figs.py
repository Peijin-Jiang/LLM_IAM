"""
make_si_figs.py
───────────────
Supplementary figures from si_figure_data.xlsx (built by build_si_data.py;
one sheet per figure, the data tables one below the other, see make_main_figs.py). Palette, rcParams and
helpers come from make_main_figs.py.
  figS1_hyperparam       hyper-parameter sweeps (KNN, cVAE, finetuned Qwen3.5-9B)
  figS2_region           per-region test accuracy of the finetuned model
  figS3_coverage         scenarios per design dimension and per IAM family
  figS4_dist_totals      2100 distributions; total primary / final energy per case
Usage:  python3 make_si_figs.py [S1 S2 S3 S4]
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from make_main_figs import (TEALS, GOLD_L, GRAY_L, INK, MUTED, ROSE, AR6_C, SYN, CMAP_TEAL, MM, EKW,
                            set_yticks2, plabel, blabel, save, sheet, _NAMES, err, DIM_SHORT, REGION_SHORT, _short_var, box_stats)

SI_XLSX = _HERE / "si_figure_data.xlsx"
CMAP_GOLD = LinearSegmentedColormap.from_list("gold_soft", ["#FAF4E3", "#B9954A"])
CMAP_CORAL = LinearSegmentedColormap.from_list("coral_soft", ["#FBF2EF", "#C97C68"])

# The data tables of each sheet, in their order on the sheet.
SI_BLOCKS = {
    "FigS1": ["figS1_knn_sweep", "figS1_cvae_sweep", "figS1_ft9b_lr_val"],
    "FigS2": ["figS2_region_metrics", "figS2_reference_all_test"],
    "FigS3": ["figS3_scenario_coverage"],
    "FigS4": ["figS4_box_quartiles_2100", "figS4_total_trajectories"],
}
_NAMES[SI_XLSX] = SI_BLOCKS

def si_sheet(name):
    return sheet(name, SI_XLSX)


def panel_label(ax, s, dx=-0.14):
    ax.text(dx, 1.24, s, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")


def _plot_dimension_bars(ax, d):
    """Scenarios per design dimension."""
    dd = d[d.level.eq("dimension")].sort_values("n_scenarios")
    pos = np.arange(len(dd))
    ax.barh(pos, dd.n_scenarios, 0.55, color=TEALS[3], zorder=3)
    for p, v in zip(pos, dd.n_scenarios):
        ax.text(v + 6, p, f"{int(v)}", va="center", fontsize=6.2, color=INK)
    ax.set_yticks(pos)
    ax.set_yticklabels([DIM_SHORT[x] for x in dd.dimension], fontsize=6)
    ax.set_xlim(0, 430)
    ax.set_xticks([0, 200, 400])
    ax.set_xlabel("Scenarios")
    ax.set_title("Per policy dimension", pad=8, fontsize=7.5)
    return dd

def _plot_family_bars(ax, d):
    """Scenarios per IAM family."""
    ff = d[d.level.eq("family")].sort_values("n_scenarios")
    pos = np.arange(len(ff))
    ax.barh(pos, ff.n_scenarios, 0.6, color=TEALS[3], zorder=3)
    for p, v in zip(pos, ff.n_scenarios):
        ax.text(v + 3, p, f"{int(v)}", va="center", fontsize=6.2, color=INK)
    ax.set_yticks(pos)
    ax.set_yticklabels(ff.model_family, fontsize=6)
    ax.set_xlim(0, 205)
    ax.set_xticks([0, 100, 200])
    ax.set_xlabel("Scenarios")
    ax.set_title("Per IAM family", pad=8, fontsize=7.5)
    return ff


# ══════════════════════════════ fig S1: hyper-parameter selection ═══════════
def heat(ax, piv, title, winner, fmt="{:.2f}", cmap=CMAP_TEAL):
    vals = piv.to_numpy(float)
    ax.imshow(vals, cmap=cmap, aspect="auto",
              vmin=np.nanmin(vals), vmax=np.nanmax(vals))
    thr = np.nanmin(vals) + 0.55 * (np.nanmax(vals) - np.nanmin(vals))
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, fmt.format(vals[i, j]), ha="center", va="center",
                    fontsize=5.6, color="white" if vals[i, j] > thr else INK)
    if winner is not None:
        ax.add_patch(Rectangle((winner[1] - 0.5, winner[0] - 0.5), 1, 1,
                               fill=False, edgecolor=ROSE, lw=1.4))
    ax.set_xticks(range(vals.shape[1]))
    ax.set_xticklabels(piv.columns, fontsize=6)
    ax.set_yticks(range(vals.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=6)
    ax.set_title(title, pad=8, fontsize=7.5)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)


def figS1():
    """2 x 2: row 1 = KNN (both weightings), row 2 = cVAE + Qwen3.5-9B."""
    fig, axes = plt.subplots(2, 2, figsize=(150 * MM, 118 * MM))
    fig.subplots_adjust(hspace=0.62, wspace=0.42, top=0.88)
    fig.suptitle("Hyper-parameter selection (validation MAE)",
                 fontsize=8.5, fontweight="bold", y=1.00)

    kn = si_sheet("figS1_knn_sweep").rename(columns={"weights": "wt"})
    sel = kn[kn.selected.astype(bool)].iloc[0]
    for ax, wt, lab in ((axes[0, 0], "uniform", "a"), (axes[0, 1], "similarity", "b")):
        piv = (kn[kn.wt == wt].pivot(index="k", columns="alpha", values="mae")
               .sort_index())
        win = ((list(piv.index).index(sel.k), list(piv.columns).index(sel.alpha))
               if sel.wt == wt else None)
        heat(ax, piv, f"KNN: $k$ × $\\alpha$ ({wt} weights)", win, cmap=CMAP_CORAL)
        ax.set_xlabel("Model-channel weight $\\alpha$"); ax.set_ylabel("$k$")
        panel_label(ax, lab, dx=-0.20)

    ax = axes[1, 0]
    cv = si_sheet("figS1_cvae_sweep")
    piv = cv.pivot(index="learning_rate", columns="hidden_size", values="mae").sort_index()
    def _sci(v):
        e = int(np.floor(np.log10(v)))
        return f"{v / 10 ** e:g}×10$^{{{e}}}$"
    piv.index = [_sci(v) for v in piv.index]
    win = np.unravel_index(np.nanargmin(piv.to_numpy()), piv.shape)
    heat(ax, piv, "cVAE: lr × hidden size", win, cmap=CMAP_GOLD)
    ax.set_xlabel("Hidden size"); ax.set_ylabel("Learning rate")
    panel_label(ax, "c", dx=-0.30)

    ax = axes[1, 1]
    ft = si_sheet("figS1_ft9b_lr_val").sort_values("learning_rate").reset_index(drop=True)
    rows = [ft.iloc[i] for i in range(3)]
    y = [r_.mae for r_ in rows]; x = np.arange(3); e = err(rows)
    ax.bar(x, y, 0.30, color=[TEALS[1], TEALS[4], TEALS[1]],
           yerr=e, error_kw=EKW, zorder=3)
    ax.set_ylim(0, 8.6); set_yticks2(ax, [0, 4, 8], dec=0)
    blabel(ax, x, y, fmt="{:.2f}", err=e[1])
    ax.set_xticks(x)
    ax.set_xticklabels(["2×10$^{-4}$", "5×10$^{-4}$", "1×10$^{-3}$"], fontsize=6.5)
    ax.set_xlabel("LoRA learning rate")
    ax.set_ylabel("MAE (EJ yr$^{-1}$)")
    ax.set_title("Qwen3.5-9B: learning rate", pad=8)
    panel_label(ax, "d", dx=-0.22)

    save(fig, "figS1_hyperparam")


# ═══════════════════════════════════ fig S2: per-region test accuracy ═══════
def figS2():
    """Qwen3.5-9B test accuracy by R10 region (dashed line = test-set mean)."""
    agg = si_sheet("figS2_region_metrics").set_index("region")
    allt = si_sheet("figS2_reference_all_test").iloc[0]

    fig = plt.figure(figsize=(140 * MM, 72 * MM))
    gs = fig.add_gridspec(1, 2, left=0.235, right=0.985,
                          top=0.775, bottom=0.135, wspace=0.10)
    fig.text(0.5, 0.985, "Qwen3.5-9B test accuracy by region",
             ha="center", va="top", fontsize=8.5, fontweight="bold")
    ys = np.arange(len(agg))[::-1]              # best (lowest MAE) on top
    halo = [pe.withStroke(linewidth=1.6, foreground="white")]
    for j, (c, lab, xlim, tks, dec) in enumerate(
            [("mae", "MAE (EJ yr$^{-1}$) $\\downarrow$", (0, 0.80), [0, 0.40, 0.80], 2),
             ("smape_%", "sMAPE (%) $\\downarrow$", (0, 20), [0, 10, 20], 0)]):
        ax = fig.add_subplot(gs[0, j])
        v = agg[c].to_numpy(float)
        for y0, x0 in zip(ys, v):
            ax.plot([xlim[0], x0], [y0, y0], color="#E3E9EC", lw=0.9, zorder=1)
        ax.scatter(v, ys, s=20, color=TEALS[3], zorder=3,
                   edgecolors="white", linewidths=0.6)
        for y0, x0 in zip(ys, v):
            ax.text(x0 + 0.035 * (xlim[1] - xlim[0]), y0, f"{x0:.2f}",
                    ha="left", va="center", fontsize=5.2, color=INK,
                    path_effects=halo)
        ax.axvline(allt[c], color="#6E7A85", lw=0.7, ls=(0, (3, 2)), zorder=2)
        ax.set_xlim(*xlim); ax.set_xticks(tks)
        ax.set_xticklabels([f"{t:.{dec}f}" if dec else f"{t:g}" for t in tks])
        ax.set_ylim(-0.7, len(agg) - 0.3)
        ax.set_yticks(ys)
        ax.set_yticklabels([REGION_SHORT[r] for r in agg.index] if j == 0
                           else [], fontsize=6)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_title(lab, pad=8, fontsize=7.5)
        plabel(ax, "ab"[j], pad=0.135 if j == 0 else 0.055)
    save(fig, "figS2_region")


# ═══════════════════════════════════ fig S3: scenario coverage ══════════════
def figS3():
    """Scenario counts by dimension and IAM family."""
    d = si_sheet("figS3_scenario_coverage")
    fig = plt.figure(figsize=(180 * MM, 68 * MM))
    gs = fig.add_gridspec(1, 2, left=0.115, right=0.985,
                          top=0.770, bottom=0.185, wspace=0.55)
    fig.text(0.5, 0.985, "Scenario coverage of the AR6 R10 corpus",
             ha="center", va="top", fontsize=8.5, fontweight="bold")
    ax = fig.add_subplot(gs[0, 0])
    _plot_dimension_bars(ax, d)
    plabel(ax, "a", pad=0.075)
    ax = fig.add_subplot(gs[0, 1])
    _plot_family_bars(ax, d)
    plabel(ax, "b", pad=0.075)
    save(fig, "figS3_coverage")


# ═══════════════════════ fig S4: 2100 distributions and total energy per case ══
def figS4():
    """a, AR6 vs synthetic
    distributions in 2100 for all 22 variables; b-c, total primary
    and total final energy per case over 2010-2100 (medians and IQR), AR6 vs
    synthetic."""
    bq = si_sheet("figS4_box_quartiles_2100"); tt = si_sheet("figS4_total_trajectories")
    PE = sorted(v for v in bq.variable.unique() if v.startswith("Primary")); FE = sorted(v for v in bq.variable.unique() if v.startswith("Final")); order_v = PE + FE
    xv = np.arange(len(order_v))
    fig = plt.figure(figsize=(185 * MM, 112 * MM))
    gs_a = fig.add_gridspec(1, 1, left=0.085, right=0.985, top=0.945, bottom=0.640)
    ax = fig.add_subplot(gs_a[0, 0])
    for i, v in enumerate(order_v):
        for src, x_off, face, edge in (("AR6", -0.2, GOLD_L, AR6_C), ("synthetic", 0.2, TEALS[2], SYN)):
            r_ = bq[bq.variable.eq(v) & bq.source.eq(src)].iloc[0]
            box_stats(ax, i + x_off, max(r_.whisker_low, 0.01), max(r_.q1, 0.01), max(r_["median"], 0.01), max(r_.q3, 0.01),
                      max(r_.whisker_high, 0.01), face, edge, width=0.34, lw=0.6)
    ax.axvline(len(PE) - 0.5, color=GRAY_L, lw=0.6, ls=(0, (3, 2)), zorder=0)
    for xc, txt in ((len(PE) / 2 - 0.5, "Primary energy"), (len(PE) + len(FE) / 2 - 0.5, "Final energy")):
        ax.text(xc, 1.0, txt, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.2, color=MUTED)
    ax.set_xlim(-0.7, len(order_v) - 0.3); ax.set_yscale("log"); ax.set_ylim(0.01, 3000)
    ax.set_yticks([0.01, 1, 100]); ax.set_yticklabels(["0.01", "1", "100"]); ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylabel("2100, EJ yr$^{-1}$ (log)"); ax.set_xticks(xv)
    ax.set_xticklabels([_short_var(v) for v in order_v], fontsize=5.4, rotation=32, ha="right", rotation_mode="anchor"); ax.tick_params(axis="x", length=0)
    ax.legend([Rectangle((0, 0), 1, 1, facecolor=GOLD_L, edgecolor=AR6_C, lw=0.6), Rectangle((0, 0), 1, 1, facecolor=TEALS[2], edgecolor=SYN, lw=0.6)],
              ["AR6", "Synthetic (Qwen3.5-9B)"], frameon=False, loc="upper left", ncol=2, fontsize=5.8, handlelength=1.2, columnspacing=1.4, borderaxespad=0.2)
    plabel(ax, "a", pad=0.065)
    gs_b = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.400, bottom=0.145, wspace=0.35)
    for j, (name, title, yl, tks) in enumerate([("primary", "Total primary energy per case", (0, 130), [0, 60, 120]),
                                                ("final", "Total final energy per case", (0, 95), [0, 45, 90])]):
        ax = fig.add_subplot(gs_b[0, j])
        dq = tt[tt.total.eq(name) & tt.source.eq("synthetic")].set_index("year").sort_index()
        da = tt[tt.total.eq(name) & tt.source.eq("AR6")].set_index("year").sort_index()
        ax.fill_between(dq.index, dq.q25, dq.q75, color=TEALS[1], alpha=0.6, lw=0, zorder=2)
        ax.plot(dq.index, dq["median"], color=SYN, lw=2.0, zorder=5, solid_capstyle="round")
        for q in ("q25", "q75"): ax.plot(da.index, da[q], color=AR6_C, lw=0.7, ls=(0, (2, 1.4)), zorder=4)
        ax.plot(da.index, da["median"], color=AR6_C, lw=1.3, ls=(0, (2.4, 2.0)), zorder=6, solid_capstyle="butt")
        ax.set_xlim(2008, 2102); ax.set_xticks([2010, 2050, 2100]); ax.set_ylim(*yl); set_yticks2(ax, tks, dec=0)
        ax.set_xlabel("Year"); ax.set_ylabel("EJ yr$^{-1}$")
        ax.text(0.03, 0.95, f"n = {int(dq.n_cases.iloc[0]):,} cases", transform=ax.transAxes, ha="left", va="top", fontsize=5.4, color=MUTED)
        ax.set_title(title, pad=7, fontsize=7.5); plabel(ax, "bc"[j], pad=0.065 if j == 0 else 0.055)
    hc = [plt.Line2D([], [], color=SYN, lw=2.0), Rectangle((0, 0), 1, 1, color=TEALS[1], alpha=0.6),
          plt.Line2D([], [], color=AR6_C, lw=1.3, ls=(0, (2.4, 2.0))), plt.Line2D([], [], color=AR6_C, lw=0.7, ls=(0, (2, 1.4)))]
    fig.legend(hc, ["Synthetic median", "Synthetic IQR", "AR6 median", "AR6 IQR (Q1, Q3)"], frameon=False, loc="upper center",
               bbox_to_anchor=(0.54, 0.055), ncol=4, fontsize=6, handlelength=1.8, columnspacing=1.6)
    save(fig, "figS4_dist_totals")


if __name__ == "__main__":
    which = sys.argv[1:] or ["S1", "S2", "S3", "S4"]
    for w in which:
        {"S1": figS1, "S2": figS2, "S3": figS3, "S4": figS4}[w]()
    print("SI figures written to", _HERE)
