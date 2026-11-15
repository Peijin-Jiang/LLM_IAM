"""
make_si_figs.py
───────────────
Supplementary figures from si_figure_data.xlsx (built by build_si_data.py;
one sheet per figure, the data tables one below the other, see make_main_figs.py). Palette, rcParams and
helpers come from make_main_figs.py.
  figS1_hyperparam       hyper-parameter sweeps (KNN, cVAE, finetuned Qwen3.5-9B)
  figS2_region           per-region test accuracy of the finetuned model
  figS3_coverage         scenarios per design dimension and per IAM family
  figS4_trajectories     medians and percentile ranges of all 22 variables over time
  figS5_distributions    distributions of all 22 variables in 2050 and 2100
  figS6_year_values      error of the four methods by year; all decadal values
Usage:  python3 make_si_figs.py [S1 S2 S3 S4 S5 S6]
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
from make_main_figs import (TEALS, GOLD_L, GRAY_L, INK, MUTED, ROSE, AR6_C, SYN, CMAP_TEAL, CMAP_DENS, MM, EKW, M_COLORS, M_LABELS,
                            set_yticks2, plabel, blabel, save, sheet, _NAMES, err, DIM_SHORT, REGION_SHORT, _short_var, box_stats)

SI_XLSX = _HERE / "si_figure_data.xlsx"
CMAP_GOLD = LinearSegmentedColormap.from_list("gold_soft", ["#FAF4E3", "#B9954A"])
CMAP_CORAL = LinearSegmentedColormap.from_list("coral_soft", ["#FBF2EF", "#C97C68"])

# The data tables of each sheet, in their order on the sheet.
SI_BLOCKS = {
    "FigS1": ["figS1_knn_sweep", "figS1_cvae_sweep", "figS1_ft9b_lr_val"],
    "FigS2": ["figS2_region_metrics", "figS2_reference_all_test"],
    "FigS3": ["figS3_scenario_coverage"],
    "FigS4": ["figS4_envelopes"],
    "FigS5": ["figS5_box_quartiles"],
    "FigS6": ["figS6_error_by_year", "figS6_value_density", "figS6_value_density_grid"],
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
    """Medians and 10th-90th percentile ranges across test cases over 2010-2100
    for each of the 22 variables, AR6 vs synthetic."""
    env = si_sheet("figS4_envelopes")
    PE = sorted(v for v in env.variable.unique() if v.startswith("Primary")); FE = sorted(v for v in env.variable.unique() if v.startswith("Final"))
    fig = plt.figure(figsize=(185 * MM, 160 * MM))
    for g, (group, vars_, top, bottom) in enumerate((("Primary energy", PE, 0.930, 0.580), ("Final energy", FE, 0.472, 0.122))):
        gs = fig.add_gridspec(2, 6, left=0.085, right=0.985, top=top, bottom=bottom, wspace=0.52, hspace=0.66)
        n_cases = int(env[env.variable.eq(vars_[0])].n_cases.iloc[0]); assert env[env.variable.isin(vars_)].n_cases.eq(n_cases).all()
        fig.text(0.085, top + 0.027, f"{group} (n = {n_cases:,} cases)", fontsize=6.6, color=MUTED, ha="left", va="bottom")
        for i, v in enumerate(vars_):
            ax = fig.add_subplot(gs[i // 6, i % 6]); e_ = env[env.variable.eq(v)].sort_values("year"); yrs = e_.year.to_numpy(int)
            ax.fill_between(yrs, e_.syn_p10, e_.syn_p90, color=TEALS[1], alpha=0.55, lw=0, zorder=2)
            ax.plot(yrs, e_.syn_median, color=SYN, lw=1.4, zorder=5)
            for q in ("ar6_p10", "ar6_p90"): ax.plot(yrs, e_[q], color=AR6_C, lw=0.6, ls=(0, (2, 1.4)), zorder=4)
            ax.plot(yrs, e_.ar6_median, color=AR6_C, lw=1.0, ls=(0, (2.6, 1.8)), zorder=6)
            ax.set_ylim(0, max(float(max(e_.ar6_p90.max(), e_.syn_p90.max())) * 1.12, 0.05)); ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
            ax.set_xlim(2008, 2102); ax.set_xticks([2020, 2060, 2100]); ax.tick_params(labelsize=5.2, pad=1.5)
            ax.set_title(_short_var(v), fontsize=6.0, pad=3, loc="left")
            if i % 6 == 0: ax.set_ylabel("EJ yr$^{-1}$", fontsize=6, labelpad=2)
            if g == 1 and i // 6 == 1: ax.set_xlabel("Year", fontsize=6, labelpad=1.5)
    hB = [Rectangle((0, 0), 1, 1, color=TEALS[1], alpha=0.55), plt.Line2D([], [], color=SYN, lw=1.4),
          plt.Line2D([], [], color=AR6_C, lw=0.6, ls=(0, (2, 1.4))), plt.Line2D([], [], color=AR6_C, lw=1.0, ls=(0, (2.6, 1.8)))]
    fig.legend(hB, ["Synthetic, 10th–90th percentile", "Synthetic median", "AR6, 10th and 90th percentile", "AR6 median"], frameon=False,
               loc="upper center", bbox_to_anchor=(0.54, 0.050), ncol=4, fontsize=5.8, handlelength=1.6, columnspacing=1.4)
    save(fig, "figS4_trajectories")


def figS5():
    """AR6 vs synthetic distributions of all 22 variables in 2050 (a) and 2100 (b)."""
    bq = si_sheet("figS5_box_quartiles")
    PE = sorted(v for v in bq.variable.unique() if v.startswith("Primary")); FE = sorted(v for v in bq.variable.unique() if v.startswith("Final")); order_v = PE + FE
    xv = np.arange(len(order_v))
    fig = plt.figure(figsize=(185 * MM, 100 * MM))
    for j, (year, top, bottom) in enumerate(((2050, 0.930, 0.640), (2100, 0.530, 0.240))):
        ax = fig.add_subplot(fig.add_gridspec(1, 1, left=0.085, right=0.985, top=top, bottom=bottom)[0, 0])
        for i, v in enumerate(order_v):
            for src, x_off, face, edge in (("AR6", -0.2, GOLD_L, AR6_C), ("synthetic", 0.2, TEALS[2], SYN)):
                r_ = bq[bq.variable.eq(v) & bq.source.eq(src) & bq.year.eq(year)].iloc[0]
                box_stats(ax, i + x_off, max(r_.whisker_low, 0.01), max(r_.q1, 0.01), max(r_["median"], 0.01), max(r_.q3, 0.01),
                          max(r_.whisker_high, 0.01), face, edge, width=0.34, lw=0.6)
        ax.axvline(len(PE) - 0.5, color=GRAY_L, lw=0.6, ls=(0, (3, 2)), zorder=0)
        ax.set_xlim(-0.7, len(order_v) - 0.3); ax.set_yscale("log"); ax.set_ylim(0.01, 3000)
        ax.set_yticks([0.01, 1, 100]); ax.set_yticklabels(["0.01", "1", "100"]); ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.set_ylabel(f"{year}, EJ yr$^{{-1}}$ (log)"); ax.set_xticks(xv); ax.tick_params(axis="x", length=0)
        if j == 0:
            for xc, txt in ((len(PE) / 2 - 0.5, "Primary energy"), (len(PE) + len(FE) / 2 - 0.5, "Final energy")):
                ax.text(xc, 1.0, txt, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.2, color=MUTED)
            ax.set_xticklabels([])
            ax.legend([Rectangle((0, 0), 1, 1, facecolor=GOLD_L, edgecolor=AR6_C, lw=0.6), Rectangle((0, 0), 1, 1, facecolor=TEALS[2], edgecolor=SYN, lw=0.6)],
                      ["AR6", "Synthetic (Qwen3.5-9B)"], frameon=False, loc="upper left", ncol=2, fontsize=5.8, handlelength=1.2, columnspacing=1.4, borderaxespad=0.2)
        else:
            ax.set_xticklabels([_short_var(v) for v in order_v], fontsize=5.4, rotation=32, ha="right", rotation_mode="anchor")
        plabel(ax, "ab"[j], pad=0.065)
    save(fig, "figS5_distributions")


def figS6():
    """a, b, test-set MAE and sMAPE of the four methods in each year; c, all
    decadal values of the finetuned model, AR6 vs synthetic."""
    ey = si_sheet("figS6_error_by_year"); order = ["KNN", "cVAE", "RAG", "Qwen3.5-9B"]
    fig = plt.figure(figsize=(185 * MM, 68 * MM))
    gs = fig.add_gridspec(1, 3, left=0.075, right=0.915, top=0.870, bottom=0.300, wspace=0.46)
    for j, (col, ylab, ymax, tks, dec) in enumerate((("mae", "MAE (EJ yr$^{-1}$)", 1.6, [0, 0.5, 1.0, 1.5], 1), ("smape_%", "sMAPE (%)", 38, [0, 10, 20, 30], 0))):
        ax = fig.add_subplot(gs[0, j]); lo_c, hi_c = ("mae_ci2.5", "mae_ci97.5") if col == "mae" else ("smape_ci2.5", "smape_ci97.5")
        for m in order:
            d = ey[ey.method.eq(m)].sort_values("year"); x = d.year.to_numpy(int); y = d[col].to_numpy(float)
            ax.fill_between(x, d[lo_c].to_numpy(float), d[hi_c].to_numpy(float), color=M_COLORS[m], alpha=0.25, lw=0, zorder=2)
            ax.plot(x, y, color=M_COLORS[m], lw=1.6 if m == "Qwen3.5-9B" else 1.2, marker="o", ms=2.6, mfc=M_COLORS[m], mec="white", mew=0.5,
                    zorder=5 if m == "Qwen3.5-9B" else 3)
        ax.set_xlim(2005, 2105); ax.set_xticks([2010, 2040, 2070, 2100]); ax.set_ylim(0, ymax); set_yticks2(ax, tks, dec=dec)
        ax.set_xlabel("Year"); ax.set_ylabel(ylab); plabel(ax, "ab"[j], pad=0.055 if j == 0 else 0.050)
    handles = [plt.Line2D([], [], color=M_COLORS[m], lw=1.6, marker="o", ms=2.6, mec="white", mew=0.5) for m in order]
    fig.legend(handles, [M_LABELS[m] for m in order], frameon=False, loc="upper center", bbox_to_anchor=(0.36, 0.115), ncol=4, fontsize=5.8,
               handlelength=1.8, columnspacing=1.4)
    # c: hexbin of all decadal values (log10(v + 0.01))
    dn = si_sheet("figS6_value_density"); g_ = si_sheet("figS6_value_density_grid").iloc[0]
    ax = fig.add_subplot(gs[0, 2]); f = lambda v: np.log10(np.clip(v, 0, None) + 0.01)
    hb = ax.hexbin(dn.ar6_log10, dn.synthetic_log10, C=dn.n_values, reduce_C_function=np.sum, gridsize=int(g_.gridsize),
                   extent=(g_.ar6_log10_min, g_.ar6_log10_max, g_.synthetic_log10_min, g_.synthetic_log10_max),
                   bins="log", cmap=CMAP_DENS, linewidths=0.1, edgecolors="none")
    lim = (f(0), f(200)); ax.plot(lim, lim, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=6)
    tk = [0, 0.1, 1, 10, 100]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mticker.FixedLocator([f(t) for t in tk])); axis.set_major_formatter(mticker.FixedFormatter(["0", "0.1", "1", "10", "100"]))
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("AR6 value (EJ yr$^{-1}$)"); ax.set_ylabel("Synthetic value (EJ yr$^{-1}$)", labelpad=1)
    ax.text(0.04, 0.96, f"n = {int(dn.n_values.sum()):,} values", transform=ax.transAxes, ha="left", va="top", fontsize=5.0, color=INK)
    cax = ax.inset_axes([1.04, 0.0, 0.045, 1.0]); cb = fig.colorbar(hb, cax=cax); cb.ax.tick_params(labelsize=5.4, length=1.5, width=0.4)
    cb.set_ticks([1, 10, 100, 1000, 10000]); cb.set_ticklabels(["1", "10", "100", "1,000", "10,000"]); cb.ax.minorticks_off()
    cb.set_label("Number of values per bin", fontsize=5.8, labelpad=2); cb.outline.set_linewidth(0.4)
    plabel(ax, "c", pad=0.050)
    save(fig, "figS6_year_values")


if __name__ == "__main__":
    which = sys.argv[1:] or ["S1", "S2", "S3", "S4", "S5", "S6"]
    for w in which:
        {"S1": figS1, "S2": figS2, "S3": figS3, "S4": figS4, "S5": figS5, "S6": figS6}[w]()
    print("SI figures written to", _HERE)
