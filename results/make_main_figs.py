"""
make_main_figs.py
─────────────────
Main-text figures 2-5 from main_figure_data.xlsx (built by build_main_data.py).
The workbook has one sheet per figure. A sheet holds the data tables of that
figure one below the other: a title row that names the panels and describes
the table, a header row and the data rows, with an empty row between tables.
The tables are identified by their order on the sheet (MAIN_BLOCKS). This
script needs only the workbook.
Fig. 1 is the framework schematic and is drawn elsewhere.
Colour rule: teal = synthetic (finetuned Qwen3.5-9B), gold = AR6, greys =
baselines, other IAM families and references; in Fig. 4 gold also marks the
lower-coverage / unseen strata.
  fig2_methods           RAG ablation (validation) + four-method test comparison
  fig3_fidelity          stacked group means, error by variable, extremum timing,
                         maximum parity (Qwen)
  fig4_generalisation    coverage matrix, test-set composition, metrics by stratum,
                         one-assumption examples
  fig5_usecase           four held-out IAM family x scenario cells vs the other families
Every panel is drawn from its block as stored: the statistics are computed in
build_main_data.py, and nothing is computed here beyond the layout.
make_si_figs.py imports this module.
Usage:  python3 make_main_figs.py [2 3 4 5]
"""

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
XLSX = _HERE / "main_figure_data.xlsx"

# ── palette ─────────────────────────────────────────────────────────────────
TEALS = ["#D9EAE5", "#AFD6CB", "#7FBDAD", "#4FA28D", "#25836C"]   # light→dark
GOLD, GOLD_D = "#E7CC85", "#D9B662"
GOLD_L = "#F2E4B8"
ROSE = "#C4756B"
CORAL = "#E8A79A"
BLUE_L, BLUE_D = "#D9E7F1", "#5E92BC"
CMAP_GOLD = LinearSegmentedColormap.from_list("gold_soft", ["#FAF4E3", "#B9954A"])
CMAP_CORAL = LinearSegmentedColormap.from_list("coral_soft", ["#FBF2EF", "#C97C68"])
CMAP_GRAY = LinearSegmentedColormap.from_list("gray_soft", ["#F3F5F7", "#5F6E7A"])
CMAP_BLUE = LinearSegmentedColormap.from_list("blue_soft", ["#EFF5FA", "#39749E"])
BLUE_SOFT = "#9DC3DC"
GRAY_L, GRAY_D = "#CDD3D8", "#9BA4AB"
INK, MUTED = "#2E3338", "#6A737B"
CMAP_TEAL = LinearSegmentedColormap.from_list("teal_soft", ["#EDF6F3", "#1E7A63"])

GREYS = ["#C3CAD1", "#9AA6B1", "#6E7F8E"]            # KNN, cVAE, RAG (light -> dark)
M_COLORS = {"KNN": GREYS[0], "cVAE": GREYS[1], "RAG": GREYS[2], "Qwen3.5-9B": TEALS[4]}
M_LABELS = {"KNN": "KNN", "cVAE": "cVAE", "RAG": "RAG, Gemini 3.5 Flash-Lite", "Qwen3.5-9B": "Finetuned Qwen3.5-9B"}
# ablation bars (fig 2 a-c): other settings light grey / adopted setting teal
ABL_L, ABL_D = GREYS[0], TEALS[4]
AR6_C = "#B9954A"                                     # AR6 lines and box edges (dark gold)
SYN = TEALS[4]                                        # synthetic

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Lato", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "axes.labelcolor": INK,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})
MM = 1 / 25.4
EKW = dict(elinewidth=0.7, capsize=1.6, capthick=0.7, ecolor=INK)

def set_yticks2(ax, ticks, dec=2):
    """Few, evenly spaced ticks; two decimals on fractional axes."""
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.{dec}f}" for t in ticks] if dec else
                       [f"{t:g}" for t in ticks])


def plabel(ax, s, pad=0.075, fig=None, yscale=1.24):
    """Panel label at a FIXED figure-space offset left of the GRID CELL.
    Two reasons not to use axes-fraction offsets or ax.get_position():
    rows with different axes widths would not line up, and an aspect-equal
    axes (the pie) is shrunk inside its cell, which would drop its label
    below its neighbours'. The subplotspec rect is immune to both."""
    if hasattr(ax, "get_subplotspec"):
        spec, fig = ax.get_subplotspec(), ax.figure
    else:
        spec = ax                                   # a SubplotSpec was passed
    pos = spec.get_position(fig)
    fig.text(pos.x0 - pad, pos.y0 + yscale * pos.height, s, fontsize=9,
             fontweight="bold", va="top", ha="left")

def blabel(ax, xs, vals, fmt="{:.2f}", dy=0.02, rot=0, err=None, fs=5.6, z=3):
    """Small rotated value label above each bar (the project's house style).
    err = upper error lengths; labels are lifted above the error-bar cap."""
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    if err is None:
        err = [0.0] * len(vals)
    for x, v, e in zip(xs, vals, err):
        ax.text(x, v + e + dy * span, fmt.format(v), ha="center", va="bottom",
                fontsize=fs, rotation=rot, color=INK, zorder=z,
                path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])

def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(_HERE / f"{stem}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("saved", stem)

# The data tables of each sheet, in their order on the sheet.
MAIN_BLOCKS = {
    "Fig2": ["fig2_rag_val_ablation", "fig2_km_availability", "fig2_test_four_methods"],
    "Fig3": ["fig3_group_trajectories", "fig3_error_by_variable", "fig3_extremum_timing", "fig3_maximum_parity",
             "fig3_maximum_statistics"],
    "Fig4": ["fig4_scenario_coverage", "fig4_test_composition", "fig4_group_metrics", "fig4_reference_lines",
             "fig4_contrast_examples"],
    "Fig5": ["fig5_heldout_deviations"],
}

def read_blocks(path, names):
    """Parse a figure workbook into {block name: DataFrame}. `names` gives, for
    each sheet, the names of its data tables in their order on the sheet."""
    out = {}
    for sheet_name, raw in pd.read_excel(path, sheet_name=None, header=None).items():
        empty = raw.isna().all(axis=1).to_numpy(); i, n, tables = 0, len(raw), []
        while i < n:
            if empty[i]: i += 1; continue
            hdr = [h for h in raw.iloc[i + 1].tolist() if not (isinstance(h, float) and np.isnan(h))]   # row i is the title
            j = i + 2
            while j < n and not empty[j]: j += 1
            block = raw.iloc[i + 2:j, :len(hdr)].copy()
            block.columns = [str(int(h)) if isinstance(h, float) and h.is_integer() else str(h) for h in hdr]   # year headers
            for c in block.columns:
                try: block[c] = pd.to_numeric(block[c])
                except (ValueError, TypeError): pass
            tables.append(block.reset_index(drop=True)); i = j
        expected = names.get(sheet_name, [])
        if len(tables) != len(expected):
            raise SystemExit(f"{Path(path).name}, sheet {sheet_name}: found {len(tables)} data tables, expected {len(expected)}")
        out.update(zip(expected, tables))
    return out

_BLOCKS = {}
_NAMES = {XLSX: MAIN_BLOCKS}
def sheet(name, path=None):
    """One data table of a figure workbook."""
    path = path or XLSX
    if path not in _BLOCKS:
        _BLOCKS[path] = read_blocks(path, _NAMES[path])
    return _BLOCKS[path][name].copy()

def err(rows):
    return [[r.mae - r["ci2.5"] for r in rows], [r["ci97.5"] - r.mae for r in rows]]


# ═══════════════════════════════ shared coverage panels: fig 4a and fig S3 ══
DIM_SHORT = {"Mitigation constraint": "Mitigation\nconstraint",
             "New climate policy": "New climate\npolicy",
             "New technology assumption": "New technology\nassumption",
             "Alternative socioeconomic pathway": "Alt. socioecon.\npathway",
             "None of the four stated": "None\nstated"}

REGION_SHORT = {
    "Countries of Latin America and the Caribbean": "Latin America",
    "Countries of South Asia; primarily India": "South Asia",
    "Countries of Sub-Saharan Africa": "Sub-Sah. Africa",
    "Countries of centrally-planned Asia; primarily China": "C.-planned Asia",
    "Countries of the Middle East; Iran, Iraq, Israel, Saudi Arabia, Qatar, etc.": "Middle East",
    "Eastern and Western Europe (i.e., the EU28)": "Europe",
    "North America; primarily the United States of America and Canada": "North America",
    "Other countries of Asia": "Other Asia",
    "Pacific OECD": "Pacific OECD",
    "Reforming Economies of Eastern Europe and the Former Soviet Union; primarily Russia": "Reform. econ.",
}


def _plot_dimension_family_heatmap(ax, d, compact=False, show_title=True):
    """Dimension x IAM-family counts with lower-coverage cells outlined."""
    xf = d[d.level.eq("dimension x family")]
    dd = d[d.level.eq("dimension")].sort_values("n_scenarios")
    ff = d[d.level.eq("family")].sort_values("n_scenarios")
    dims = list(dd.dimension)[::-1]                 # largest on top
    fams = list(ff.model_family)[::-1]
    piv = (xf.pivot(index="dimension", columns="model_family",
                    values="n_scenarios").loc[dims, fams])
    sparse = (xf.assign(is_sparse=xf.coverage_status.eq("Lower coverage"))
              .pivot(index="dimension", columns="model_family",
                     values="is_sparse").loc[dims, fams])
    vals = piv.to_numpy(float)
    show = np.ma.masked_where(vals == 0, vals)
    ax.imshow(show, cmap=CMAP_TEAL, aspect="auto", vmin=0, vmax=np.nanmax(vals))
    thr = 0.55 * np.nanmax(vals)
    for i, dim in enumerate(dims):
        for j, fam in enumerate(fams):
            v = int(vals[i, j])
            if v == 0:
                ax.text(j, i, "0", ha="center", va="center",
                        fontsize=5.2 if compact else 6,
                        color="#B7BEC5")
            else:
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=5.2 if compact else 6,
                        color="white" if v > thr else INK)
            if bool(sparse.loc[dim, fam]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=GOLD_D, lw=1.4))
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(fams, fontsize=4.7 if compact else 6,
                       rotation=38 if compact else 25, ha="right")
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels([DIM_SHORT[x] for x in dims],
                       fontsize=5.2 if compact else 6)
    title = "Dimension × IAM family coverage" if compact else (
             "Scenarios per dimension × IAM family "
             "(gold outline = lower coverage, grey 0 = gap)")
    if show_title:
        ax.set_title(title, pad=6 if compact else 8,
                     fontsize=6.8 if compact else 7.5)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)


# ═══════════════════════ fig 2: ablation + test comparison + distributions ══
def fig2():
    """a-c   RAG retrieval and prompt ablation (validation set); b is the
          k_m sweep scored only where a same-scenario neighbour exists.
    d-g   test-set accuracy of the four methods (bootstrap 95% CI)."""
    fig = plt.figure(figsize=(185 * MM, 107 * MM))
    gs_r1 = fig.add_gridspec(1, 3, left=0.100, right=0.985,
                             top=0.916, bottom=0.638, wspace=0.50)
    gs_r2 = fig.add_gridspec(1, 4, left=0.100, right=0.985,
                             top=0.379, bottom=0.100, wspace=0.50)

    # ── a-c: RAG ablation ───────────────────────────────────────────────────
    df = sheet("fig2_rag_val_ablation")
    r = lambda c: df[df.config == c].iloc[0]
    n_val = int(df.n_cases.iloc[0])            # 663 val cases, every config
    TFS = 6.8                                  # two-line titles, n on line 2
    t210 = r("topk210")
    axes = [fig.add_subplot(gs_r1[0, j]) for j in range(3)]

    ax = axes[0]
    rows = [r("topk010"), r("topk110"), t210, r("topk310")]
    y = [x_.mae for x_ in rows]; x = np.arange(4); e = err(rows)
    ax.bar(x, y, 0.45, color=[ABL_L, ABL_L, ABL_D, ABL_L],
           yerr=e, error_kw=EKW, zorder=3)
    ax.set_ylim(0, 6.2); set_yticks2(ax, [0, 2, 4, 6], dec=0)
    blabel(ax, x, y, err=e[1])
    ax.set_xticks(x); ax.set_xticklabels(["0", "1", "2", "3"])
    ax.set_xlabel("Different scenarios,\nsame IAM family ($k_s$)")
    ax.set_ylabel("MAE (EJ yr$^{-1}$)")
    ax.set_title(f"Same-family neighbours\n(n = {n_val})", pad=7,
                 fontsize=TFS)
    plabel(ax, "a")

    # b: the k_m sweep scored on the validation cases for which a neighbour of
    # the same scenario and a different IAM family exists (418 of 663). For the
    # other 245 cases the four prompts are identical.
    ax = axes[1]
    ka = sheet("fig2_km_availability").set_index("k_m").loc[[0, 1, 2, 3]]
    y = ka.mae.to_numpy(float); x = np.arange(4)
    e_ = [y - ka["ci2.5"].to_numpy(float), ka["ci97.5"].to_numpy(float) - y]
    ax.bar(x, y, 0.45, color=[ABL_L, ABL_D, ABL_L, ABL_L],
           yerr=e_, error_kw=EKW, zorder=3)
    ax.set_ylim(0, 0.95); set_yticks2(ax, [0, 0.30, 0.60, 0.90])
    blabel(ax, x, y, dy=0.03, err=e_[1])
    ax.set_xticks(x); ax.set_xticklabels(["0", "1", "2", "3"])
    ax.set_xlabel("Different IAM families,\nsame scenario ($k_m$)")
    ax.set_ylabel("MAE (EJ yr$^{-1}$)")
    ax.set_title(f"Same-scenario neighbours,\nwhere available (n = {int(ka.n_cases.iloc[0])})",
                 pad=7, fontsize=TFS)
    plabel(ax, "b", pad=0.045)

    ax = axes[2]
    rows = [t210, r("no_fingerprint (topk210)"), r("no_scenario_desc (topk210)"),
            r("norag")]
    y = [x_.mae for x_ in rows]; x = np.arange(4); e = err(rows)
    ax.bar(x, y, 0.45, color=[ABL_D, ABL_L, ABL_L, ABL_L],
           yerr=e, error_kw=EKW, zorder=3)
    ax.set_ylim(0, 6.2); set_yticks2(ax, [0, 2, 4, 6], dec=0)
    blabel(ax, x, y, err=e[1])
    ax.set_xticks(x)
    ax.set_xticklabels(["Full\nprompt", "w/o\nFP", "w/o scen.\ndesc.",
                        "w/o\nretrieval"], fontsize=6)
    ax.set_ylabel("MAE (EJ yr$^{-1}$)")
    ax.set_title(f"Removing prompt components\n(n = {n_val})", pad=7,
                 fontsize=TFS)
    plabel(ax, "c", pad=0.045)

    # ── d-g: four-method test comparison ────────────────────────────────────
    dm = sheet("fig2_test_four_methods").copy()
    names = {"KNN (cond1 k2 uniform a0.75)": "KNN",
             "cVAE (cond1 lr1e-4 hid2048)": "cVAE",
             "RAG (topk210, gemini-3.5-flash-lite)": "RAG",
             "Finetune Qwen3.5-9B (lr5e-4)": "Qwen3.5-9B"}
    dm["m"] = dm.method.map(names)
    order = ["KNN", "cVAE", "RAG", "Qwen3.5-9B"]
    dm = dm.set_index("m").loc[order]
    metrics = [("mae", "MAE (EJ yr$^{-1}$) $\\downarrow$", (0, 1.15), "{:.2f}", [0, 0.50, 1.00], 2),
               ("smape_%", "sMAPE (%) $\\downarrow$", (0, 30), "{:.2f}", [0, 10, 20, 30], 0),
               ("pearson", "Pearson $r$ $\\uparrow$", (0.88, 1.005), "{:.3f}", [0.90, 0.95, 1.00], 2),
               ("spearman", "Spearman $\\rho$ $\\uparrow$", (0.86, 1.005), "{:.3f}", [0.90, 0.95, 1.00], 2)]
    x = np.arange(4)
    ax_m = [fig.add_subplot(gs_r2[0, j]) for j in range(4)]
    for ax, (m, lab, ylim, fmt, tks, dec) in zip(ax_m, metrics):
        y = dm[m].to_numpy(float)
        lo = y - dm[f"{m}_ci2.5"].to_numpy(float)
        hi = dm[f"{m}_ci97.5"].to_numpy(float) - y
        ax.bar(x, y, 0.62, color=[M_COLORS[o] for o in order],
               yerr=[lo, hi], error_kw=EKW, zorder=3)
        ax.set_ylim(*ylim); set_yticks2(ax, tks, dec=dec)
        blabel(ax, x, y, fmt=fmt, dy=0.025, err=hi)
        ax.set_xticks([])
        ax.set_title(lab, pad=8, fontsize=7.5)
    for ax, s_, pd_ in zip(ax_m, "defg", [0.075, 0.045, 0.045, 0.045]):
        plabel(ax, s_, pad=pd_)
    handles = [Rectangle((0, 0), 1, 1, color=M_COLORS[o]) for o in order]
    fig.legend(handles, [M_LABELS[o] for o in order], frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=4, handlelength=1.1,
               columnspacing=1.8)

    save(fig, "fig2_methods")


def _short_var(v):
    return (v.replace("Primary Energy|", "").replace("Final Energy|", "")
             .replace("Residential and Commercial", "Res.&Comm.")
             .replace("Transportation", "Transport")
             .replace("|Electricity", "|Elec."))


def box_stats(ax, x, lo, q1, med, q3, hi, face, edge, width=0.6, lw=0.7, z=3):
    ax.add_patch(Rectangle((x - width / 2, q1), width, max(q3 - q1, 1e-9), facecolor=face, edgecolor=edge, lw=lw, zorder=z))
    ax.plot([x - width / 2, x + width / 2], [med, med], color=edge, lw=lw + 0.4, zorder=z + 1)
    ax.plot([x, x], [lo, q1], color=edge, lw=lw * 0.8, zorder=z - 1)
    ax.plot([x, x], [q3, hi], color=edge, lw=lw * 0.8, zorder=z - 1)
    cap = width * 0.3
    for yv in (lo, hi): ax.plot([x - cap, x + cap], [yv, yv], color=edge, lw=lw * 0.8, zorder=z - 1)

# ═══════════════════════════════════ fig 3: fidelity of the finetuned model ═════
YEARS = [str(y) for y in range(2010, 2101, 10)]

def fig3():
    fig = plt.figure(figsize=(185 * MM, 194 * MM))
    # a: mean energy per case, stacked by group: three groups of primary energy sources (left) and the three final
    #    energy sectors (right). Areas = synthetic, dashed lines = AR6.
    gt_ = sheet("fig3_group_trajectories"); SHADES = [TEALS[4], TEALS[2], TEALS[0]]; col_l, col_r = [], []   # axes of the left and right column
    for r, (energy, title, left, right) in enumerate((("primary", "Primary energy", 0.085, 0.385), ("final", "Final energy", 0.585, 0.885))):
        ax = fig.add_subplot(fig.add_gridspec(1, 1, left=left, right=right, top=0.925, bottom=0.745)[0, 0]); (col_l, col_r)[r].append(ax)
        d_ = gt_[gt_.energy.eq(energy)]; groups = list(dict.fromkeys(d_.group))
        n_cases = int(d_.n_cases.iloc[0]); assert d_.n_cases.eq(n_cases).all()
        yrs = np.sort(d_.year.unique()); base_s = np.zeros(len(yrs)); base_a = np.zeros(len(yrs))
        for g, shade in zip(groups, SHADES):
            vs = d_[d_.group.eq(g) & d_.source.eq("synthetic")].set_index("year").loc[yrs, "mean"].to_numpy(float)
            va = d_[d_.group.eq(g) & d_.source.eq("AR6")].set_index("year").loc[yrs, "mean"].to_numpy(float)
            ax.fill_between(yrs, base_s, base_s + vs, facecolor=shade, edgecolor="white", lw=0.6, zorder=2)
            ax.plot(yrs, base_a + va, color=AR6_C, lw=1.1, ls=(0, (2.6, 1.8)), zorder=5)
            ax.text(2103, base_s[-1] + vs[-1] / 2, g, fontsize=5.8, ha="left", va="center", color=INK)
            base_s, base_a = base_s + vs, base_a + va
        ax.set_xlim(2010, 2100); ax.set_xticks([2010, 2050, 2100]); ax.set_ylim(0, 1.12 * float(max(base_s.max(), base_a.max())))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))
        ax.set_xlabel("Year", labelpad=1.5); ax.set_ylabel("Mean per case (EJ yr$^{-1}$)")
        ax.set_title(f"{title} (n = {n_cases:,} cases)", pad=6, fontsize=7.5)
        if r == 0:
            ax.legend([Rectangle((0, 0), 1, 1, facecolor=TEALS[2], edgecolor="white", lw=0.6), plt.Line2D([], [], color=AR6_C, lw=1.1, ls=(0, (2.6, 1.8)))],
                      ["Synthetic (areas)", "AR6 (dashed lines)"], frameon=False, loc="upper left", fontsize=5.8, handlelength=1.6, labelspacing=0.35, borderaxespad=0.3)
            plabel(ax, "a", pad=0.065)

    # b: per-variable MAE / sMAPE
    bv = sheet("fig3_error_by_variable").set_index("variable")
    PE = sorted(v for v in bv.index if v.startswith("Primary")); FE = sorted(v for v in bv.index if v.startswith("Final")); order_v = PE + FE
    xv = np.arange(len(order_v))
    gs_b = fig.add_gridspec(1, 1, left=0.085, right=0.945, top=0.640, bottom=0.495)
    SMAPE_C = AR6_C                                   # sMAPE markers and right axis in gold
    ax = fig.add_subplot(gs_b[0, 0]); axr = ax.twinx(); col_l.append(ax)
    axr.spines["top"].set_visible(False); axr.spines["right"].set_visible(True); axr.spines["right"].set_color(SMAPE_C)
    axr.tick_params(axis="y", colors=SMAPE_C, width=0.6); ax.spines["left"].set_color(SYN); ax.tick_params(axis="y", colors=SYN, width=0.6)
    v_ = bv.loc[order_v, "mae"].to_numpy(float); lo = v_ - bv.loc[order_v, "mae_ci2.5"].to_numpy(float); hi = bv.loc[order_v, "mae_ci97.5"].to_numpy(float) - v_
    ax.bar(xv - 0.16, v_, 0.36, color=TEALS[3], yerr=[lo, hi], error_kw=dict(elinewidth=0.5, capsize=1.0, capthick=0.5, ecolor=INK), zorder=3)
    v_ = bv.loc[order_v, "smape_%"].to_numpy(float); lo = v_ - bv.loc[order_v, "smape_ci2.5"].to_numpy(float); hi = bv.loc[order_v, "smape_ci97.5"].to_numpy(float) - v_
    axr.errorbar(xv + 0.22, v_, yerr=[lo, hi], fmt="o", ms=3.4, mfc=GOLD, mec=SMAPE_C, mew=0.9, ecolor=SMAPE_C, elinewidth=0.6, capsize=1.0, capthick=0.6, zorder=5)
    ax.set_ylim(0, 1.45); set_yticks2(ax, [0, 0.5, 1.0], dec=1); axr.set_ylim(0, 36.25); axr.set_yticks([0, 10, 20, 30])
    ax.set_ylabel("MAE (EJ yr$^{-1}$)", color=SYN); axr.set_ylabel("sMAPE (%)", color=SMAPE_C, rotation=270, labelpad=10)
    ax.axvline(len(PE) - 0.5, color=GRAY_L, lw=0.6, ls=(0, (3, 2)), zorder=0)
    for xc, txt in ((len(PE) / 2 - 0.5, "Primary energy"), (len(PE) + len(FE) / 2 - 0.5, "Final energy")):
        ax.text(xc, 1.0, txt, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.2, color=MUTED)
    ax.set_xlim(-0.7, len(order_v) - 0.3); ax.set_xticks(xv)
    ax.set_xticklabels([_short_var(v) for v in order_v], fontsize=5.4, rotation=32, ha="right", rotation_mode="anchor"); ax.tick_params(axis="x", length=0)
    ax.set_zorder(axr.get_zorder() + 1); ax.patch.set_visible(False)
    ax.legend([Rectangle((0, 0), 1, 1, color=TEALS[3]), plt.Line2D([], [], marker="o", ls="none", ms=3.4, mfc=GOLD, mec=SMAPE_C, mew=0.9)],
              ["MAE (left axis)", "sMAPE (right axis)"], frameon=False, fontsize=5.8, loc="upper left", ncol=2, handlelength=1.1,
              columnspacing=1.2, borderaxespad=0.2)
    plabel(ax, "b", pad=0.065)

    # c, d: extremum timing and maximum-value parity, in the same columns as the two panels of a
    # c: extremum timing
    et = sheet("fig3_extremum_timing").pivot(index="offset_years", columns="extremum", values="trajectories_%")
    offs = [-30, -20, -10, 0, 10, 20, 30]
    ax = fig.add_subplot(fig.add_gridspec(1, 1, left=0.085, right=0.385, top=0.365, bottom=0.079)[0, 0]); xo = np.arange(len(offs)); w = 0.38
    pk = [float(et.loc[k, "maximum"]) for k in offs]; tr = [float(et.loc[k, "minimum"]) for k in offs]
    ax.bar(xo - w / 2, pk, w, color=SYN, zorder=3, label="Maximum")
    ax.bar(xo + w / 2, tr, w, color=GOLD, zorder=3, label="Minimum")
    pk10, tr10 = sum(pk[2:5]), sum(tr[2:5])
    for k, (val, col, lab) in enumerate(((pk10, SYN, "Max."), (tr10, AR6_C, "Min."))):
        yb = 102 + 15 * k
        ax.plot([xo[2] - w, xo[4] + w], [yb, yb], color=col, lw=0.7, zorder=4, clip_on=False)
        for xe_ in (xo[2] - w, xo[4] + w): ax.plot([xe_, xe_], [yb - 2.5, yb], color=col, lw=0.7, zorder=4)
        ax.text(xo[3], yb + 1.5, f"{lab} {val:.0f}% within $\\pm$10 yr", ha="center", va="bottom", fontsize=5.4, color=col)
    ax.set_xticks(xo); ax.set_xticklabels(["$\\leq$$-$30", "$-$20", "$-$10", "0", "+10", "+20", "$\\geq$+30"], fontsize=6.2)
    ax.set_xlim(-0.6, len(offs) - 0.4); ax.set_ylim(0, 137); set_yticks2(ax, [0, 25, 50, 75, 100], dec=0)
    ax.set_xlabel("Synthetic minus AR6 extremum year (yr)"); ax.set_ylabel("Trajectories (%)")
    ax.legend(frameon=False, fontsize=5.8, loc="center left", bbox_to_anchor=(0.0, 0.52), handlelength=1.0, labelspacing=0.35, borderaxespad=0.2)
    ax.set_title("Extremum timing", pad=8, fontsize=7.5); plabel(ax, "c", pad=0.065); col_l.append(ax)

    # d: maximum-value parity
    pk_ = sheet("fig3_maximum_parity"); gpk = pk_.ar6_maximum.to_numpy(float); ppk = pk_.synthetic_maximum.to_numpy(float)
    st = sheet("fig3_maximum_statistics").set_index("quantity").value
    ax = fig.add_subplot(fig.add_gridspec(1, 1, left=0.585, right=0.885, top=0.365, bottom=0.079)[0, 0]); hi_ = 150
    ax.plot([0, hi_], [0, hi_], color=INK, lw=0.8, ls=(0, (3, 2)), zorder=6)
    ax.scatter(gpk, ppk, s=4, color=SYN, alpha=0.30, linewidths=0, zorder=3, rasterized=True)
    ax.set_xlim(-3, hi_); ax.set_ylim(-3, hi_); ax.set_aspect("equal", adjustable="box")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mticker.FixedLocator([0, 50, 100, 150])); axis.set_major_formatter(mticker.FixedFormatter(["0", "50", "100", "150"]))
    ax.set_xlabel("AR6 maximum (EJ yr$^{-1}$)"); ax.set_ylabel("Synthetic maximum (EJ yr$^{-1}$)", labelpad=1)
    ax.text(0.04, 0.96, f"median rel. error {100 * float(st['maximum_median_relative_error']):.1f}%\n$R^2$ = {float(st['maximum_r2']):.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=5.4, color=INK, linespacing=1.25)
    ax.set_title("Maximum value", pad=8, fontsize=7.5); plabel(ax, "d", pad=0.065); col_r.append(ax)
    fig.align_ylabels(col_l); fig.align_ylabels(col_r)
    save(fig, "fig3_fidelity")

# ══════════════════════════════════ fig 4: generalisation of Qwen3.5-9B ═════
# dark shades identify the coverage split; light shades identify seen/unseen.
GRP_BARS = [("All test", "All test", "#6E7A85"),
            ("Lower coverage", "Lower coverage", GOLD_D),
            ("Higher coverage", "Higher coverage", TEALS[4]),
            ("Unseen", "Unseen", GOLD),
            ("Seen", "Seen", TEALS[2])]


def _test_case_composition():
    """Composition of all region-specific cases in the test set."""
    n = sheet("fig4_test_composition").set_index("group").n_cases
    return {"total": int(n["All test"]), "seen": int(n["Seen"]), "unseen": int(n["Unseen"]),
            "sparse": int(n["Lower coverage"]), "dense": int(n["Higher coverage"])}


def _composition_double_donut(ax, counts, compact=True):
    """Two nested rings on the 1,351 test cases: outer ring = unseen / seen
    scenario, inner ring = lower / higher coverage. Rings on the left, colour
    key with counts on the right."""
    total = counts["total"]
    outer_v, outer_c, outer_l = [counts["unseen"], counts["seen"]], [GOLD, TEALS[2]], ["Unseen", "Seen"]
    inner_v, inner_c, inner_l = [counts["sparse"], counts["dense"]], [GOLD_D, TEALS[4]], ["Lower coverage", "Higher coverage"]
    r_out, w = 0.80, 0.21
    ax.pie(outer_v, colors=outer_c, startangle=90, counterclock=False, radius=r_out, center=(0, 0),
           wedgeprops=dict(width=w, edgecolor="white", linewidth=1.1))
    ax.pie(inner_v, colors=inner_c, startangle=90, counterclock=False, radius=r_out - w - 0.04, center=(0, 0),
           wedgeprops=dict(width=w, edgecolor="white", linewidth=1.1))
    ax.text(0, 0.06, f"{total}", ha="center", va="center", fontsize=8.0, fontweight="bold", color=INK)
    ax.text(0, -0.15, "test cases", ha="center", va="center", fontsize=5.0, color=MUTED)
    # colour key, two groups
    fig = ax.figure; pos = ax.get_position()
    ratio = (pos.width * fig.get_figwidth()) / (pos.height * fig.get_figheight())
    ax.set_ylim(-0.92, 0.92); ax.set_xlim(-0.98, -0.98 + 1.84 * ratio); ax.set_anchor("W")
    xm, xt = 1.08, 1.20
    rows = [("Outer ring: scenario", None, None), (outer_l[0], outer_c[0], outer_v[0]), (outer_l[1], outer_c[1], outer_v[1]),
            ("Inner ring: coverage", None, None), (inner_l[0], inner_c[0], inner_v[0]), (inner_l[1], inner_c[1], inner_v[1])]
    ys = [0.66, 0.46, 0.28, -0.06, -0.26, -0.44]
    for (lab, col, v), y in zip(rows, ys):
        if col is None:
            ax.text(xm - 0.04, y, lab, ha="left", va="center", fontsize=5.4, color=MUTED)
        else:
            ax.scatter([xm], [y], marker="s", s=22, color=col, edgecolors="white", linewidths=0.4, clip_on=False)
            ax.text(xt, y, f"{lab}  {v} ({v / total:.0%})", ha="left", va="center", fontsize=5.6, color=INK)
    ax.set_axis_off()


# Direct variables for which the scenario descriptions support an ex-ante
# directional hypothesis.  Variables affected only through system-wide
# re-optimisation are deliberately excluded from the directional test.

def fig4():
    """Coverage and Qwen3.5-9B generalisation on lower/higher coverage and seen/unseen
    splits (a-f), against two references: the distance of every test case to
    its description-most-similar sibling scenario (same IAM family and region)
    and the spread between two IAMs running the same scenario; g-j, four
    example pairs of a test case and a retrieved training run that differs in
    one assumption.
    The last one is far outside the metric range, so each of the four metric
    panels carries a broken axis: a narrow band holds it and the main axis
    holds the subgroup estimates."""
    df = sheet("fig4_group_metrics").set_index("group")
    # Reference lines (AR6 data only, block fig4_reference_lines). Dotted: for
    # every test case, the scenario with the most similar description among
    # those the same IAM family ran in the same region, scored against the
    # case. Dash-dotted (broken axis): the average output of the other IAM
    # families for the same scenario and region.
    ref = sheet("fig4_reference_lines").set_index("reference")
    pair = ref.loc["Similar scenario, same IAM family"]; iam = ref.loc["Same scenario, different IAM family"]
    # All four panels use dots. Bars imply a zero baseline: c/d are bounded
    # near 1 with a truncated axis, and with a break underneath a bar would
    # read as if it continued into the band. Dots have no baseline, so the
    # break stays honest - and the four panels keep one visual grammar.
    #      col        title                              main ylim   fmt      ticks              dec side     half   kind
    metrics = [
        # y limits chosen so that the dotted reference sits inside the main axis
        ("mae",      "MAE (EJ yr$^{-1}$) $\\downarrow$", (0, 1.25),  "{:.2f}", [0, 0.50, 1.00],   2, "top",    0.55, "dot"),
        ("smape_%",  "sMAPE (%) $\\downarrow$",          (0, 22.5),  "{:.2f}", [0, 10, 20],       0, "top",    9.0,  "dot"),
        ("pearson",  "Pearson $r$ $\\uparrow$",          (0.86, 1.004), "{:.3f}", [0.90, 0.95, 1.00], 2, "bottom", 0.07, "dot"),
        ("spearman", "Spearman $\\rho$ $\\uparrow$",     (0.86, 1.004), "{:.3f}", [0.90, 0.95, 1.00], 2, "bottom", 0.07, "dot"),
    ]

    fig = plt.figure(figsize=(185 * MM, 203.1 * MM))
    counts = _test_case_composition()
    gs_comp = fig.add_gridspec(1, 4, left=0.090, right=0.990,
                               top=0.9039, bottom=0.688, wspace=0.48)

    coverage = sheet("fig4_scenario_coverage")
    ax = fig.add_subplot(gs_comp[0, 0:2])
    outer = gs_comp[0, 0:2].get_position(fig)
    ax.set_position([outer.x0, outer.y0 + 0.0403,
                     outer.width, outer.height - 0.0979])
    _plot_dimension_family_heatmap(ax, coverage, compact=True,
                                   show_title=False)
    plabel(gs_comp[0, 0:2], "a", pad=0.025, fig=fig, yscale=1.035)

    ax = fig.add_subplot(gs_comp[0, 2:4])
    outer = gs_comp[0, 2:4].get_position(fig)
    ax.set_position([outer.x0 + 0.015, outer.y0 + 0.010,
                     outer.width - 0.015, 0.8700 - (outer.y0 + 0.010)])
    _composition_double_donut(ax, counts)
    plabel(gs_comp[0, 2:4], "b", pad=0.025, fig=fig, yscale=1.035)

    # These axes have deliberately different internal aspect ratios.  Shared
    # figure-space titles keep the three top-row headings on one baseline.
    title_y = 0.8819
    for spec, title in (
            (gs_comp[0, 0:2], "Dimension × IAM family coverage"),
            (gs_comp[0, 2:4], "Test-set composition")):
        pos = spec.get_position(fig)
        fig.text((pos.x0 + pos.x1) / 2, title_y, title,
                 ha="center", va="bottom", fontsize=7.5)

    gs = fig.add_gridspec(1, 4, left=0.100, right=0.985,
                          top=0.5968, bottom=0.4392, wspace=0.42)
    x = np.array([0.0, 1.5, 2.3, 3.8, 4.6])
    XBLOCKS = ((0.0, 0.0, "All"), (1.5, 2.3, "Coverage"), (3.8, 4.6, "Scenario"))
    brk = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=4.5, linestyle="none",
               color=INK, mec=INK, mew=0.7, clip_on=False)
    for jx, (m, lab, ylim, fmt, tks, dec, side, half, kind) in enumerate(metrics):
        ratios = [1, 4.4] if side == "top" else [4.4, 1]
        sub = gs[0, jx].subgridspec(2, 1, height_ratios=ratios, hspace=0.12)
        ax_band = fig.add_subplot(sub[0 if side == "top" else 1, 0])
        ax = fig.add_subplot(sub[1 if side == "top" else 0, 0])

        vals = [df.loc[g, m] for g, _, _ in GRP_BARS]
        cols = [c for _, _, c in GRP_BARS]
        if kind == "bar":
            ax.bar(x, vals, 0.58, color=cols, zorder=3)
        else:
            ax.scatter(x, vals, s=42, c=cols, zorder=3,
                       edgecolors="white", linewidths=0.7)
        ax.axhline(pair[m], color=ROSE, lw=0.8, ls=(0, (1, 1.5)), zorder=4)
        ax.set_ylim(*ylim)
        set_yticks2(ax, tks, dec=dec)
        # z=6: labels sit above the reference lines (white halo keeps them legible)
        blabel(ax, x, vals, fmt=fmt, dy=0.035, fs=5.2, z=6)
        ax.set_xlim(-0.6, 5.2)

        v = float(iam[m])
        ax_band.axhline(v, color=BLUE_D, lw=0.9, ls=(0, (4, 1.4, 1, 1.4)),
                        zorder=4)
        ax_band.set_ylim(v - half, v + half)
        ax_band.set_yticks([v])
        ax_band.set_yticklabels([f"{v:.{dec}f}"])
        ax_band.set_xlim(*ax.get_xlim())
        ax_band.set_xticks([])

        # only the outer edges keep a spine; break marks on the facing ones
        # no x tick labels: five categories in a 31 mm panel collide, and the
        # colours are already spelled out in the legend (same as fig2 d-g)
        ax.set_xticks([])
        ax_band.set_xticks([])
        top_ax, bot_ax = (ax_band, ax) if side == "top" else (ax, ax_band)
        top_ax.spines["bottom"].set_visible(False)
        top_ax.tick_params(axis="x", length=0)
        bot_ax.spines["top"].set_visible(False)
        top_ax.plot([0], [0], transform=top_ax.transAxes, **brk)
        bot_ax.plot([0], [1], transform=bot_ax.transAxes, **brk)

        # block labels under the lowest axis: a short bracket under each pair
        for x0, x1, blab in XBLOCKS:
            line_tr = mtransforms.offset_copy(bot_ax.get_xaxis_transform(), fig=fig,
                                              y=-5, units="points")
            if x1 > x0:
                bot_ax.plot([x0, x1], [0, 0], transform=line_tr, color=MUTED,
                            lw=0.6, clip_on=False)
            txt_tr = mtransforms.offset_copy(bot_ax.get_xaxis_transform(), fig=fig,
                                             y=-7, units="points")
            bot_ax.text((x0 + x1) / 2, 0, blab, transform=txt_tr, ha="center",
                        va="top", fontsize=5.8, color=MUTED)

        top_ax.set_title(lab, pad=8, fontsize=7.5)
        plabel(gs[0, jx], "cdef"[jx], pad=0.055, fig=fig, yscale=1.384)

    point_h = [plt.Line2D([], [], color=c, marker="o", ms=4.5, ls="none",
                          mec="white", mew=0.7) for _, _, c in GRP_BARS]
    ref_h = [plt.Line2D([], [], color=ROSE, lw=0.8, ls=(0, (1, 1.5))),
             plt.Line2D([], [], color=BLUE_D, lw=0.9,
                        ls=(0, (4, 1.4, 1, 1.4)))]
    # Legend entries are filled down each column.  A blank second row keeps
    # All test cases in its own column; the remaining concepts align as pairs.
    blank_h = plt.Line2D([], [], ls="none", alpha=0)
    h = [point_h[0], blank_h,
         point_h[1], point_h[2],
         point_h[3], point_h[4],
         ref_h[0], ref_h[1]]
    lab = ["All test", " ",
           "Lower coverage", "Higher coverage",
           "Unseen", "Seen",
           "Similar scenario, same IAM family",
           "Same scenario, different IAM family"]
    fig.legend(h, lab, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.3475), ncol=4, handlelength=1.1,
               labelspacing=0.6, columnspacing=1.8, fontsize=6)

    import textwrap
    ex = sheet("fig4_contrast_examples")
    TITLE_OVERRIDE = {("SSP2_noint_lc_15", "SSP2_noint_lc_50"):
                      "Lower vs higher\ncarbon tax"}   # explicit line break: not re-wrapped
    AR6_C, B_C = "#B9954A", "#8A949C"
    B_LS = (0, (1, 1.3))
    YRS = sorted(ex.year.unique())
    gs3 = fig.add_gridspec(1, 4, left=0.075, right=0.985, top=0.2513,
                           bottom=0.1065, wspace=0.42)
    for j, (pn, g) in enumerate(ex.groupby("panel", sort=True)):
        ax = fig.add_subplot(gs3[0, j])
        s = {k: v.sort_values("year").value.to_numpy(float)
             for k, v in g.groupby("series")}
        yb, ya, yp = (s["training run (B)"], s["AR6 test case (A)"],
                      s["synthetic test case (A)"])
        ax.fill_between(YRS, yb, ya, color=AR6_C, alpha=0.14, lw=0, zorder=1)
        ax.plot(YRS, yb, color=B_C, lw=1.4, ls=B_LS, zorder=2)
        # AR6 dashed drawn above the synthetic line, so it stays visible
        # where the two coincide
        ax.plot(YRS, yp, color=TEALS[4], lw=1.6, zorder=3)
        ax.plot(YRS, ya, color=AR6_C, lw=1.2, ls=(0, (3, 1.5)), zorder=4)
        r = g.iloc[0]
        title = TITLE_OVERRIDE.get((r.training_scenario_B, r.test_scenario_A),
                                   r.label)
        lines = (title.split("\n") if "\n" in title
                 else textwrap.wrap(title.replace("CO2", "CO$_2$"), 24))
        ax.set_title("\n".join(lines),
                     fontsize=6.6, pad=25, linespacing=1.1)
        PLAIN_PAIR = {
            ("SSP2_noint_lc_15", "SSP2_noint_lc_50"): "carbon tax from 2030: 15 → 50 USD per tCO$_2$",
            ("EN_NPi2020_600f_COV", "EN_NPi2020_600f"): "600 GtCO$_2$ budget: COVID-19 recovery → default",
            ("EN_NPi2020_800f", "EN_NPi2020_800"): "800 GtCO$_2$ budget: overshoot allowed → net zero",
            ("R2p1_SSP1-NPi", "R2p1_SSP1-NDC"): "SSP1: current policies → NDCs",
        }
        ax.text(0.5, 1.03,
                f"{r.model_family}, {REGION_SHORT.get(r.region, r.region)}\n"
                f"{_plain_var(r.variable)}\n"
                f"{PLAIN_PAIR.get((r.training_scenario_B, r.test_scenario_A), r.training_scenario_B + ' → ' + r.test_scenario_A)}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=4.9,
                color=MUTED, linespacing=1.15)
        ax.set_xticks([2020, 2060, 2100])
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        ax.set_box_aspect(0.9)
        if j == 0:
            ax.set_ylabel("EJ yr$^{-1}$")
        plabel(gs3[0, j], "ghij"[j], pad=0.030 if j == 0 else 0.035, fig=fig,
               yscale=1.56)
    fig.legend([plt.Line2D([], [], color=B_C, lw=1.4, ls=B_LS),
                plt.Line2D([], [], color=AR6_C, lw=1.3, ls=(0, (3, 1.5))),
                plt.Line2D([], [], color=TEALS[4], lw=1.6),
                plt.Rectangle((0, 0), 1, 1, color=AR6_C, alpha=0.14, lw=0)],
               ["Training run (B), retrieved neighbour", "Test case, AR6 (A)",
                "Test case, synthetic (A)", "AR6 change from B to A"],
               frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.062),
               ncol=4, handlelength=2.0, columnspacing=1.8, fontsize=6)


    save(fig, "fig4_generalisation")


def _plain_var(v):
    """AR6 variable name -> plain wording for panel subtitles."""
    parts = [p.replace("w/o CCS", "without CCS").replace("w/ CCS", "with CCS")
             for p in v.split("|")]
    out = parts[0].capitalize() + ", " + ", ".join(p.lower() for p in parts[1:])
    return out.replace("ccs", "CCS")


# ═══════════════════════════════ fig 5: held-out cells vs the other families ══
def fig5():
    """Four seen-stratum test cells treated as
    missing IAM family x scenario combinations, compared with the other
    families that ran the scenario (a-d)."""
    H = 76.0
    fr = lambda mm: mm / H
    fig = plt.figure(figsize=(185 * MM, H * MM))
    _heldout_row(fig, top=fr(55.0), bottom=fr(20.0), legend_y=fr(4.0), letters="abcd")
    save(fig, "fig5_usecase")

def _heldout_row(fig, top, bottom, legend_y, letters="abcd"):
    """Four seen-stratum test cells treated as missing IAM family x scenario
    combinations (ENGAGE). y = deviation from the median of the other families
    that ran the scenario (percentage points; per cent for total final energy)
    on a symlog axis (linear within +-1; limit 2.3 x the largest deviation so
    nothing sits on the frame). Grey bar = the other families' range, ticks =
    each of them, teal = held-out family's synthetic value, gold hollow = its
    AR6 truth."""
    Pf = sheet("fig5_heldout_deviations"); Cl = Pf[["family", "scenario", "n_other_families"]].drop_duplicates()
    ORDER = ["fossil_unab_share", "ccs_share", "sw_share", "bio_share", "nuc_share", "FE_total", "ind_elec_share", "tr_elec_share", "bld_elec_share"]
    SHORT = {"fossil_unab_share": "Fossil w/o CCS", "ccs_share": "Fossil w/ CCS", "sw_share": "Wind + solar", "bio_share": "Biomass",
             "nuc_share": "Nuclear", "FE_total": "Final energy (%)", "ind_elec_share": "Elec. industry", "tr_elec_share": "Elec. transport", "bld_elec_share": "Elec. buildings"}
    def plain(scen):
        if scen == "EN_NPi2100": return "no carbon budget\nimplemented national policies continued"
        pre, bud, f = re.match(r"EN_(NPi2020|INDCi2030)_(\d+)(f?)", scen).groups()
        return f"{bud} GtCO$_2$ budget\n{'NDCs until 2030' if pre == 'INDCi2030' else 'immediate action'}, {'overshoot allowed' if f else 'no overshoot'}"
    gsh = fig.add_gridspec(1, 4, left=0.075, right=0.985, top=top, bottom=bottom, wspace=0.30)
    axes, ymax = [], 1.0
    for j, c in enumerate(Cl.itertuples()):
        ax = fig.add_subplot(gsh[0, j]); axes.append(ax)
        d = Pf[(Pf.family == c.family) & (Pf.scenario == c.scenario)]
        inds = [k for k in ORDER if k in set(d.indicator)]
        ax.axhline(0, color=GRAY_D, lw=0.6, zorder=0.5)
        singles = set()
        for i, k in enumerate(inds):
            dk = d[d.indicator == k]
            o = dk[dk.series == "other family"].deviation.to_numpy(float)
            if len(o) == 1: singles.add(k)
            ax.plot([i, i], [o.min(), o.max()], color=GRAY_L, lw=5, solid_capstyle="butt", zorder=1)
            ax.scatter([i] * len(o), o, marker="_", s=46, color=GRAY_D, lw=0.9, zorder=2)
            syn, tru = float(dk[dk.series == "synthetic"].deviation.iloc[0]), float(dk[dk.series == "AR6 truth"].deviation.iloc[0])
            ymax = max(ymax, abs(syn), abs(tru), abs(o).max())
            ax.scatter(i, syn, s=24, color=SYN, zorder=4, edgecolors="white", linewidths=0.5)
            ax.scatter(i, tru, s=30, facecolors="none", edgecolors=AR6_C, linewidths=1.1, zorder=5)
        ax.set_xticks(range(len(inds))); ax.set_xticklabels([SHORT[k] + ("*" if k in singles else "") for k in inds], fontsize=5.4, rotation=50, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, len(inds) - 0.4)
        ax.set_title(f"{c.family}\n{plain(c.scenario)}", fontsize=6.0, pad=11, linespacing=1.2)
        ax.text(0.5, 1.02, f"range of {c.n_other_families} other IAM families", transform=ax.transAxes, ha="center", va="bottom", fontsize=4.9, color=MUTED)
        if j == 0: ax.set_ylabel("Deviation from the other families'\nmedian in 2050 (pp; % for final energy)", linespacing=1.15)
        plabel(gsh[0, j], letters[j], pad=0.055 if j == 0 else 0.03, fig=fig, yscale=1.4)
    top_lim = ymax * 2.3
    for ax in axes:
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.8); ax.set_ylim(-top_lim, top_lim)
        ticks = [t for t in (-100, -30, -10, -3, -1, 0, 1, 3, 10, 30, 100) if abs(t) <= top_lim]
        ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks]); ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.axhspan(-1, 1, color="#F3F5F7", lw=0, zorder=0)
    Hh = [Rectangle((0, 0), 1, 1, color=GRAY_L), plt.Line2D([], [], marker="_", color=GRAY_D, ls="none", ms=6, mew=0.9),
          plt.Line2D([], [], marker="o", color=SYN, ls="none", ms=4.5, mec="white", mew=0.5),
          plt.Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor=AR6_C, ls="none", ms=5, mew=1.1)]
    L = ["Range of the other IAM families (AR6), 2050", "Individual other family", "Held-out family, synthetic", "Held-out family, AR6 truth"]
    fig.legend(Hh, L, frameon=False, loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=4, handlelength=1.6, columnspacing=1.4, fontsize=6, handletextpad=0.5)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["2", "3", "4", "5"]
    for w in which:
        {"2": fig2, "3": fig3, "4": fig4, "5": fig5}[w]()
    print("main figures written to", _HERE)
