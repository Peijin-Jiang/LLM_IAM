"""
build_si_data.py
────────────────
Compute the data behind the Supplementary figures into si_figure_data.xlsx:
one sheet per Supplementary figure (FigS1 .. FigS4), holding the data tables of
that figure one below the other (title row, header, rows, then an empty row).
A table holds exactly what the panel draws; make_si_figs.py plots the tables
without further computation. Scoring code and the workbook writer come from
build_main_data.py.

Usage:   python3 build_si_data.py [--sheets figS1,figS2,...]
Output:  si_figure_data.xlsx
         python3 build_si_data.py --statistics writes si_statistics.xlsx, the
         complete statistics behind the figures together with the numbers
         reported in Supplementary Tables S9 and S10 and Supplementary Text S5.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import build_main_data as B

# complete statistics per block; the figure blocks below are reduced from them
SI_STAT_BUILDERS = {
    # ── Supplementary Fig. S1: hyper-parameter selection on the validation set ──
    "figS1_knn_sweep":            ("knn",    lambda c: B.sheet3()),
    "figS1_cvae_sweep":           ("cvae",   lambda c: B.sheet2(c["gt_val"])),
    "figS1_ft9b_lr_val":          ("ft9b",   lambda c: B.sheet4(c["gt_val"])),
    # ── Supplementary Fig. S2: test accuracy of the finetuned model by region ──
    "figS2_region_metrics":       ("region", lambda c: B.sheet_region(c["gt_test"])),
    "figS2_reference_all_test":   ("s2ref",  lambda c: B.sheet7().query("group == 'ALL test'").reset_index(drop=True)),
    # ── Supplementary Fig. S3: scenario coverage (same data as Fig. 4a) ───────
    "figS3_scenario_coverage":    ("s3cov",  lambda c: B.sheet_coverage()),
    # ── Supplementary Fig. S4: 2100 distributions and trajectories of all variables ──
    "figS4_box_quartiles_2100":   ("s4box",  lambda c: B.sheet_box_quartiles(c["gt_test"]).query("year == 2100").reset_index(drop=True)),
    "figS4_envelopes":            ("s4env",  lambda c: B.sheet_envelopes(c["gt_test"])),
    # ── numbers reported in the Supplementary tables and text ────────────────
    "consistency_checks": ("cons",   lambda c: B.sheet_constraints(c["gt_test"])),
    "cost_time":         ("cost",   lambda c: B.sheet6()),
    "neighbour_statistics":      ("nbr",    lambda c: B.sheet1b()),
}

def _s(c, name):
    return B._memo(c, name, SI_STAT_BUILDERS)

def figS1_knn(c):
    d = _s(c, "figS1_knn_sweep").copy()
    d[["k", "weights", "alpha"]] = d.config.str.extract(r"k(\d+)_(\w+)_a([0-9.]+)")
    d["k"] = d.k.astype(int); d["alpha"] = d.alpha.astype(float)
    d["selected"] = d.mae.eq(d.mae.min()) & (d.index == d.index[0])     # rows are sorted by MAE
    return d[["weights", "k", "alpha", "mae", "selected"]]

def figS1_cvae(c):
    d = _s(c, "figS1_cvae_sweep").copy()
    d["learning_rate"] = d.config.str.extract(r"lr([0-9.e-]+)_")[0].astype(float)
    d["hidden_size"] = d.config.str.extract(r"hid(\d+)")[0].astype(int)
    return d[["learning_rate", "hidden_size", "mae"]]

FT_LR = {"9b_lr2e-4": 2e-4, "9b_lr5e-4": 5e-4, "9b_lr1e-3": 1e-3}

def figS1_ft(c):
    d = _s(c, "figS1_ft9b_lr_val").copy(); d["learning_rate"] = d.config.map(FT_LR)
    return d.sort_values("learning_rate")[["learning_rate", "mae", "ci2.5", "ci97.5"]].reset_index(drop=True)

def figS3_coverage(c):
    d = _s(c, "figS3_scenario_coverage")
    return d[d.level.isin(["dimension", "family"])][["level", "dimension", "model_family", "n_scenarios"]].reset_index(drop=True)

SI_BUILDERS = {
    "figS1_knn_sweep":            ("figS1", figS1_knn),
    "figS1_cvae_sweep":           ("figS1", figS1_cvae),
    "figS1_ft9b_lr_val":          ("figS1", figS1_ft),
    "figS2_region_metrics":       ("figS2", lambda c: _s(c, "figS2_region_metrics")[["region", "mae", "smape_%"]]),
    "figS2_reference_all_test":   ("figS2", lambda c: _s(c, "figS2_reference_all_test")[["mae", "smape_%"]]),
    "figS3_scenario_coverage":    ("figS3", figS3_coverage),
    "figS4_box_quartiles_2100":   ("figS4", lambda c: _s(c, "figS4_box_quartiles_2100").drop(columns=["year", "n_cases"])),
    "figS4_envelopes":            ("figS4", lambda c: _s(c, "figS4_envelopes").drop(columns=["median_diff_syn_minus_ar6"])),
}
SI_TITLES = {   # block -> title row of its table in the workbook
    "figS1_knn_sweep":          "Supplementary Fig. S1a and b. Validation MAE of the KNN baseline for each k and α under uniform and similarity weights. The selected setting is marked.",
    "figS1_cvae_sweep":         "Supplementary Fig. S1c. Validation MAE of the cVAE baseline for each learning rate and hidden size.",
    "figS1_ft9b_lr_val":        "Supplementary Fig. S1d. Validation MAE of the finetuned model for each learning rate, with bootstrap 95% confidence intervals.",
    "figS2_region_metrics":     "Supplementary Fig. S2a and b. Test-set MAE and sMAPE of the finetuned model for each R10 region.",
    "figS2_reference_all_test": "Supplementary Fig. S2a and b. Test-set MAE and sMAPE over all cases, drawn as the dashed reference lines.",
    "figS3_scenario_coverage":  "Supplementary Fig. S3a and b. Number of scenarios for each design dimension and for each IAM family.",
    "figS4_box_quartiles_2100": "Supplementary Fig. S4a. Box statistics of the 2100 values of all 22 variables, for AR6 and synthetic trajectories. Values below 0.01 EJ yr⁻¹ were set to 0.01 EJ yr⁻¹.",
    "figS4_envelopes":          "Supplementary Fig. S4b. The 10th, 50th and 90th percentiles across test cases for each variable and year, for AR6 and synthetic trajectories (EJ yr⁻¹).",
}
SI_STAT_LAYOUT = {   # sheet -> [(block, where it is reported, description)]
    "FigS1": [("figS1_knn_sweep", "a, b", ""), ("figS1_cvae_sweep", "c", ""), ("figS1_ft9b_lr_val", "d", "")],
    "FigS2": [("figS2_region_metrics", "a, b", ""), ("figS2_reference_all_test", "a, b", "")],
    "FigS3": [("figS3_scenario_coverage", "a, b", "")],
    "FigS4": [("figS4_box_quartiles_2100", "a", ""), ("figS4_envelopes", "b", "")],
    "Tables": [("consistency_checks", "Table S9", "sectoral consistency checks on the generated trajectories"),
               ("cost_time", "Table S10", "run time and cost per method"),
               ("neighbour_statistics", "Text S5", "neighbour availability in the retrieval step")],
}
SI_XLSX = _HERE / "si_figure_data.xlsx"
SI_STAT_XLSX = _HERE / "si_statistics.xlsx"

if __name__ == "__main__":
    args = B._parse(SI_BUILDERS, "build si_figure_data.xlsx (one sheet per Supplementary figure, only the data drawn)")
    want = set(args.sheets.split(",")) if args.sheets else None
    if args.statistics: B.build_workbook(SI_STAT_XLSX, SI_STAT_BUILDERS, B.statistics_layout(SI_STAT_LAYOUT), want, label="SI statistics")
    else:
        from make_si_figs import SI_BLOCKS
        B.build_workbook(SI_XLSX, SI_BUILDERS, B.figure_layout(SI_BLOCKS, SI_TITLES), want, label="SI")
