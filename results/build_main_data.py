"""
build_main_data.py
──────────────────
Compute the data behind the main-text figures (Figs 2-5) into
main_figure_data.xlsx: one sheet per figure, holding the data tables of that
figure one below the other (a title row that names the panels and describes
the table, a header row, the data rows, then an empty row). A table holds
exactly what a panel draws; make_main_figs.py reads the tables and plots them
without further computation. The AR6 trajectories are not written
to this workbook.

Nothing is re-run here: prediction CSVs, eval workbooks and timing JSONs are
read as they are from the project folder (DATA_ROOT, found automatically; see
_data_root), so re-running this script after the upstream outputs change
updates the workbook. The computation is the scoring of predictions against the
AR6 test truth, the paired case-level bootstrap (B=1000, seed 42: per-trajectory
MAE, resample case ids, same draws for every config in a block) and the
reduction of these statistics to the plotted quantities.

build_si_data.py imports the scoring code and the shared builders from here
and writes the Supplementary workbook.

Usage:   python3 build_main_data.py [--sheets fig2,fig3,...]
Output:  main_figure_data.xlsx
         python3 build_main_data.py --statistics writes main_statistics.xlsx,
         the complete statistics behind the figures and the numbers quoted in
         the text (it contains AR6 trajectories and is not Source Data).
"""


import os
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from scipy import stats            # only the corr=True paths need it
except ImportError:
    stats = None

_HERE = Path(__file__).resolve().parent

def _data_root():
    """The folder that holds the pipeline outputs read below (outputs/,
    make_datasets/output/, policy_coverage_analysis/, finetune/data/, _cache/).
    It is the nearest folder above this script that contains them, so the
    scripts run unchanged wherever this code folder sits inside the project;
    the environment variable IAM_LLM_DATA_ROOT overrides the search."""
    env = os.environ.get("IAM_LLM_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for d in _HERE.parents:
        if (d / "outputs").is_dir() and (d / "make_datasets" / "output" / "split").is_dir():
            return d
    return _HERE.parent          # not found: build_workbook reports it; reading a workbook needs no input data

DATA_ROOT = _data_root()
SPLIT = DATA_ROOT / "make_datasets" / "output" / "split"
OUTS = DATA_ROOT / "outputs"
RAGV = OUTS / "rag_val_reults"
YEARS = [str(y) for y in range(2010, 2101, 10)]
B, SEED = 1000, 42

def load_gt(name):
    gt = pd.read_excel(SPLIT / name)
    gt.columns = [str(c).strip() for c in gt.columns]
    return gt

def traj_table(pred_csv, gt, corr=False):
    """One row per GT trajectory: mae (+rmse/smape/pearson/spearman if corr).
    Missing predicted values are scored as 0.0."""
    pred = pd.read_csv(pred_csv)
    pred.columns = [str(c).strip() for c in pred.columns]
    m = gt[["id", "variable"] + YEARS].merge(
        pred[["id", "variable"] + YEARS], on=["id", "variable"],
        how="left", suffixes=("_t", "_p"), validate="one_to_one")
    yt = m[[f"{y}_t" for y in YEARS]].to_numpy(float)
    yp = m[[f"{y}_p" for y in YEARS]].to_numpy(float)
    valid = np.isfinite(yt)
    ypf = np.where(np.isfinite(yp), yp, 0.0)
    nv = valid.sum(axis=1)
    keep = nv > 0
    diff = np.where(valid, np.abs(yt - ypf), 0.0)
    out = pd.DataFrame({"id": m["id"], "variable": m["variable"],
                        "score": diff.sum(axis=1) / np.maximum(nv, 1)})[keep]
    if not corr:
        return out.reset_index(drop=True)
    rows = []
    for i in np.where(keep)[0]:
        t, p = yt[i][valid[i]], ypf[i][valid[i]]
        denom = (np.abs(t) + np.abs(p)) / 2.0
        sm = float(np.mean(np.abs(t - p) / np.maximum(denom, 1e-12)) * 100.0)
        rm = float(np.sqrt(np.mean((t - p) ** 2)))
        def _corr(fn):
            if t.size < 3 or np.ptp(t) == 0 or np.ptp(p) == 0:
                return np.nan
            with np.errstate(all="ignore"):
                return float(fn(t, p)[0])
        rows.append((sm, rm, _corr(stats.pearsonr), _corr(stats.spearmanr)))
    ext = pd.DataFrame(rows, columns=["smape_%", "rmse", "pearson", "spearman"])
    return pd.concat([out.reset_index(drop=True), ext], axis=1)

def boot_mae(tables):
    """Paired case bootstrap over a dict {label: traj_table}. Same resampled
    ids for every label."""
    ids = sorted(set.intersection(*[set(t["id"]) for t in tables.values()]))
    agg = {}
    for lab, t in tables.items():
        a = t[t["id"].isin(ids)].groupby("id").score.agg(["count", "sum"])
        agg[lab] = (a["count"].reindex(ids).to_numpy(float),
                    a["sum"].reindex(ids).to_numpy(float))
    rng = np.random.default_rng(SEED)
    boot = {lab: np.empty(B) for lab in tables}
    for b in range(B):
        take = rng.integers(0, len(ids), len(ids))
        for lab, (n, s) in agg.items():
            boot[lab][b] = s[take].sum() / n[take].sum()
    return pd.DataFrame([{
        "config": lab,
        "n_cases": len(ids),
        "n_trajectories": int(agg[lab][0].sum()),
        "mae": round(float(t.score.mean()), 4),
        "boot_sd": round(float(boot[lab].std(ddof=1)), 4),
        "ci2.5": round(float(np.percentile(boot[lab], 2.5)), 4),
        "ci97.5": round(float(np.percentile(boot[lab], 97.5)), 4),
    } for lab, t in tables.items()])

TOPK210_RUN = "20260829_0611"
TOPK210_VAL = f"*val10_pred_{TOPK210_RUN}_*topk210_fp-papers*desc-on_rrll15x3*.csv"

def rv(pattern):
    hits = sorted(glob.glob(str(RAGV / pattern)))
    assert hits, f"no match: {pattern}"
    return hits

def sheet1(gt_val):
    """One row per configuration, sorted by MAE.  topk210 is the single
    hard-coded run TOPK210_RUN (the other three repeats are not loaded).
    The k_m sweep (topk200/220/230) is in fig2_km_availability."""
    tables, files = {}, {}
    def one(lab, pattern):
        p = rv(pattern); assert len(p) == 1, (lab, p)
        tables[lab] = traj_table(p[0], gt_val); files[lab] = Path(p[0]).name
    one("norag", "*val10_pred*_norag_*desc-on_sd22.csv")
    one("no_scenario_desc (topk210)", "*pred*topk210_fp-papers*desc-NA*.csv")
    one("no_fingerprint (topk210)", "*pred*topk210_fp-no_fingerprint_desc-on*.csv")
    one("topk210", TOPK210_VAL)
    one("topk010", "*pred*topk010_fp-papers*.csv")
    one("topk110", "*pred*topk110_fp-papers*.csv")
    one("topk310", "*pred*topk310_fp-papers*.csv")
    df = boot_mae(tables)
    df["pred_file"] = df["config"].map(files)
    return df.sort_values("mae").reset_index(drop=True)

def sheet2(gt_val):
    tables = {}
    for p in sorted(glob.glob(str(OUTS / "cvae" / "cond1_*" /
                                  "AR6_R10sd_val10_pred_cvae_sweep_seed42.csv"))):
        lab = re.search(r"(lr[0-9.e-]+_hid\d+)", Path(p).parent.name).group(1)
        tables[lab] = traj_table(p, gt_val)
    assert len(tables) == 12, f"expected 12 cond1 cells, got {len(tables)}"
    return boot_mae(tables).sort_values("mae").reset_index(drop=True)

def sheet3():
    p = glob.glob(str(OUTS / "knn" / "knn_cond1_*" / "bootstrap_val.xlsx"))[0]
    df = pd.read_excel(p)
    keep = ["config", "n_cases", "n_trajectories",
            "mae", "mae_boot_sd", "mae_ci2.5", "mae_ci97.5"]
    return (df[keep].rename(columns={"mae_boot_sd": "boot_sd",
                                     "mae_ci2.5": "ci2.5", "mae_ci97.5": "ci97.5"})
            .sort_values("mae").round(4).reset_index(drop=True))

def sheet4(gt_val):
    fp = OUTS / "finetune_preds"
    tables = {
        "9b_lr2e-4": traj_table(fp / "ft_val_infer_qwen3.5-9b_lr0.0002_r16_bf16_best_vllm_20260830_2017.csv", gt_val),
        "9b_lr5e-4": traj_table(fp / "ft_val_infer_qwen3.5-9b_lr0.0005_r16_bf16_best_vllm_20260830_2017.csv", gt_val),
        "9b_lr1e-3": traj_table(fp / "ft_val_infer_qwen3.5-9b_lr0.001_r16_bf16_best_vllm_20260830_2017.csv", gt_val),
    }
    return boot_mae(tables)

TEST_PREDS = {
    "KNN (cond1 k2 uniform a0.75)":
        "knn/knn_cond1_k2_uniform_a0.75_fp-fingerprint_20260831_0226/AR6_R10sd_test_pred_knn.csv",
    "cVAE (cond1 lr1e-4 hid2048)":
        "cvae/cond1_lr0.0001_hid2048_lat32_beta1.0_fp-fingerprint_20260830_0309/AR6_R10sd_test_pred_cvae_seed42.csv",
    "RAG (topk210, gemini-3.5-flash-lite)":
        "AR6_R10sd_test_pred_20260831_0241_gemini-3.5-flash-lite_topk210_fp-papers-comp_evidence_gpt-5.6_20260828_0525_struct_desc-on_rrll15x3_sd22.csv",
    "Finetune Qwen3.5-9B (lr5e-4)":
        "finetune_preds/ft_test_qwen3.5-9b_lr0.0005_r16_bf16_best_vllm_20260830_2009.csv",
}

def sheet5(gt_test):
    tables = {lab: traj_table(OUTS / rel, gt_test, corr=True)
              for lab, rel in TEST_PREDS.items()}
    metrics = {"mae": "score", "smape_%": "smape_%",
               "pearson": "pearson", "spearman": "spearman"}
    ids = sorted(set.intersection(*[set(t["id"]) for t in tables.values()]))
    # per-id (count, sum) per metric; corr rows with NaN are excluded from both
    agg = {}
    for lab, t in tables.items():
        k = t[t["id"].isin(ids)]
        agg[lab] = {}
        for m, col in metrics.items():
            g = k[["id", col]].dropna().groupby("id")[col].agg(["count", "sum"])
            agg[lab][m] = (g["count"].reindex(ids).fillna(0).to_numpy(float),
                           g["sum"].reindex(ids).fillna(0).to_numpy(float))
    rng = np.random.default_rng(SEED)
    boot = {lab: {m: np.empty(B) for m in metrics} for lab in tables}
    for b in range(B):
        take = rng.integers(0, len(ids), len(ids))   # same draws for all methods
        for lab in tables:
            for m in metrics:
                n, s = agg[lab][m]
                boot[lab][m][b] = s[take].sum() / n[take].sum()
    rows = []
    for lab, t in tables.items():
        r = {"method": lab, "n_cases": t["id"].nunique(), "n_trajectories": len(t)}
        for m, col in metrics.items():
            vals = t["score"] if col == "score" else t[col]
            r[m] = round(float(vals.mean()), 4)
            r[f"{m}_ci2.5"] = round(float(np.percentile(boot[lab][m], 2.5)), 4)
            r[f"{m}_ci97.5"] = round(float(np.percentile(boot[lab][m], 97.5)), 4)
        r["n_valid_pearson"] = int(t.pearson.notna().sum())
        r["n_valid_spearman"] = int(t.spearman.notna().sum())
        rows.append(r)
    return pd.DataFrame(rows)

def sheet6():
    rag_eval = glob.glob(str(OUTS / "AR6_R10sd_test_eval_20260831_0241_*topk210*.xlsx"))[0]
    rc = pd.read_excel(rag_eval, sheet_name="run_config").set_index("key")["value"]
    knn = json.load(open(glob.glob(str(OUTS / "knn" / "knn_cond1_*" /
                                       "test_metrics.json"))[0]))["timing"]
    cva = json.load(open(glob.glob(str(OUTS / "cvae" / "cond1_lr0.0001_hid2048_*" /
                                       "TEST_TIMING.json"))[0]))
    ret = json.load(open(DATA_ROOT / "finetune" / "data" /
                         "RETRIEVAL_TIMING_test_nall_c100.json"))
    ri = json.load(open(OUTS / "finetune_preds" /
                        "ft_test_qwen3.5-9b_lr0.0005_r16_bf16_best_vllm_20260830_2009_run_info.json"))
    ts = ri["timing_sec"]
    def row(method, component, cost, cost_case, wall, per_case, lat, note=""):
        return {"method": method, "component": component,
                "total_cost_usd": cost, "cost_per_case_usd": cost_case,
                "test_wallclock_sec": wall, "sec_per_case": per_case,
                "per_case_latency_sec": lat, "note": note}
    rows = [
        row("KNN", "test inference", None, None, knn["rpt_total_wallclock_sec"],
            knn["rpt_sec_per_case_batch"], knn["rpt_sec_per_trajectory"],
            "local compute, no API cost; batched -> latency = batch per-case"),
        row("cVAE", "training (winner, timed retrain)", None, None,
            cva["rpt_train_wallclock_sec"], None, None, f"device {cva['device']}"),
        row("cVAE", "test inference", None, None, cva["rpt_total_wallclock_sec"],
            cva["rpt_sec_per_case_batch"], cva["rpt_sec_per_trajectory"],
            "one batched forward pass"),
        row("RAG", "test inference (retrieval+rerank+generation)",
            float(rc["rpt_total_cost_usd"]), float(rc["rpt_cost_per_case_usd"]),
            float(rc["rpt_total_wallclock_sec"]), float(rc["rpt_sec_per_case_batch"]),
            float(rc["rpt_sec_per_trajectory"]),
            "gemini-3.5-flash-lite, 60 workers; latency = p50 under load"),
        row("Finetune 9B", "retrieval + prompt build (topk210)",
            ret["rpt_total_cost_usd"], ret["rpt_cost_per_case_usd"],
            ret["rpt_total_wallclock_sec"], ret["rpt_sec_per_case_batch"],
            ret["rpt_sec_per_trajectory"], "100 workers; rerank LLM calls"),
        row("Finetune 9B", "vLLM generation", None, None,
            ts["generate"], ts["generate_per_case"], None,
            f"A100-80GB; engine build {ts['engine_build']}s excluded; "
            "per-request latency not populated by this vLLM version"),
        row("Finetune 9B", "COMBINED (retrieval + generation)",
            ret["rpt_total_cost_usd"], ret["rpt_cost_per_case_usd"],
            round(ret["rpt_total_wallclock_sec"] + ts["generate"], 1),
            round(ret["rpt_sec_per_case_batch"] + ts["generate_per_case"], 3),
            None, "GPU training/inference cost is hardware time, not API dollars"),
    ]
    return pd.DataFrame(rows)

def sheet7():
    return pd.read_excel(OUTS / "sparse_analysis_9b_test.xlsx", sheet_name="summary")


def sheet1b():
    return pd.read_excel(OUTS / "neighbor_stats_val.xlsx", sheet_name="summary")


def sheet_coverage():
    """Scenario-distribution counts for Fig. 4a and Fig. S3.

    Policy-coverage conventions:
    distinct scenarios; a scenario counts in every dimension it states
    (none stated -> 'None of the four stated'); family counts = distinct
    scenarios the family ran (train70+val10+test combined)."""
    DIM_OF_KEY = {
        "stated_quantified_mitigation_constraint": "Mitigation constraint",
        "stated_non_ssp2_future_socioeconomic_driver": "Alternative socioeconomic pathway",
        "stated_supply_side_technology_setting": "New technology assumption",
        "stated_demand_side_technology_setting": "New technology assumption",
        "stated_side_unspecified_technology_setting": "New technology assumption",
        "stated_new_climate_policy": "New climate policy",
    }
    OTHER = "None of the four stated"
    lab = pd.read_csv(DATA_ROOT / "policy_coverage_analysis" / "coverage" / "output" /
                      "scenario_labels.csv")
    sdims = {}
    for _, r in lab.iterrows():
        d = {DIM_OF_KEY[k] for k in DIM_OF_KEY if bool(r[k])}
        sdims[str(r["scenario"]).strip()] = d if d else {OTHER}
    gt = pd.concat([pd.read_excel(DATA_ROOT / "make_datasets" / "output" / "split" / f)
                    for f in ("AR6_R10sd_train70_ground_truth.xlsx",
                              "AR6_R10sd_val10_gt.xlsx", "AR6_R10sd_test_gt.xlsx")],
                   ignore_index=True)
    pairs = (gt[["scenario", "model_family"]].astype(str).drop_duplicates())
    rows = []
    from collections import Counter
    dim_n = Counter()
    for s, d in sdims.items():
        for dd in d:
            dim_n[dd] += 1
    for dd in sorted(dim_n):          # sorted: set iteration order is not stable
        rows.append({"level": "dimension", "dimension": dd, "model_family": None,
                     "n_scenarios": dim_n[dd]})
    for f, n in pairs.groupby("model_family").scenario.nunique().items():
        rows.append({"level": "family", "dimension": None, "model_family": f,
                     "n_scenarios": int(n)})
    flagged = pd.read_excel(
        DATA_ROOT / "policy_coverage_analysis" / "coverage" / "report" /
        "policy_coverage.xlsx", sheet_name="Gaps", header=3)
    dxf_status = {
        (str(r["Class"]), str(r["Model family"])): str(r["Status"])
        for _, r in flagged[flagged["Sheet"].eq("Dimension x family")].iterrows()
    }
    cell = Counter()
    for _, r in pairs.iterrows():
        for dd in sdims.get(r.scenario, {OTHER}):
            cell[(dd, r.model_family)] += 1
    dims = sorted(dim_n)
    fams = sorted(pairs.model_family.unique())
    for dd in dims:
        for f in fams:
            n = cell.get((dd, f), 0)
            rows.append({"level": "dimension x family", "dimension": dd,
                         "model_family": f, "n_scenarios": n,
                         "coverage_status": dxf_status.get(
                             (dd, f), "Higher coverage")})
    return pd.DataFrame(rows)


def sheet_region(gt_test):
    """Per-region test metrics for the finetuned model (Supplementary Fig. S2),
    with the traj_table conventions.
    """
    tt = traj_table(OUTS / TEST_PREDS["Finetune Qwen3.5-9B (lr5e-4)"],
                    gt_test, corr=True)
    reg = gt_test.drop_duplicates("id").set_index("id")["region"]
    tt["region"] = tt["id"].map(reg)
    agg = (tt.groupby("region")
             .agg(n_cases=("id", "nunique"), n_trajectories=("score", "size"),
                  mae=("score", "mean"), smape=("smape_%", "mean"),
                  pearson=("pearson", "mean"), spearman=("spearman", "mean"))
             .rename(columns={"smape": "smape_%"})
             .sort_values("mae").reset_index())
    return agg.round(4)


# ── Supplementary Table S9: structural checks on the generated trajectories ──
# The prompt asks for three sectoral inequalities, but nothing in the pipeline
# enforces them (only non-negativity is projected, and rounding is applied on
# top). This table reports whether the outputs satisfy them anyway.
SECTOR_PAIRS = [("Final Energy|Industry|Electricity",
                 "Final Energy|Industry"),
                ("Final Energy|Transportation|Electricity",
                 "Final Energy|Transportation"),
                ("Final Energy|Residential and Commercial|Electricity",
                 "Final Energy|Residential and Commercial")]

def _structural_stats(df):
    """Sectoral-consistency violations and negative values for one wide
    (id, variable, years) table (Supplementary Table S9).
    Compare electricity directly with its sector total; no numerical tolerance."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    n_chk = n_vio = 0
    mags = []
    for child, parent in SECTOR_PAIRS:
        c = df[df.variable.eq(child)].set_index("id")[YEARS].astype(float)
        pa = df[df.variable.eq(parent)].set_index("id")[YEARS].astype(float)
        ids = c.index.intersection(pa.index)
        cv, pv = c.loc[ids].to_numpy(), pa.loc[ids].to_numpy()
        ok = np.isfinite(cv) & np.isfinite(pv)
        d = cv - pv
        n_chk += int(ok.sum())
        v = ok & (cv > pv)
        n_vio += int(v.sum())
        mags += list(d[v])
    sector_vars = {x for pr in SECTOR_PAIRS for x in pr}
    neg = int(sum((df[df.variable.eq(v)][YEARS].astype(float).to_numpy() < 0).sum()
                  for v in sector_vars))
    m = np.array(mags) if mags else np.array([0.0])
    return {"n_checks": n_chk, "n_violations": n_vio,
            "violation_rate_%": round(100 * n_vio / n_chk, 4),
            "median_violation_EJ_yr": round(float(np.median(m)), 4) if mags else 0.0,
            "max_violation_EJ_yr": round(float(m.max()), 4) if mags else 0.0,
            "n_negative_values": neg}


def sheet_constraints(gt_test):
    rows = [{"source": "AR6 ground truth", **_structural_stats(gt_test)}]
    for lab, rel in TEST_PREDS.items():
        short = {"KNN (cond1 k2 uniform a0.75)": "KNN",
                 "cVAE (cond1 lr1e-4 hid2048)": "cVAE",
                 "RAG (topk210, gemini-3.5-flash-lite)": "RAG (topk210)",
                 "Finetune Qwen3.5-9B (lr5e-4)": "Qwen3.5-9B (finetune)"}[lab]
        rows.append({"source": short,
                     **_structural_stats(pd.read_csv(OUTS / rel))})
    return pd.DataFrame(rows)


# ── reference error scales that the model is compared against ──────────────────
def sheet_error_scales(gt_test):
    """One row per reference quantity, on the project's per-trajectory
    convention, for all four metrics. Rows are NOT on a common subset (the
    inter-IAM ones only exist where another IAM ran the same scenario+region),
    so n is reported per row."""
    tt = traj_table(OUTS / TEST_PREDS["Finetune Qwen3.5-9B (lr5e-4)"],
                    gt_test, corr=True)
    MET = ["mae", "smape_%", "pearson", "spearman"]

    pool = pd.concat([pd.read_excel(SPLIT / f) for f in
                      ("AR6_R10sd_train70_ground_truth.xlsx",
                       "AR6_R10sd_val10_gt.xlsx", "AR6_R10sd_test_gt.xlsx")],
                     ignore_index=True)
    pool.columns = [str(c).strip() for c in pool.columns]
    key = ["region", "scenario", "variable"]
    m = (gt_test[["id"] + key + ["model_family"] + YEARS]
         .merge(pool[key + ["model_family"] + YEARS], on=key,
                suffixes=("_t", "_o")))
    m = m[m.model_family_t != m.model_family_o]
    a = m[[f"{y}_t" for y in YEARS]].to_numpy(float)
    b = m[[f"{y}_o" for y in YEARS]].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    nv = ok.sum(axis=1)
    sel = nv > 0
    a, b, ok, nv, m = a[sel], b[sel], ok[sel], nv[sel], m[sel].copy()
    m["mae"] = np.where(ok, np.abs(a - b), 0.0).sum(axis=1) / np.maximum(nv, 1)
    den = (np.abs(a) + np.abs(b)) / 2.0
    m["smape_%"] = (np.where(ok, np.abs(a - b) / np.maximum(den, 1e-12), 0.0)
                    .sum(axis=1) / np.maximum(nv, 1) * 100.0)
    pe_, sp_ = [], []
    for i in range(len(a)):
        t, q = a[i][ok[i]], b[i][ok[i]]
        if t.size < 3 or np.ptp(t) == 0 or np.ptp(q) == 0:
            pe_.append(np.nan); sp_.append(np.nan); continue
        with np.errstate(all="ignore"):
            pe_.append(float(stats.pearsonr(t, q)[0]))
            sp_.append(float(stats.spearmanr(t, q)[0]))
    m["pearson"], m["spearman"] = pe_, sp_
    per = m.groupby(["id", "variable"])[MET].mean()
    per_min_mae = m.groupby(["id", "variable"])["mae"].min()
    matched = tt.merge(per.reset_index()[["id", "variable"]],
                       on=["id", "variable"], how="inner")

    xls = OUTS / "scenario_embedding_pairs.xlsx"
    summ = pd.read_excel(xls, sheet_name="summary")
    pooled = summ[summ.aggregation.str.startswith("pooled")].iloc[0]
    p9 = (pd.read_excel(xls, sheet_name="pairs")
            .sort_values("cosine_sim", ascending=False).iloc[0])

    def row(q, vals, n, note):
        d = {"quantity": q, "n_trajectories": int(n), "note": note}
        d.update({k: (None if v is None else round(float(v), 4))
                  for k, v in zip(MET, vals)})
        return d

    def mv(df, col="score"):
        return [df[col].mean(), df["smape_%"].mean(),
                df.pearson.mean(), df.spearman.mean()]

    rows = [
        row("Qwen3.5-9B, matched subset", mv(matched), len(matched),
            "test trajectories that also have >=1 other-IAM counterpart"),
        row("Most-similar scenario pair",
            [p9.mae, p9["smape_%"], p9.pearson, p9.spearman],
            p9.n_trajectories,
            f"single closest pair: {p9.scenario} vs {p9.nearest_by_embedding}"),
        row("Qwen3.5-9B, all test cases", mv(tt), len(tt), "headline test MAE"),
        row("Nearest scenario, same IAM",
            [pooled.mae, pooled["smape_%"], pooled.pearson, pooled.spearman],
            pooled.n,
            "substitute the description-nearest other scenario, model+region fixed"),
        row("Closest other IAM", [per_min_mae.mean(), None, None, None],
            len(per_min_mae),
            "same scenario+region, the single most similar other model family"),
        row("Average other IAM", [per[c].mean() for c in MET], len(per),
            "same scenario+region, averaged over every other model family"),
    ]
    return pd.DataFrame(rows)[["quantity"] + MET + ["n_trajectories", "note"]]


# ── Fig. 4c-f dotted line: description-nearest scenario, same family+region ──
def _traj_metrics_np(a, b, ok):
    """Per-row MAE / sMAPE / Pearson / Spearman of b scored against a on the
    entries flagged ok, with numpy only. Same rules as traj_table: NaN
    correlation when <3 points or either series is flat; Spearman = Pearson of
    average ranks (= scipy.stats.spearmanr)."""
    nv = ok.sum(axis=1)
    mae = np.where(ok, np.abs(a - b), 0.0).sum(axis=1) / np.maximum(nv, 1)
    den = (np.abs(a) + np.abs(b)) / 2.0
    with np.errstate(all="ignore"):
        sm = (np.where(ok, np.abs(a - b) / np.maximum(den, 1e-12), 0.0)
              .sum(axis=1) / np.maximum(nv, 1) * 100.0)

    def corr(rank):
        out = np.full(len(a), np.nan)
        for i in range(len(a)):
            t, q = a[i][ok[i]], b[i][ok[i]]
            if t.size < 3 or np.ptp(t) == 0 or np.ptp(q) == 0:
                continue
            if rank:
                t = pd.Series(t).rank().to_numpy()
                q = pd.Series(q).rank().to_numpy()
            out[i] = np.corrcoef(t, q)[0, 1]
        return out
    return mae, sm, corr(False), corr(True)


def _scenario_desc_embeddings():
    """scenario -> unit-norm description embedding, built exactly the way
    rag_core.py builds the same-family retrieval (Methods 4.4): description =
    first non-empty train70 text per scenario, supplemented from the mixed
    dataset for scenarios absent from train70; vectors from
    _cache/cache_scenario_desc_embeddings.json (text-embedding-3-large, key
    'SCEN_DESC:<text>'). No API call - refuses if a text is not cached."""
    tr = load_gt("AR6_R10sd_train70_ground_truth.xlsx")
    d = tr[["scenario", "scenario_description"]].copy()
    d["scenario"] = d.scenario.astype(str).str.strip()
    d["scenario_description"] = d.scenario_description.fillna("").astype(str)
    d = (d.sort_values("scenario").groupby("scenario", as_index=False)
          .agg({"scenario_description":
                lambda x: next((t for t in x if str(t).strip()), "")}))
    s2d = dict(zip(d.scenario, d.scenario_description))
    mixed = sorted((DATA_ROOT / "make_datasets" / "output").glob("AR6_R10_sd_mixed_*.xlsx"))
    xl = pd.ExcelFile(mixed[-1])
    sn = next(s for s in xl.sheet_names
              if "scenario" in [str(c).strip() for c in
                                pd.read_excel(xl, sheet_name=s, nrows=1).columns])
    md = pd.read_excel(xl, sheet_name=sn)
    md.columns = [str(c).strip() for c in md.columns]
    sup = {}
    for s, t in zip(md.scenario, md.scenario_description):
        s, t = str(s).strip(), "" if pd.isna(t) else str(t).strip()
        if s and t:
            sup.setdefault(s, t)
    for s, t in sup.items():
        if not str(s2d.get(s, "")).strip():
            s2d[s] = t
    cache = json.load(open(DATA_ROOT / "_cache" / "cache_scenario_desc_embeddings.json"))
    assert cache.get("__embed_model__") == "text-embedding-3-large"
    out = {}
    for s, t in s2d.items():
        k = f"SCEN_DESC:{str(t).strip()}"
        assert k in cache, f"description of {s!r} not in embedding cache"
        v = np.asarray(cache[k], float)
        out[s] = v / np.linalg.norm(v)
    return out


def sheet_nearest_desc_scenario(gt_test):
    """Fig. 4c-f dotted line. For every test case (family x scenario x region) take
    the other scenario whose description embedding is most similar among the
    scenarios the SAME IAM family ran in the SAME region (pool = all AR6 GT:
    train70+val10+test), score its trajectories as a 'prediction' of the test
    case with the per-trajectory convention of the model, and average over
    trajectories - i.e. how far the target is from its most similar sibling
    run. Rows: the five Fig. 4c-f groups, with Qwen3.5-9B on the identical
    trajectories, plus a train70-only pool (= the retrieval pool) as a
    sensitivity check. MAE CIs: case bootstrap B=1000 seed 42, paired for the
    difference."""
    MET = ["mae", "smape_%", "pearson", "spearman"]
    emb = _scenario_desc_embeddings()
    parts = []
    for f, split in (("AR6_R10sd_train70_ground_truth.xlsx", "train"),
                     ("AR6_R10sd_val10_gt.xlsx", "val"),
                     ("AR6_R10sd_test_gt.xlsx", "test")):
        g = load_gt(f)[["region", "model_family", "scenario", "variable"] + YEARS]
        g["split"] = split
        parts.append(g)
    pool = pd.concat(parts, ignore_index=True)
    pool["scenario"] = pool.scenario.astype(str).str.strip()
    pool = pool.drop_duplicates(["region", "model_family", "scenario", "variable"])
    P = pool.set_index(["region", "model_family", "scenario", "variable"])[YEARS]
    runs = pool.drop_duplicates(["region", "model_family", "scenario"])
    split_of = dict(zip(zip(runs.region, runs.model_family, runs.scenario), runs.split))

    t = gt_test.copy()
    t["scenario"] = t.scenario.astype(str).str.strip()
    cases = t.drop_duplicates("id")[["id", "region", "model_family", "scenario"]]

    # Qwen on the same trajectories (traj_table rules: missing pred -> 0.0)
    pred = _qwen_pred()
    mq = t[["id", "variable"] + YEARS].merge(
        pred[["id", "variable"] + YEARS], on=["id", "variable"], how="left",
        suffixes=("_t", "_p"), validate="one_to_one")
    yt = mq[[f"{y}_t" for y in YEARS]].to_numpy(float)
    yp = mq[[f"{y}_p" for y in YEARS]].to_numpy(float)
    okq = np.isfinite(yt)
    qw = pd.DataFrame(dict(zip(MET, _traj_metrics_np(
        yt, np.where(np.isfinite(yp), yp, 0.0), okq))))
    qw["id"], qw["variable"] = mq["id"].values, mq["variable"].values
    qw = qw[okq.sum(axis=1) > 0]

    pc = pd.read_excel(OUTS / "sparse_analysis_9b_test.xlsx", sheet_name="per_case")
    pc = pc.set_index("id")
    groups = [
        ("ALL test", lambda ids: np.ones(len(ids), bool)),
        ("Lower coverage: dimension x IAM family",
         lambda ids: pc.loc[ids, "lower_coverage_dim_x_family"].to_numpy(bool)),
        ("Higher coverage: dimension x IAM family",
         lambda ids: ~pc.loc[ids, "lower_coverage_dim_x_family"].to_numpy(bool)),
        ("not scenario seen in train (other family)",
         lambda ids: ~pc.loc[ids, "scenario_seen_in_train"].to_numpy(bool)),
        ("scenario seen in train (other family)",
         lambda ids: pc.loc[ids, "scenario_seen_in_train"].to_numpy(bool)),
    ]

    rows = []
    for pool_name, sel in (("all AR6 runs", None), ("train70 only", "train")):
        R = runs if sel is None else runs[runs.split == sel]
        sib = R.groupby(["region", "model_family"]).scenario.apply(list).to_dict()
        nn, cs = {}, {}
        for c in cases.itertuples():
            cand = [s for s in sib.get((c.region, c.model_family), [])
                    if s != c.scenario]
            if not cand:
                continue
            sims = np.array([emb[c.scenario] @ emb[s] for s in cand])
            j = int(np.argmax(sims))
            nn[c.id], cs[c.id] = cand[j], float(sims[j])
        k = t[t.id.isin(nn)].copy()
        k["nn"] = k.id.map(nn)
        a = k[YEARS].to_numpy(float)
        b = P.reindex(list(zip(k.region, k.model_family, k.nn, k.variable))).to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        has = ok.sum(axis=1) > 0
        ref = pd.DataFrame(dict(zip(MET, _traj_metrics_np(a[has], b[has], ok[has]))))
        ref["id"], ref["variable"] = k.id.values[has], k.variable.values[has]
        both = ref.merge(qw, on=["id", "variable"], suffixes=("_ref", "_qwen"))

        for gname, gfun in groups:
            ids_all = np.array(sorted(both.id.unique()))
            gid = ids_all[gfun(ids_all)]
            sub = both[both.id.isin(gid)]
            # case bootstrap on trajectory-weighted MAE (paired)
            agg = sub.groupby("id").agg(n=("mae_ref", "size"),
                                        r=("mae_ref", "sum"), q=("mae_qwen", "sum"))
            n_, r_, q_ = (agg[c].to_numpy(float) for c in ("n", "r", "q"))
            rng = np.random.default_rng(SEED)
            br, bd = np.empty(B), np.empty(B)
            for bi in range(B):
                tk = rng.integers(0, len(n_), len(n_))
                br[bi] = r_[tk].sum() / n_[tk].sum()
                bd[bi] = (q_[tk].sum() - r_[tk].sum()) / n_[tk].sum()
            d = {"pool": pool_name, "group": gname,
                 "n_cases": int(sub.id.nunique()), "n_trajectories": len(sub),
                 "mean_cosine": round(float(np.mean([cs[i] for i in gid])), 4)}
            for m in MET:
                d[m] = round(float(sub[f"{m}_ref"].mean()), 4)
            d["mae_ci2.5"] = round(float(np.percentile(br, 2.5)), 4)
            d["mae_ci97.5"] = round(float(np.percentile(br, 97.5)), 4)
            for m in MET:
                d[f"qwen_{m}"] = round(float(sub[f"{m}_qwen"].mean()), 4)
            d["qwen_minus_ref_mae"] = round(float(sub.mae_qwen.mean() - sub.mae_ref.mean()), 4)
            d["diff_ci2.5"] = round(float(np.percentile(bd, 2.5)), 4)
            d["diff_ci97.5"] = round(float(np.percentile(bd, 97.5)), 4)
            d["share_traj_qwen_below_ref_%"] = round(
                100 * float((sub.mae_qwen < sub.mae_ref).mean()), 2)
            src = [split_of[(c.region, c.model_family, nn[c.id])]
                   for c in cases[cases.id.isin(gid)].itertuples()]
            d["nn_from_train/val/test"] = "/".join(
                str(src.count(s)) for s in ("train", "val", "test"))
            rows.append(d)
    return pd.DataFrame(rows)


# ── Fig. 4g-j: one-assumption pairs (test case A vs retrieved training run B) ─
# For every test case A, its counterparts B are TRAINING runs of the same IAM
# family in the same region whose scenario differs from A's in exactly ONE
# design assumption, everything else equal. Assumptions are read from each
# study's naming convention (checked against the scenario descriptions):
# ENGAGE EN_<near-term policy>_<budget>[f]_<tags>; IMAGE SSPx_SPAx_<forcing>I_
# <variant>[_LB]; MESSAGEix SSP2_<grid>_<cost>_<tax|CB>; GCAM R_MAC_<net-zero
# year>_n<0|8>; REMIND R2p1_<SSP>-<policy>, CEMICS_*, TechCost-*, Transport_*,
# BEG-*; WITCH DISCRATE_*; COMMIT CO_*; SSPx-baseline. Studies whose variants
# differ in several assumptions at once (NGFS, the COMMIT policy packages)
# are not paired. For an ordered assumption (budget, forcing, carbon tax,
# net-zero year, discount rate) only the nearest TRAINING level below and
# above A is kept, so each pair is the closest training run in that
# direction. Question: does yhat_A move away from y_B the way y_A does?
#   d = y_A - y_B (AR6),  dh = yhat_A - y_B (synthetic).
CONTRAST_YEARS = YEARS[1:]          # 2020-2100; 2010 is common history
ASSUMPTION_ORDER = [
    "Carbon budget vs reference policy",
    "Mitigation stringency, nearest level",
    "Net-negative CO2 allowed vs not",
    "Current policies vs NDCs",
    "No policy vs current policies",
    "Delayed action",
    "Carbon-price path",
    "Socioeconomic pathway (SSP)",
    "Technology or demand option",
    "Technology cost",
    "Discount rate",
    "COVID-19 variant",
    "Updated NDCs",
    "Limited BECCS or bioenergy",
]
_LEVEL_KEYS = {"ENGAGE": "budget", "IMAGE SPA": "lvl", "MESSAGEix grid": "lvl",
               "GCAM R_MAC": "lvl", "REMIND R2p1": "lvl", "REMIND CEMICS": "lvl",
               "WITCH DISCRATE": "lvl"}


def _scenario_design(s):
    """(study, {assumption: value}) parsed from the scenario name, or None."""
    s = str(s).strip()
    if s.startswith("EN_"):
        tok = s[3:].split("_")
        near = {"NPi2020": "NPi", "NPi2100": "NPi", "INDCi2030": "NDC",
                "INDCi2100": "NDC", "NoPolicy": "none"}.get(tok[0])
        if near is None:
            return None
        bud, f, tags = None, None, []
        for t in tok[1:]:
            m = re.fullmatch(r"(\d+)(f?)", t)
            if m and bud is None:
                bud, f = int(m.group(1)), bool(m.group(2))
            else:
                tags.append(t)
        dr = [t for t in tags if t.startswith("DR")]
        return "ENGAGE", {"near": near, "long": "budget" if bud else "reference",
                          "budget": bud, "f": f, "COV": "COV" in tags,
                          "NDCp": "NDCp" in tags, "lowBECCS": "lowBECCS" in tags,
                          "DR": dr[0] if dr else "default"}
    pats = [
        ("IMAGE SPA", r"SSP(\d)_SPA(\d)_(\d\d)I_([A-Z]+)(_LB)?",
         lambda m: {"ssp": "SSP" + m[1], "spa": "SPA" + m[2],
                    "lvl": int(m[3]), "variant": m[4], "LB": bool(m[5])}),
        ("MESSAGEix grid", r"SSP2_(int|noint|openres)_(lc|mc)_(CB)?(\d+)",
         lambda m: {"grid": m[1], "cost": m[2], "instrument": "CB" if m[3] else "tax",
                    "lvl": int(m[4])}),
        ("GCAM R_MAC", r"R_MAC_(\d+)_n(\d)",
         lambda m: {"lvl": int(m[1]), "neg": m[2]}),
        ("REMIND R2p1", r"R2p1_(SSP\d)-(Base|NPi|NDC|PkBudg(\d+))",
         lambda m: {"ssp": m[1], "pol": "Budg" if m[3] else m[2],
                    "lvl": int(m[3]) if m[3] else None}),
        ("REMIND CEMICS", r"CEMICS_(GDPgrowth|HotellingConst|Linear|opt)_(1p5|2C)",
         lambda m: {"path": m[1], "lvl": {"1p5": 15, "2C": 20}[m[2]]}),
        ("REMIND CEMICS CDR", r"CEMICS_(SSP\d)-(1p5C|2C)-(fullCDR|minCDR)",
         lambda m: {"ssp": m[1], "target": m[2], "cdr": m[3]}),
        ("REMIND CEMICS NPi", r"CEMICS_(SSP\d)-Npi", lambda m: {"ssp": m[1]}),
        ("REMIND TechCost", r"TechCost-SSP2-B1100-(\w+)H", lambda m: {"tech": m[1]}),
        ("REMIND Transport", r"Transport_(NPi|NDC|Budg1100)_(Conv|ConvSyn|ElecPush|H2Push)(-LowD)?",
         lambda m: {"pol": m[1], "tech": m[2], "lowD": bool(m[3])}),
        ("WITCH DISCRATE", r"DISCRATE_Ref_dr(\d+)p", lambda m: {"lvl": int(m[1])}),
        ("REMIND BEG", r"BEG-(Base|Budg600|EG|Budg600-EG)",
         lambda m: {"pol": "Budg600" if "Budg600" in m[1] else "Base",
                    "EG": m[1].endswith("EG")}),
        ("COMMIT timing", r"CO_2Deg(2020|2030)", lambda m: {"start": m[1]}),
        ("COMMIT policy", r"CO_(BAU|CurPol|NDCplus|GPP|Bridge)", lambda m: {"pol": m[1]}),
        ("SSP baseline", r"(SSP\d)-baseline", lambda m: {"ssp": m[1]}),
    ]
    for study, pat, fn in pats:
        m = re.fullmatch(pat, s)
        if m:
            return study, fn(m)
    return None


def _assumption_type(study, diff, a, b):
    """Name of the single assumption in which a and b differ, else None."""
    if study == "ENGAGE" and "long" in diff and set(diff) <= {"long", "budget", "f"}:
        return "Carbon budget vs reference policy"
    if len(diff) != 1:
        return None
    k = diff[0]
    va, vb = a[k], b[k]
    if k == _LEVEL_KEYS.get(study):
        if study == "WITCH DISCRATE":
            return "Discount rate"
        return "Mitigation stringency, nearest level"
    if k in ("near", "pol"):
        pair = {va, vb}
        if pair == {"NPi", "NDC"}:
            return "Current policies vs NDCs"
        if pair in ({"none", "NPi"}, {"Base", "NPi"}, {"BAU", "CurPol"}):
            return "No policy vs current policies"
        if study == "REMIND BEG" and pair == {"Base", "Budg600"}:
            return "Carbon budget vs reference policy"
        return None
    return {"f": "Net-negative CO2 allowed vs not", "neg": "Net-negative CO2 allowed vs not",
            "start": "Delayed action", "path": "Carbon-price path",
            "ssp": "Socioeconomic pathway (SSP)",
            "variant": "Technology or demand option", "grid": "Technology or demand option",
            "tech": ("Technology cost" if study == "REMIND TechCost"
                     else "Technology or demand option"),
            "lowD": "Technology or demand option", "EG": "Technology or demand option",
            "cost": "Technology cost", "DR": "Discount rate",
            "COV": "COVID-19 variant", "NDCp": "Updated NDCs",
            "lowBECCS": "Limited BECCS or bioenergy", "LB": "Limited BECCS or bioenergy",
            "cdr": "Limited BECCS or bioenergy"}.get(k)


def _test_same_family_neighbours():
    """test id -> set of same-family neighbour scenarios in its SFT prompt
    (finetune/data/sft_test.jsonl, the prompts behind the Qwen test CSV)."""
    pat = re.compile(r'"neighbor_type":\s*"SAME-MODEL-FAMILY[^"]*".*?'
                     r'"scenario":\s*"([^"]+)"', re.S)
    out = {}
    with open(DATA_ROOT / "finetune" / "data" / "sft_test.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            u = next(m["content"] for m in d["messages"] if m["role"] == "user")
            out[d["id"]] = set(pat.findall(u))
    return out


_PAIR_CACHE = {}


def _contrast_pairs(gt_test):
    """One row per (test case A, training run B) pair that differs in exactly
    one assumption (same family, same region, same study). Returns the pairs
    and the AR6 values of all runs, indexed (region, family, scenario, variable)."""
    if "P" in _PAIR_CACHE:
        return _PAIR_CACHE["P"], _PAIR_CACHE["G"]
    tr = load_gt("AR6_R10sd_train70_ground_truth.xlsx")
    tr = tr[["region", "model_family", "scenario", "variable"] + YEARS].copy()
    te = gt_test[["id", "region", "model_family", "scenario", "variable"] + YEARS].copy()
    for g in (tr, te):
        g["scenario"] = g["scenario"].astype(str).str.strip()
    G = (pd.concat([tr, te.drop(columns="id")], ignore_index=True)
         .drop_duplicates(["region", "model_family", "scenario", "variable"])
         .set_index(["region", "model_family", "scenario", "variable"])[CONTRAST_YEARS])
    T = tr.drop_duplicates(["region", "model_family", "scenario"])[
        ["region", "model_family", "scenario"]].copy()
    A = te.drop_duplicates("id")[["id", "region", "model_family", "scenario"]].copy()
    for d in (T, A):
        d["design"] = d.scenario.map(_scenario_design)
    T = T[T.design.notna()]
    nbr = _test_same_family_neighbours()
    tgroups = {key: list(g.itertuples()) for key, g in T.groupby(["region", "model_family"])}
    rows = []
    for a in A[A.design.notna()].itertuples():
        study, fa = a.design
        for b in tgroups.get((a.region, a.model_family), []):
            sb, fb = b.design
            if sb != study or b.scenario == a.scenario:
                continue
            diff = [k for k in fa if fa[k] != fb[k]]
            t = _assumption_type(study, diff, fa, fb)
            if t is None:
                continue
            lk = _LEVEL_KEYS.get(study)
            step = (fb[lk] - fa[lk]) if (diff == [lk] and fa[lk] is not None
                                          and fb[lk] is not None) else np.nan
            loose = (fb["budget"] if study == "ENGAGE" and fa["long"] == "reference"
                     and t == "Carbon budget vs reference policy" else np.nan)
            rows.append({"id_A": a.id, "region": a.region, "model_family": a.model_family,
                         "study": study, "A": a.scenario, "B": b.scenario, "type": t,
                         "B_is_neighbour": b.scenario in nbr.get(a.id, set()),
                         "step": step, "loose_budget": loose,
                         "B_f": fb.get("f")})
    P = pd.DataFrame(rows)
    # ordered assumptions: nearest training level below and above A
    lv = P.step.notna()
    L = P[lv].assign(side=np.sign(P.step[lv]), gap=P.step[lv].abs())
    L = L.loc[L.groupby(["id_A", "type", "side"]).gap.idxmin()]
    # reference A against budget runs: the loosest training budget (per f)
    R = P[P.loose_budget.notna()]
    R = R.loc[R.groupby(["id_A", "B_f"]).loose_budget.idxmax()]
    P = pd.concat([P[~lv & P.loose_budget.isna()], L.drop(columns=["side", "gap"]), R],
                  ignore_index=True).drop(columns=["loose_budget", "B_f"])
    _PAIR_CACHE["P"], _PAIR_CACHE["G"] = P, G
    return P, G


def _contrast_entries(gt_test):
    """Pairs expanded to (pair, variable) rows with the 2020-2100 AR6
    contrast d = y_A - y_B and the synthetic contrast dh = yhat_A - y_B.
    Material entries: |d| >= max(0.1 EJ/yr, 5% of max(|y_A|, |y_B|))."""
    P, G = _contrast_pairs(gt_test)
    pred = _qwen_pred().set_index(["id", "variable"])[CONTRAST_YEARS]
    k = P.merge(gt_test[["id", "variable"]].rename(columns={"id": "id_A"}), on="id_A")
    yA = G.reindex(list(zip(k.region, k.model_family, k.A, k.variable))).to_numpy(float)
    yB = G.reindex(list(zip(k.region, k.model_family, k.B, k.variable))).to_numpy(float)
    pA = pred.reindex(list(zip(k.id_A, k.variable))).to_numpy(float)
    d, dh = yA - yB, pA - yB
    ok = np.isfinite(d) & np.isfinite(dh)
    scale = np.maximum(np.abs(yA), np.abs(yB))
    mat = ok & (np.abs(d) >= np.maximum(0.1, 0.05 * scale))
    agree = mat & (np.sign(dh) == np.sign(d))
    return k, d, dh, ok, mat, agree, scale


ASSUMPTION_LABEL = {
    "Carbon budget vs reference policy": "Carbon budget added vs current policy to 2100",
    "Mitigation stringency, nearest level": "Climate target one level stricter or looser",
    "Net-negative CO2 allowed vs not": "Net-negative CO2 allowed vs net zero",
    "Current policies vs NDCs": "Current policies vs NDCs",
    "No policy vs current policies": "No climate policy vs current policy",
    "Delayed action": "Mitigation starting 2020 vs 2030",
    "Carbon-price path": "Carbon-price trajectory",
    "Socioeconomic pathway (SSP)": "Socioeconomic pathway (SSP)",
    "Technology or demand option": "Technology or demand-side measure",
    "Technology cost": "Cost of one technology",
    "Discount rate": "Discount rate",
    "COVID-19 variant": "COVID-19 sensitivity vs default recovery",
    "Updated NDCs": "Updated NDCs (December 2020)",
    "Limited BECCS or bioenergy": "Limited BECCS or CDR",
}


def sheet_contrast_examples(gt_test):
    """Fig. 4g-j: one example pair for each of four frequent assumption
    types (the four most frequent in the test set, except that
    "Carbon budget vs reference policy" - whose training run is rarely a
    retrieved neighbour, 20 test cases from two IAM families - is replaced by
    the next most frequent, "COVID-19 variant"), restricted to pairs whose training run
    B was one of the retrieved same-family neighbours in the finetuning prompt
    of A (so the model saw B and had to move away from it). Rule (no hand
    picking): per type, candidates are those pairs with >= 5 material years
    among the 10% with the largest AR6 change (max |y_A - y_B|, EJ/yr), ranked
    by |beta - median beta of the panel's candidate pool| (the type's neighbour
    pairs, restricted by POOL below). The four panels are chosen JOINTLY as
    the combination with the smallest sum of ranks that uses a different IAM
    family and a different variable in each panel and at most two panels from
    the same study. If no combination exists, the 10% filter is dropped.
    POOL: panel g comes from a study other than ENGAGE and from an IAM
    family other than REMIND, so that at most two panels are ENGAGE (COVID-19
    variant and net-negative CO2 exist only in ENGAGE) and REMIND appears once
    (j: current policies vs NDCs outside ENGAGE exist only in REMIND R2p1).
    Long format: one row per (panel, series, year)."""
    POOL = {"g": lambda x: (x.study != "ENGAGE") & (x.model_family != "REMIND")}
    k, d, dh, ok, mat, agree, scale = _contrast_entries(gt_test)
    P, G = _contrast_pairs(gt_test)
    pred = _qwen_pred().set_index(["id", "variable"])[CONTRAST_YEARS]
    d0, h0 = np.where(ok, d, 0.0), np.where(ok, dh, 0.0)
    with np.errstate(all="ignore"):
        beta = np.where(mat.any(1), (h0 * d0).sum(1) / (d0 ** 2).sum(1), np.nan)
    k = k.assign(beta=beta, nmat=mat.sum(1),
                 dmax=np.nanmax(np.where(ok, np.abs(d), np.nan), axis=1))
    top4 = ["Mitigation stringency, nearest level", "COVID-19 variant",
            "Net-negative CO2 allowed vs not", "Current policies vs NDCs"]
    panels = list("ghij")

    def candidates(t, panel, top10):
        nb = k[(k.type == t) & k.B_is_neighbour]
        if panel in POOL:
            nb = nb[POOL[panel](nb)]
        med = np.nanmedian(nb.beta)
        g = nb[nb.nmat >= 5]
        if top10:
            g = g[g.dmax >= g.dmax.quantile(0.90)]
        g = (g.assign(dist=(g.beta - med).abs().round(6), neg=-g.dmax)
               .sort_values(["dist", "neg", "id_A", "variable"], kind="mergesort"))
        return med, list(g.itertuples())[:200]

    def search(cands):
        best = [None, np.inf]

        def dfs(j, chosen, cost):
            if cost >= best[1]:
                return
            if j == len(cands):
                best[0], best[1] = list(chosen), cost
                return
            for rank, r in enumerate(cands[j]):
                if cost + rank >= best[1]:
                    break
                fams = [c.model_family for c in chosen]
                vars_ = [c.variable for c in chosen]
                studs = [c.study for c in chosen]
                if r.model_family in fams or r.variable in vars_ or studs.count(r.study) >= 2:
                    continue
                chosen.append(r)
                dfs(j + 1, chosen, cost + rank)
                chosen.pop()
        dfs(0, [], 0)
        return best[0]

    picks = None
    for top10 in (True, False):
        meds, cands = zip(*[candidates(t, p, top10) for t, p in zip(top4, panels)])
        picks = search(list(cands))
        if picks:
            break
    rows = []
    for panel, t, med, r in zip(panels, top4, meds, picks):
        series = {"training run (B)": G.loc[(r.region, r.model_family, r.B, r.variable)],
                  "AR6 test case (A)": G.loc[(r.region, r.model_family, r.A, r.variable)],
                  "synthetic test case (A)": pred.loc[(r.id_A, r.variable)]}
        for s, v in series.items():
            for y, val in zip(CONTRAST_YEARS, v.to_numpy(float)):
                rows.append({"panel": panel, "assumption": t, "study": r.study,
                             "label": ASSUMPTION_LABEL[t], "model_family": r.model_family,
                             "region": r.region, "variable": r.variable,
                             "training_scenario_B": r.B, "test_scenario_A": r.A,
                             "B_was_retrieved_neighbour": bool(r.B_is_neighbour),
                             "beta": round(float(r.beta), 3),
                             "median_beta_of_candidate_pool": round(float(med), 3),
                             "series": s, "year": int(y), "value": round(float(val), 3)})
    return pd.DataFrame(rows)


def sheet_km_availability(gt_val):
    """The k_m sweep (0/1/2/3 same-scenario, different-IAM neighbours) split
    by whether such a neighbour EXISTS for the case.  For 37% of validation
    cases none does, so all four prompts are identical there and any spread
    is LLM noise; an all-case aggregate is diluted by them.
    Same algorithm as fig2_rag_val_ablation: MAE averaged over
    trajectories, paired case bootstrap (boot_mae, B/SEED as elsewhere),
    and k_m=1 = the hard-coded topk210 run TOPK210_RUN."""
    def one(pattern):
        p = rv(pattern); assert len(p) == 1, (pattern, p); return p[0]
    files = {0: one("*pred*topk200_fp-papers*desc-on_rrll15x3*.csv"),
             1: one(TOPK210_VAL),
             2: one("*pred*topk220_fp-papers*.csv"),
             3: one("*pred*topk230_fp-papers*.csv")}
    tabs = {k: traj_table(f, gt_val) for k, f in files.items()}
    nb = pd.read_excel(OUTS / "neighbor_stats_val.xlsx",
                       sheet_name="per_case").set_index("id")
    ids = nb.index
    for t in tabs.values():
        ids = ids.intersection(pd.Index(t["id"].unique()))
    nb = nb.loc[ids]
    groups = {"k_m unavailable": nb.index[nb.scen_nbrs.eq(0)],
              "k_m available": nb.index[nb.scen_nbrs.ge(1)],
              "all cases": nb.index}
    rows = []
    for g, gid in groups.items():
        sub = {k: t[t["id"].isin(gid)] for k, t in tabs.items()}
        bm = boot_mae(sub).drop(columns="config")
        bm.insert(0, "k_m", list(sub.keys()))
        bm.insert(0, "group", g)
        bm["pred_file"] = [Path(files[k]).name for k in sub]
        rows.append(bm)
    return pd.concat(rows, ignore_index=True)


def _pointwise(gt_test):
    """Long table of every (method, id, variable, year) prediction error on the
    test set: abs error and sMAPE term, with the GT value. Missing predicted
    values count as 0.0 (project convention); GT-missing cells are dropped."""
    frames = []
    base = gt_test[["id", "variable"] + YEARS]
    for lab, rel in TEST_PREDS.items():
        p = pd.read_csv(OUTS / rel); p.columns = [str(c).strip() for c in p.columns]
        m = base.merge(p[["id", "variable"] + YEARS], on=["id", "variable"],
                       how="left", suffixes=("_t", "_p"), validate="one_to_one")
        yt = m[[f"{y}_t" for y in YEARS]].to_numpy(float)
        yp = np.round(m[[f"{y}_p" for y in YEARS]].to_numpy(float), 2)
        yp = np.where(np.isfinite(yp), yp, 0.0)
        ok = np.isfinite(yt)
        ii, jj = np.nonzero(ok)
        a, p_ = yt[ii, jj], yp[ii, jj]
        den = np.maximum((np.abs(a) + np.abs(p_)) / 2.0, 1e-12)
        frames.append(pd.DataFrame({
            "method": lab, "id": m["id"].to_numpy()[ii],
            "variable": m["variable"].to_numpy()[ii],
            "year": np.array(YEARS)[jj].astype(int),
            "gt": a, "abs_err": np.abs(a - p_),
            "smape_term": np.abs(a - p_) / den * 100.0}))
    return pd.concat(frames, ignore_index=True)


def _boot_by(pw, key):
    """Mean abs error / sMAPE per (method, key) with a paired case bootstrap
    (same case draws for every method and key level)."""
    ids = np.array(sorted(pw["id"].unique()))
    rng = np.random.default_rng(SEED)
    takes = [rng.integers(0, len(ids), len(ids)) for _ in range(B)]
    rows = []
    for (lab, lv), g in pw.groupby(["method", key], sort=False):
        agg = g.groupby("id").agg(n=("abs_err", "size"), ae=("abs_err", "sum"),
                                  sm=("smape_term", "sum"))
        agg = agg.reindex(ids).fillna(0.0)
        n, ae, sm = (agg[c].to_numpy(float) for c in ("n", "ae", "sm"))
        bm = np.array([ae[k].sum() / max(n[k].sum(), 1) for k in takes])
        bs = np.array([sm[k].sum() / max(n[k].sum(), 1) for k in takes])
        rows.append({"method": lab, key: lv, "n_points": int(n.sum()),
                     "n_cases": int((n > 0).sum()),
                     "mae": round(float(ae.sum() / n.sum()), 4),
                     "mae_ci2.5": round(float(np.percentile(bm, 2.5)), 4),
                     "mae_ci97.5": round(float(np.percentile(bm, 97.5)), 4),
                     "smape_%": round(float(sm.sum() / n.sum()), 4),
                     "smape_ci2.5": round(float(np.percentile(bs, 2.5)), 4),
                     "smape_ci97.5": round(float(np.percentile(bs, 97.5)), 4),
                     "mean_abs_gt": round(float(g["gt"].abs().mean()), 4)})
    return pd.DataFrame(rows)


def sheet_error_by_variable(gt_test):
    """Test-set error of the four methods per variable (all decades
    pooled); mean_abs_gt gives the variable's scale for reading MAE."""
    out = _boot_by(_pointwise(gt_test), "variable")
    return out.sort_values(["method", "variable"]).reset_index(drop=True)


def sheet_error_by_year(gt_test):
    """Supplementary Fig. S5a: test-set error of the four methods in each year
    (all variables pooled). The figure shows the finetuned model."""
    out = _boot_by(_pointwise(gt_test), "year")
    return out.sort_values(["method", "year"]).reset_index(drop=True)


def extremum_timing(truth, prediction, kind="maximum"):
    """Return minimum year distance between all exactly tied extrema.

    Endpoint extrema and partial ties are valid, without a range threshold.
    Constant AR6 trajectories have undefined timing (NaN); a constant
    prediction for nonconstant AR6 is a failure at every tolerance (infinity).
    Input prediction precision is set by the calling evaluator.
    """
    if kind not in ("maximum", "minimum"):
        raise ValueError("kind must be maximum or minimum")
    reduce = np.max if kind == "maximum" else np.min
    true_years = truth == reduce(truth, axis=1, keepdims=True)
    pred_years = prediction == reduce(prediction, axis=1, keepdims=True)
    years = np.array([int(y) for y in YEARS])
    gaps = np.abs(years[:, None] - years[None, :])
    eligible = true_years[:, :, None] & pred_years[:, None, :]
    distance = np.where(eligible, gaps, np.inf).min(axis=(1, 2))
    distance[np.ptp(prediction, axis=1) == 0] = np.inf
    distance[np.ptp(truth, axis=1) == 0] = np.nan
    return distance, true_years, pred_years


def extremum_offset_weights(true_years, pred_years):
    """Nearest signed offsets (prediction minus AR6), one unit per trajectory.

    If equally close pairs exist in both directions, each direction receives
    half the trajectory's weight. Multiple pairs with the same offset do not
    increase that offset's weight. Overlapping year sets receive zero offset.
    Constant AR6 or predicted trajectories have no signed offset and receive
    zero weight in the numerical bins. The caller retains constant prediction
    failures in the denominator and reports their share separately.
    """
    years = np.array([int(y) for y in YEARS])
    gaps = years[None, :] - years[:, None]
    eligible = true_years[:, :, None] & pred_years[:, None, :]
    nearest = np.where(eligible, np.abs(gaps), np.inf).min(axis=(1, 2))
    closest = eligible & (np.abs(gaps)[None, :, :] == nearest[:, None, None])
    offsets = np.unique(gaps)
    present = np.stack([(closest & (gaps == k)).any(axis=(1, 2))
                        for k in offsets], axis=1)
    weights = present / present.sum(axis=1, keepdims=True)
    weights[true_years.all(axis=1) | pred_years.all(axis=1)] = 0
    return offsets, weights


def sheet_shape_fidelity(gt_test):
    """Fig. 3c,d: extrema timing and maximum magnitude.

    Timing is assessed on complete, nonconstant AR6 trajectories, including
    endpoint extrema and partial ties. Constant predictions are timing failures.
    Otherwise timing error is the minimum distance between the two year sets.
    Maximum-magnitude metrics use the same nonconstant AR6 sample, including
    constant predictions and all other timing failures.
    Predictions are rounded to two decimals. Columns named peak/trough hold
    the global maximum/minimum, not interior turning points. Timing shares
    carry paired case-bootstrap CIs.
    """
    base = gt_test[["id", "variable"] + YEARS]
    yt = base[YEARS].to_numpy(float)
    full = np.isfinite(yt).all(axis=1)
    yt, ids = yt[full], base["id"].to_numpy()[full]
    peak = trough = np.ones(len(yt), dtype=bool)
    timing = np.ptp(yt, axis=1) != 0
    positive_max = timing & (yt.max(1) > 0)
    uid = np.unique(ids)
    rng = np.random.default_rng(SEED)
    takes = [rng.choice(uid, len(uid)) for _ in range(B)]
    pos = {u: np.flatnonzero(ids[timing] == u) for u in uid}

    def share_ci(hit, mask):
        """share of hits within mask, case bootstrap."""
        h = hit & mask
        per_n = pd.Series(mask.astype(int)).groupby(ids).sum()
        per_h = pd.Series(h.astype(int)).groupby(ids).sum()
        n_ = per_n.reindex(uid).to_numpy(float); h_ = per_h.reindex(uid).to_numpy(float)
        idx = {u: i for i, u in enumerate(uid)}
        bs = []
        for tk in takes:
            ii = np.fromiter((idx[u] for u in tk), int, len(tk))
            bs.append(100 * h_[ii].sum() / max(n_[ii].sum(), 1))
        return (round(100 * h.sum() / max(mask.sum(), 1), 2),
                round(float(np.percentile(bs, 2.5)), 2),
                round(float(np.percentile(bs, 97.5)), 2))

    rows = []
    for lab, rel in TEST_PREDS.items():
        p = pd.read_csv(OUTS / rel); p.columns = [str(c).strip() for c in p.columns]
        m = base.merge(p[["id", "variable"] + YEARS], on=["id", "variable"],
                       how="left", suffixes=("_t", "_p"), validate="one_to_one")
        yp = np.round(m[[f"{y}_p" for y in YEARS]].to_numpy(float), 2)[full]
        yp = np.where(np.isfinite(yp), yp, 0.0)
        dpk, gt_max_years, pred_max_years = extremum_timing(yt, yp, "maximum")
        dtr, gt_min_years, pred_min_years = extremum_timing(yt, yp, "minimum")
        dir_ok = np.sign(yt[:, -1] - yt[:, 0]) == np.sign(yp[:, -1] - yp[:, 0])
        r = {"method": lab, "n_trajectories": int(full.sum()),
             "n_peak": int(peak.sum()), "n_trough": int(trough.sum()),
             "n_constant_ar6": int((np.ptp(yt, axis=1) == 0).sum()),
             "n_constant_prediction": int((np.ptp(yp, axis=1) == 0).sum()),
             "n_tied_maximum_ar6": int((gt_max_years.sum(1) > 1).sum()),
             "n_tied_minimum_ar6": int((gt_min_years.sum(1) > 1).sum()),
             "n_maximum_relative_error": int(positive_max.sum())}
        constant_failure = timing & (np.ptp(yp, axis=1) == 0)
        r["n_peak"] = r["n_trough"] = r["n_timing_eligible"] = int(timing.sum())
        r["n_constant_prediction_failures"] = int(constant_failure.sum())
        r["constant_prediction_failure_%"] = round(100 * constant_failure.sum() / timing.sum(), 4)
        for name, hit, mask in [("peak_year_exact", dpk == 0, timing),
                                ("peak_year_within10", dpk <= 10, timing),
                                ("peak_year_within20", dpk <= 20, timing),
                                ("trough_year_exact", dtr == 0, timing),
                                ("trough_year_within10", dtr <= 10, timing),
                                ("direction_correct", dir_ok, peak)]:
            v, lo, hi = share_ci(hit, mask)
            r[f"{name}_%"] = v; r[f"{name}_ci2.5"] = lo; r[f"{name}_ci97.5"] = hi
        # Absolute minimum timing gap, with the final bin including >=30 years.
        for name, delta in [("peak", dpk), ("trough", dtr)]:
            # Undefined prediction timing is not a >=30-year error.
            off = np.where(np.isfinite(delta), np.minimum(delta, 30), np.nan)
            for k in (0, 10, 20, 30):
                r[f"{name}_absolute_error_{k}_%"] = round(
                    100 * float((off[timing] == k).mean()), 4)
        dv_all = yp.max(1) - yt.max(1)
        dv = dv_all[timing]
        r["peak_value_bias"] = round(float(dv.mean()), 4)
        r["peak_value_bias_ci2.5"], r["peak_value_bias_ci97.5"] = (
            round(float(x), 4) for x in np.percentile(
                [dv[np.concatenate([pos[u] for u in tk])].mean()
                 for tk in takes], [2.5, 97.5]))
        r["peak_value_rel_err_median"] = round(float(np.median(
            np.abs(dv_all[positive_max]) / yt.max(1)[positive_max])), 4)
        # Predictive R^2 on untransformed maxima for the full timing sample.
        # Includes zero/constant predictions and timing failures; no floor.
        gp_, pp_ = yt.max(1)[timing], yp.max(1)[timing]
        r["peak_height_r2"] = round(float(
            1 - np.sum((pp_ - gp_) ** 2) / np.sum((gp_ - gp_.mean()) ** 2)), 4)
        r["timing_scope"] = "Nonconstant AR6 trajectories; endpoint extrema and partial ties included; constant predictions count as failures; no range threshold"
        r["extremum_tie_rule"] = "Minimum distance between all tied AR6 and predicted extremum years; intersection is exact"
        r["prediction_decimal_places"] = 2
        for name, true_years, pred_years in [
                ("peak", gt_max_years, pred_max_years),
                ("trough", gt_min_years, pred_min_years)]:
            offsets, weights = extremum_offset_weights(true_years, pred_years)
            clipped = np.clip(offsets, -30, 30)
            for k in (-30, -20, -10, 0, 10, 20, 30):
                r[f"{name}_offset_{k:+d}_%"] = round(
                    100 * float(weights[timing][:, clipped == k].sum(axis=1).mean()), 4)
        r["signed_offset_tie_rule"] = "Nearest predicted-minus-AR6 year offset; equally near negative and positive offsets receive half weight each"
        r["n_maximum_magnitude"] = int(timing.sum())
        r["magnitude_scope"] = "Same nonconstant AR6 sample as timing; constant predictions and other timing failures retained"
        rows.append(r)
    return pd.DataFrame(rows)


TRAJECTORY_GROUPS = [   # Fig. 3a: three groups of primary energy sources and the three final energy sectors
    ("primary", "Fossil fuels without CCS", ["Primary Energy|Coal|w/o CCS", "Primary Energy|Gas|w/o CCS", "Primary Energy|Oil|w/o CCS"]),
    ("primary", "Fossil fuels with CCS", ["Primary Energy|Coal|w/ CCS", "Primary Energy|Gas|w/ CCS", "Primary Energy|Oil|w/ CCS"]),
    ("primary", "Non-fossil sources", ["Primary Energy|Biomass", "Primary Energy|Hydro", "Primary Energy|Nuclear",
                                       "Primary Energy|Solar", "Primary Energy|Wind"]),
    ("final", "Industry", ["Final Energy|Industry"]),
    ("final", "Buildings", ["Final Energy|Residential and Commercial"]),
    ("final", "Transport", ["Final Energy|Transportation"]),
]


def sheet_group_trajectories(gt_test):
    """Fig. 3a: per group, year and source the mean and the 10th / 50th / 90th
    percentiles across test cases of the case-level sum of the variables of the
    group, AR6 vs Qwen3.5-9B. The three primary groups contain the 11 primary energy
    variables and the three sectors add up to total final energy. A case
    enters a group in a year only when every variable of the group is present
    in both AR6 and the prediction."""
    m = _test_pairs(gt_test); rows = []
    for energy, group, members in TRAJECTORY_GROUPS:
        d = m[m.variable.isin(members)]
        for y in YEARS:
            k = d[["id", f"{y}_ar6", f"{y}_syn"]].dropna(); cnt = k.groupby("id").size()
            tot = k[k.id.isin(cnt[cnt == len(members)].index)].groupby("id")[[f"{y}_ar6", f"{y}_syn"]].sum()
            for src, col in (("AR6", f"{y}_ar6"), ("synthetic", f"{y}_syn")):
                v = tot[col].to_numpy(float); q = np.percentile(v, [10, 50, 90])
                rows.append({"energy": energy, "group": group, "source": src, "year": int(y), "n_cases": int(len(tot)),
                             "mean": round(float(v.mean()), 4), "p10": round(float(q[0]), 4), "median": round(float(q[1]), 4),
                             "p90": round(float(q[2]), 4)})
    return pd.DataFrame(rows)


TEST_PREDS_QWEN = "Finetune Qwen3.5-9B (lr5e-4)"

def _qwen_pred():
    p = pd.read_csv(OUTS / TEST_PREDS[TEST_PREDS_QWEN])
    p.columns = [str(c).strip() for c in p.columns]
    return p


def sheet_box_quartiles(gt_test):
    """Supplementary Fig. S4: per variable, year (2050, 2100) and source (AR6 / synthetic),
    the box statistics drawn in the figure - Q1, median, Q3 and the whisker
    ends (last value inside 1.5 IQR) - on the same test cases (values below
    0.01 EJ/yr are clipped to 0.01 before the statistics, as in the plot)."""
    pred = _qwen_pred(); rows = []
    for year in ("2050", "2100"):
        for v in sorted(gt_test.variable.unique()):
            g = gt_test[gt_test.variable.eq(v)].set_index("id")[year].astype(float).dropna()
            p = pred[pred.variable.eq(v)].set_index("id")[year].astype(float)
            ids = g.index.intersection(p.index)
            for src, vals in (("AR6", g.loc[ids]), ("synthetic", p.loc[ids])):
                x = np.maximum(vals.to_numpy(float), 0.01)
                q1, med, q3 = np.percentile(x, [25, 50, 75]); iqr = q3 - q1
                rows.append({"variable": v, "year": int(year), "source": src,
                             "n_cases": int(len(x)),
                             "whisker_low": round(float(x[x >= q1 - 1.5 * iqr].min()), 4),
                             "q1": round(float(q1), 4), "median": round(float(med), 4),
                             "q3": round(float(q3), 4),
                             "whisker_high": round(float(x[x <= q3 + 1.5 * iqr].max()), 4)})
    return pd.DataFrame(rows)


def sheet_peak_height(gt_test):
    """Fig. 3c,d: maxima and all tied extremum years for every trajectory.

    Columns named peak hold maxima, including endpoint maxima.
    The timing_eligible flag defines the shared sample for both panels.
    Constant AR6 rows are retained only for auditing and are not plotted.
    """
    pred = _qwen_pred()
    base = gt_test[["id", "variable"] + YEARS]
    m = base.merge(pred[["id", "variable"] + YEARS], on=["id", "variable"],
                   how="left", suffixes=("_t", "_p"), validate="one_to_one")
    yt = m[[f"{y}_t" for y in YEARS]].to_numpy(float)
    yp = np.round(m[[f"{y}_p" for y in YEARS]].to_numpy(float), 2)
    full = np.isfinite(yt).all(axis=1)
    yt, yp = yt[full], np.where(np.isfinite(yp[full]), yp[full], 0.0)
    dmax, gmax, pmax = extremum_timing(yt, yp, "maximum")
    dmin, gmin, pmin = extremum_timing(yt, yp, "minimum")
    Y = np.array([int(y) for y in YEARS])
    def year_lists(mask):
        return ["; ".join(map(str, Y[row])) for row in mask]
    out = pd.DataFrame({"id": m["id"].to_numpy()[full],
                        "variable": m["variable"].to_numpy()[full],
                        "ar6_peak_height": yt.max(1).round(4),
                        "ar6_maximum_years": year_lists(gmax),
                        "synthetic_peak_height": yp.max(1).round(4),
                        "synthetic_maximum_years": year_lists(pmax),
                        "ar6_minimum_years": year_lists(gmin),
                        "synthetic_minimum_years": year_lists(pmin),
                        "maximum_year_absolute_error": np.where(np.isfinite(dmax), dmax, np.nan),
                        "minimum_year_absolute_error": np.where(np.isfinite(dmin), dmin, np.nan),
                        "timing_eligible": np.ptp(yt, axis=1) != 0,
                        "timing_status": np.where(np.ptp(yt, axis=1) == 0,
                            "Excluded: constant AR6", np.where(np.ptp(yp, axis=1) == 0,
                            "Incorrect: constant prediction", "Defined extremum timing"))})
    return out.reset_index(drop=True)


def _test_pairs(gt_test):
    """AR6 test truth and the Qwen prediction, one row per trajectory (24,805):
    id, model_family, scenario, region, variable, 2010_ar6 .. 2100_ar6, 2010_syn .. 2100_syn."""
    pred = _qwen_pred()
    m = gt_test[["id", "model_family", "scenario", "region", "variable"] + YEARS].merge(
        pred[["id", "variable"] + YEARS], on=["id", "variable"], suffixes=("_ar6", "_syn"))
    assert len(m) == len(gt_test), (len(m), len(gt_test))
    return m.reset_index(drop=True)

def sheet_all_values(gt_test):
    """Supplementary Fig. S5b: pooled statistics over all 248,050 decadal values (24,805 trajectories x 10 decades)."""
    m = _test_pairs(gt_test)
    xa = m[[f"{y}_ar6" for y in YEARS]].to_numpy(float).ravel(); xs = m[[f"{y}_syn" for y in YEARS]].to_numpy(float).ravel()
    e = np.abs(xs - xa)
    return pd.DataFrame([dict(quantity="n_values", value=len(e)),
                         dict(quantity="median_abs_error_EJ", value=float(np.median(e))),
                         dict(quantity="share_within_0.1_EJ", value=float((e <= 0.1).mean())),
                         dict(quantity="share_within_0.5_EJ", value=float((e <= 0.5).mean())),
                         dict(quantity="share_within_1_EJ", value=float((e <= 1.0).mean())),
                         dict(quantity="pooled_pearson_r", value=float(np.corrcoef(xa, xs)[0, 1])),
                         dict(quantity="pooled_pearson_r_log10_plus_0.01", value=float(np.corrcoef(np.log10(xa + 0.01), np.log10(xs + 0.01))[0, 1])),
                         dict(quantity="pred_file", value=TEST_PREDS[TEST_PREDS_QWEN])])


HELDOUT_CELLS = [("MESSAGEix-GLOBIOM", "EN_NPi2020_500f", "differs"), ("GEM-E3", "EN_INDCi2030_600f", "differs"),
                 ("WITCH", "EN_NPi2020_600f", "within"), ("COFFEE", "EN_NPi2100", "within")]
_PRIM = ["Primary Energy|Coal|w/ CCS", "Primary Energy|Coal|w/o CCS", "Primary Energy|Gas|w/ CCS", "Primary Energy|Gas|w/o CCS",
         "Primary Energy|Oil|w/ CCS", "Primary Energy|Oil|w/o CCS", "Primary Energy|Solar", "Primary Energy|Wind", "Primary Energy|Hydro",
         "Primary Energy|Nuclear", "Primary Energy|Biomass"]
_FIN3 = ["Final Energy|Industry", "Final Energy|Transportation", "Final Energy|Residential and Commercial"]
_ELE3 = [v + "|Electricity" for v in _FIN3]
HELDOUT_INDICATORS = [("fossil_unab_share", "Fossil without CCS (% of PE)"), ("ccs_share", "Fossil with CCS (% of PE)"),
                      ("sw_share", "Wind and solar (% of PE)"), ("bio_share", "Biomass (% of PE)"), ("nuc_share", "Nuclear (% of PE)"),
                      ("FE_total", "Final energy (EJ)"), ("ind_elec_share", "Electricity in industry (%)"),
                      ("tr_elec_share", "Electricity in transport (%)"), ("bld_elec_share", "Electricity in buildings (%)")]

def _shares_2050(df, family, scenario, year="2050"):
    """2050 energy-system indicators of one family x scenario, summed over the R10 regions."""
    d = df[(df.model_family == family) & (df.scenario == scenario)]
    if d.empty: return {}
    g = lambda v: float(d[d.variable == v][year].sum())
    pe = sum(g(v) for v in _PRIM); fe = sum(g(v) for v in _FIN3); o = {}
    if pe > 0:
        o.update(fossil_unab_share=100 * (g(_PRIM[1]) + g(_PRIM[3]) + g(_PRIM[5])) / pe, sw_share=100 * (g(_PRIM[6]) + g(_PRIM[7])) / pe,
                 bio_share=100 * g(_PRIM[10]) / pe, ccs_share=100 * (g(_PRIM[0]) + g(_PRIM[2]) + g(_PRIM[4])) / pe, nuc_share=100 * g(_PRIM[9]) / pe)
    if fe > 0:
        o["FE_total"] = fe
        for sec, (fv, ev) in {"ind": (_FIN3[0], _ELE3[0]), "tr": (_FIN3[1], _ELE3[1]), "bld": (_FIN3[2], _ELE3[2])}.items():
            if g(fv) > 0: o[f"{sec}_elec_share"] = 100 * g(ev) / g(fv)
    return o

def _heldout_tables(gt_test):
    real = pd.read_excel(sorted((DATA_ROOT / "make_datasets" / "output").glob("AR6_R10_sd_mixed_*.xlsx"))[-1])
    real.columns = [str(c) for c in real.columns]
    pred = _qwen_pred()
    testc = set(map(tuple, gt_test[["model_family", "scenario"]].drop_duplicates().values))
    train = real[~real.set_index(["model_family", "scenario"]).index.isin(list(testc))]   # the other families' runs: training data only
    rows, cells = [], []
    for F, A, kind in HELDOUT_CELLS:
        assert (F, A) in testc, (F, A)
        others = sorted(g for g in train[train.scenario == A].model_family.unique() if g != F)
        r, p = _shares_2050(gt_test, F, A), _shares_2050(pred, F, A)
        for k, lab in HELDOUT_INDICATORS:
            if k not in r: continue
            ov = {g: _shares_2050(train, g, A).get(k, np.nan) for g in others}; ov = {g: v for g, v in ov.items() if not np.isnan(v)}
            if len(ov) < 1: continue
            for g, v in ov.items():
                rows.append(dict(family=F, scenario=A, kind=kind, indicator=k, label=lab, series="other family", other_family=g, value=v))
            rows.append(dict(family=F, scenario=A, kind=kind, indicator=k, label=lab, series="synthetic", other_family="", value=p[k]))
            rows.append(dict(family=F, scenario=A, kind=kind, indicator=k, label=lab, series="AR6 truth", other_family="", value=r[k]))
        cells.append(dict(family=F, scenario=A, kind=kind, n_other_families=len(others), other_families=", ".join(others)))
    return pd.DataFrame(rows), pd.DataFrame(cells)

_HELDOUT_CACHE = {}
def sheet_heldout(gt_test, which):
    if "t" not in _HELDOUT_CACHE: _HELDOUT_CACHE["t"] = _heldout_tables(gt_test)
    return _HELDOUT_CACHE["t"][0 if which == "profiles" else 1]


# ═══════════════════════════ builders, sheet layout and the workbook writer ═══
# A workbook has one sheet per figure. A sheet holds the data tables of the
# figure one below the other: a title row, the table with its header, then an
# empty row. The tables are identified by their order on the sheet, which
# make_main_figs.MAIN_BLOCKS defines for the build and the plotting scripts.
# STAT_BUILDERS / STAT_LAYOUT: the complete statistics, including quantities
# that are quoted in the text or only used on the way to a figure. They are
# written to main_statistics.xlsx with --statistics and are not Source Data.
STAT_BUILDERS = {
    # ── Fig. 2: RAG ablation + four-method test comparison ─────────────────
    "fig2_rag_val_ablation":      ("rag",    lambda c: sheet1(c["gt_val"])),
    "fig2_km_availability":       ("kmav",   lambda c: sheet_km_availability(c["gt_val"])),
    "fig2_test_four_methods":     ("test",   lambda c: sheet5(c["gt_test"])),
    # ── Fig. 3: fidelity of the finetuned model ────────────────────────────
    "fig3_group_trajectories":    ("grp3",   lambda c: sheet_group_trajectories(c["gt_test"])),
    "fig3_error_by_variable":     ("byvar",  lambda c: sheet_error_by_variable(c["gt_test"])),
    "fig3_shape_fidelity":        ("shape",  lambda c: sheet_shape_fidelity(c["gt_test"])),
    "fig3_peak_height":           ("peak",   lambda c: sheet_peak_height(c["gt_test"])),
    # ── Fig. 4: coverage, strata, one-assumption examples ──────────────────
    "fig4_scenario_coverage":     ("dist",   lambda c: sheet_coverage()),
    "fig4_group_metrics":         ("grp",    lambda c: sheet7()),
    "fig4_nearest_desc_scenario": ("nds",    lambda c: sheet_nearest_desc_scenario(c["gt_test"])),
    "fig4_error_scales":          ("scales", lambda c: sheet_error_scales(c["gt_test"])),
    "fig4_contrast_examples":     ("conex",  lambda c: sheet_contrast_examples(c["gt_test"])),
    # ── Fig. 5: held-out cells vs the other families ───────────────────────
    "fig5_heldout_cells":         ("held",   lambda c: sheet_heldout(c["gt_test"], "cells")),
    "fig5_heldout_profiles":      ("held",   lambda c: sheet_heldout(c["gt_test"], "profiles")),
}
STAT_LAYOUT = {   # sheet -> [(block, panels, description)]
    "Fig2": [("fig2_rag_val_ablation", "a, c", "validation MAE of the zero-shot RAG pipeline per retrieval / prompt configuration, bootstrap 95% CI"),
             ("fig2_km_availability", "b", "validation MAE as the number of same-scenario neighbours k_m changes, by availability group"),
             ("fig2_test_four_methods", "d-g", "test-set MAE, sMAPE, Pearson r and Spearman rho of the four methods, bootstrap 95% CI")],
    "Fig3": [("fig3_group_trajectories", "a", "mean and 10th / 50th / 90th percentiles across test cases per group of variables, year and source"),
             ("fig3_error_by_variable", "b", "MAE and sMAPE per variable, bootstrap 95% CI"),
             ("fig3_shape_fidelity", "c, d", "extremum-timing offsets and maximum-value statistics"),
             ("fig3_peak_height", "d", "AR6 and synthetic maximum per trajectory, with timing eligibility")],
    "Fig4": [("fig4_scenario_coverage", "a", "scenarios per design dimension x IAM family, with coverage status"),
             ("fig4_group_metrics", "b-f", "test-set metrics and case counts per stratum"),
             ("fig4_nearest_desc_scenario", "c-f", "dotted reference: error of the description-nearest scenario of the same family and region"),
             ("fig4_error_scales", "c-f", "dash-dotted reference: difference between IAM families under the same scenario and region"),
             ("fig4_contrast_examples", "g-j", "training run B, AR6 test case A and synthetic A for the four one-assumption examples")],
    "Fig5": [("fig5_heldout_cells", "a-d", "the four held-out IAM family x scenario cells and the other families that ran the scenario"),
             ("fig5_heldout_profiles", "a-d", "2050 indicators: other families (training data), synthetic held-out family, its AR6 truth")],
}
# ═════════════ figure blocks: exactly the data drawn in each panel ═══════════
# The functions above compute the complete statistics (STAT_BUILDERS, written
# only with --statistics). The functions below reduce them to what a figure
# draws: main_figure_data.xlsx holds these blocks and nothing else, and
# make_main_figs.py plots them without further computation. No block holds
# the AR6 trajectories themselves.
def _memo(c, name, table=None):
    """Complete statistics table `name`, computed once per run."""
    table = STAT_BUILDERS if table is None else table
    m = c.setdefault("_memo", {})
    if name not in m: m[name] = table[name][1](c)
    return m[name]

def fig2_ablation(c):
    d = _memo(c, "fig2_rag_val_ablation")
    assert TOPK210_RUN in str(d[d.config.eq("topk210")].pred_file.iloc[0])
    return d[["config", "n_cases", "mae", "ci2.5", "ci97.5"]].reset_index(drop=True)

def fig2_km(c):
    d = _memo(c, "fig2_km_availability"); d = d[d.group.eq("k_m available")]
    assert TOPK210_RUN in str(d[d.k_m.eq(1)].pred_file.iloc[0])
    return d[["k_m", "n_cases", "mae", "ci2.5", "ci97.5"]].reset_index(drop=True)

_METRICS4 = ["mae", "smape_%", "pearson", "spearman"]

def fig2_methods(c):
    d = _memo(c, "fig2_test_four_methods")
    return d[["method"] + [m + s for m in _METRICS4 for s in ("", "_ci2.5", "_ci97.5")]].reset_index(drop=True)

def fig3_groups(c):
    return _memo(c, "fig3_group_trajectories")[["energy", "group", "source", "year", "n_cases", "mean"]]

def fig3_error(c):
    d = _memo(c, "fig3_error_by_variable"); d = d[d.method.eq(TEST_PREDS_QWEN)]
    return d[["variable", "mae", "mae_ci2.5", "mae_ci97.5", "smape_%", "smape_ci2.5", "smape_ci97.5"]].reset_index(drop=True)

DENSITY_GRIDSIZE = 55

def value_density(m):
    """Supplementary Fig. S5b: all decadal values of the test set, AR6 against
    synthetic, counted on the hexagonal grid of the figure. Axes are
    log10(value + 0.01). `m` is the table of _test_pairs. Returns (bins, grid)."""
    from matplotlib.figure import Figure
    xa = m[[f"{y}_ar6" for y in YEARS]].to_numpy(float).ravel(); xs = m[[f"{y}_syn" for y in YEARS]].to_numpy(float).ravel()
    ok = np.isfinite(xa) & np.isfinite(xs); f = lambda v: np.log10(np.clip(v, 0, None) + 0.01)
    xa, xs = f(xa[ok]), f(xs[ok])
    hb = Figure().subplots().hexbin(xa, xs, gridsize=DENSITY_GRIDSIZE, mincnt=1)
    off = hb.get_offsets(); cnt = np.asarray(hb.get_array()).astype(int)
    assert cnt.sum() == len(xa)
    bins = pd.DataFrame({"ar6_log10": off[:, 0], "synthetic_log10": off[:, 1], "n_values": cnt})
    grid = pd.DataFrame([{"gridsize": DENSITY_GRIDSIZE, "ar6_log10_min": xa.min(), "ar6_log10_max": xa.max(),
                          "synthetic_log10_min": xs.min(), "synthetic_log10_max": xs.max()}])
    return bins, grid

TIMING_OFFSETS = [-30, -20, -10, 0, 10, 20, 30]

def fig3_extremum_timing(c):
    d = _memo(c, "fig3_shape_fidelity"); q = d[d.method.eq(TEST_PREDS_QWEN)].iloc[0]
    return pd.DataFrame([{"extremum": lab, "offset_years": k, "trajectories_%": float(q[f"{col}_offset_{k:+d}_%"])}
                         for lab, col in (("maximum", "peak"), ("minimum", "trough")) for k in TIMING_OFFSETS])

def fig3_maximum_parity(c):
    """Panel d points: maximum of each test trajectory with a defined extremum, AR6 and synthetic."""
    d = _memo(c, "fig3_peak_height"); d = d[d.timing_eligible.eq(True)]
    return pd.DataFrame({"ar6_maximum": d.ar6_peak_height.to_numpy(float), "synthetic_maximum": d.synthetic_peak_height.to_numpy(float)})

def fig3_maximum_statistics(c):
    d = _memo(c, "fig3_shape_fidelity"); q = d[d.method.eq(TEST_PREDS_QWEN)].iloc[0]
    return pd.DataFrame([{"quantity": "maximum_median_relative_error", "value": float(q.peak_value_rel_err_median)},
                         {"quantity": "maximum_r2", "value": float(q.peak_height_r2)}])

GROUP_LABELS = {"ALL test": "All test", "Lower coverage: dimension x IAM family": "Lower coverage",
                "Higher coverage: dimension x IAM family": "Higher coverage",
                "not scenario seen in train (other family)": "Unseen", "scenario seen in train (other family)": "Seen"}

def _groups(c):
    d = _memo(c, "fig4_group_metrics").set_index("group").loc[list(GROUP_LABELS)].rename(index=GROUP_LABELS)
    return d.rename_axis("group").reset_index()

def fig4_composition(c):
    d = _groups(c).set_index("group").n_cases
    assert d["Unseen"] + d["Seen"] == d["All test"] == d["Lower coverage"] + d["Higher coverage"]
    return _groups(c)[["group", "n_cases"]]

def fig4_group_metrics(c):
    return _groups(c)[["group"] + _METRICS4]

def fig4_reference_lines(c):
    nds = _memo(c, "fig4_nearest_desc_scenario"); sc = _memo(c, "fig4_error_scales").set_index("quantity")
    a = nds[nds["pool"].eq("all AR6 runs") & nds["group"].eq("ALL test")].iloc[0]; b = sc.loc["Average other IAM"]
    return pd.DataFrame([{"reference": "Similar scenario, same IAM family", **{m: a[m] for m in _METRICS4}},
                         {"reference": "Same scenario, different IAM family", **{m: b[m] for m in _METRICS4}}])

def fig4_contrast(c):
    return _memo(c, "fig4_contrast_examples")[["panel", "label", "model_family", "region", "variable", "training_scenario_B",
                                               "test_scenario_A", "series", "year", "value"]]

def fig5_heldout(c):
    """2050 indicators as deviations from the median of the other IAM families that ran the
    scenario: percentage points for shares, per cent for total final energy."""
    P = _memo(c, "fig5_heldout_profiles"); C = _memo(c, "fig5_heldout_cells"); rows = []
    for cell in C.itertuples():
        d = P[(P.family == cell.family) & (P.scenario == cell.scenario)]
        for k, _ in HELDOUT_INDICATORS:
            dk = d[d.indicator == k]
            if dk.empty: continue
            oth = dk[dk.series == "other family"].value.to_numpy(float); med = float(np.median(oth))
            dev = (lambda v: 100 * (v / med - 1)) if k == "FE_total" else (lambda v: v - med)
            base = dict(family=cell.family, scenario=cell.scenario, n_other_families=cell.n_other_families, indicator=k,
                        unit="%" if k == "FE_total" else "percentage points")
            rows += [dict(base, series="other family", deviation=dev(v)) for v in oth]
            rows.append(dict(base, series="synthetic", deviation=dev(float(dk[dk.series == "synthetic"].value.iloc[0]))))
            rows.append(dict(base, series="AR6 truth", deviation=dev(float(dk[dk.series == "AR6 truth"].value.iloc[0]))))
    return pd.DataFrame(rows)

MAIN_BUILDERS = {
    "fig2_rag_val_ablation":    ("fig2", fig2_ablation),
    "fig2_km_availability":     ("fig2", fig2_km),
    "fig2_test_four_methods":   ("fig2", fig2_methods),
    "fig3_group_trajectories":  ("fig3", fig3_groups),
    "fig3_error_by_variable":   ("fig3", fig3_error),
    "fig3_extremum_timing":     ("fig3", fig3_extremum_timing),
    "fig3_maximum_parity":      ("fig3", fig3_maximum_parity),
    "fig3_maximum_statistics":  ("fig3", fig3_maximum_statistics),
    "fig4_scenario_coverage":   ("fig4", lambda c: _memo(c, "fig4_scenario_coverage")),
    "fig4_test_composition":    ("fig4", fig4_composition),
    "fig4_group_metrics":       ("fig4", fig4_group_metrics),
    "fig4_reference_lines":     ("fig4", fig4_reference_lines),
    "fig4_contrast_examples":   ("fig4", fig4_contrast),
    "fig5_heldout_deviations":  ("fig5", fig5_heldout),
}
MAIN_TITLES = {   # block -> title row of its table in the workbook
    "fig2_rag_val_ablation":    "Fig. 2a and c. Validation MAE of the zero-shot RAG pipeline for each retrieval and prompt configuration, with bootstrap 95% confidence intervals.",
    "fig2_km_availability":     "Fig. 2b. Validation MAE for each number of same-scenario neighbours, scored on the cases for which such a neighbour is available, with bootstrap 95% confidence intervals.",
    "fig2_test_four_methods":   "Fig. 2d–g. Test-set MAE, sMAPE, Pearson r and Spearman ρ of the four methods, with bootstrap 95% confidence intervals.",
    "fig3_group_trajectories":  "Fig. 3a. Mean across test cases of each group of primary energy sources and each final energy sector in each decade, for AR6 and synthetic trajectories (EJ yr⁻¹).",
    "fig3_error_by_variable":   "Fig. 3b. MAE and sMAPE of the finetuned model for each variable, with bootstrap 95% confidence intervals.",
    "fig3_extremum_timing":     "Fig. 3c. Share of trajectories at each offset between the synthetic and the AR6 extremum year. The offsets of −30 and +30 years include larger offsets.",
    "fig3_maximum_parity":      "Fig. 3d. Maximum of each test trajectory with a defined extremum, for AR6 and synthetic trajectories (EJ yr⁻¹).",
    "fig3_maximum_statistics":  "Fig. 3d. Numbers printed in the panel.",
    "fig4_scenario_coverage":   "Fig. 4a. Number of scenarios for each design dimension and IAM family, with the coverage status.",
    "fig4_test_composition":    "Fig. 4b. Number of test cases in each group.",
    "fig4_group_metrics":       "Fig. 4c–f. Test-set metrics of the finetuned model for each group.",
    "fig4_reference_lines":     "Fig. 4c–f. Reference levels from AR6 data. The first is the scenario with the most similar description that the same IAM family evaluated in the same region. The second is the output of the other IAM families for the same scenario and region.",
    "fig4_contrast_examples":   "Fig. 4g–j. Training case B, AR6 test case A and synthetic test case A for the four examples that differ in one assumption (EJ yr⁻¹).",
    "fig5_heldout_deviations":  "Fig. 5a–d. Indicators in 2050 for four held-out combinations of IAM family and scenario, given as deviations from the median of the other IAM families that evaluated the scenario.",
}

def figure_layout(blocks, titles):
    """sheet -> [(block, title)] in the order that the plotting script reads."""
    return {sheet: [(b, titles[b]) for b in names] for sheet, names in blocks.items()}

def statistics_layout(layout):
    """Layout of a statistics workbook: each table is titled with its block name."""
    return {sheet: [(b, b) for b, _, _ in blocks] for sheet, blocks in layout.items()}

class _Ctx(dict):
    """Build context: the validation and test truth are loaded on first use."""
    _FILES = {"gt_val": "AR6_R10sd_val10_gt.xlsx", "gt_test": "AR6_R10sd_test_gt.xlsx"}
    def __missing__(self, key):
        self[key] = load_gt(self._FILES[key]); return self[key]

def build_workbook(out, builders, layout, want=None, label="main"):
    """Rebuild the requested blocks (all if want is None) and write `out` with
    one sheet per figure; blocks not rebuilt are kept from the existing file."""
    if not (OUTS.is_dir() and SPLIT.is_dir()):
        raise SystemExit(f"Input data not found under {DATA_ROOT}: expected outputs/ and make_datasets/output/split/ in a "
                         "folder above this script. Set IAM_LLM_DATA_ROOT to the folder that holds them.")
    if want:
        known = {k for k, _ in builders.values()}
        bad = set(want) - known
        if bad: raise SystemExit(f"unknown --sheets key(s): {sorted(bad)}; known: {sorted(known)}")
    old = {}
    if want and out.exists():                       # partial rebuild: the other blocks are kept as they are
        from make_main_figs import read_blocks
        old = read_blocks(out, {sheet: [b for b, _ in blocks] for sheet, blocks in layout.items()})
    todo = {name: (key, fn) for name, (key, fn) in builders.items() if want is None or key in want}
    ctx = _Ctx()
    frames = {}
    for name, (key, fn) in builders.items():
        if name not in todo:
            if name in old: frames[name] = old[name]
            else: print(f"[{label}] WARNING: {name} not rebuilt and not yet in {out.name}; its block is left out until built")
            continue
        print(f"[{label}] building {name} ..."); frames[name] = fn(ctx)
    for name in old:
        if name not in frames: print(f"[{label}] dropping obsolete block {name}")
    laid = {b for blocks in layout.values() for b, _ in blocks}
    assert laid == set(builders), (laid ^ set(builders))
    tmp = out.with_suffix(".tmp.xlsx")
    with pd.ExcelWriter(tmp) as xw:
        for sheet_name, blocks in layout.items():
            row = 0
            for name, title in blocks:
                if name not in frames: continue
                df = frames[name]
                pd.DataFrame([[title]]).to_excel(xw, sheet_name=sheet_name, startrow=row, header=False, index=False)
                df.to_excel(xw, sheet_name=sheet_name, startrow=row + 1, index=False)
                row += len(df) + 3
    import time
    for attempt in range(10):           # retry while the file is briefly locked
        try: os.replace(tmp, out); break
        except OSError:
            if attempt == 9: raise
            time.sleep(1.0)
    print(f"[{label}] saved", out)

def _parse(builders, desc):
    import argparse
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--sheets", default=None, help="comma list of builder keys (" + ",".join(sorted({k for k, _ in builders.values()})) +
                    ") to rebuild; others are kept from the existing workbook. Omit for a full rebuild.")
    ap.add_argument("--statistics", action="store_true",
                    help="write the complete statistics workbook instead of the figure workbook (not Source Data: it holds AR6 trajectories)")
    return ap.parse_args()

MAIN_XLSX = _HERE / "main_figure_data.xlsx"
STAT_XLSX = _HERE / "main_statistics.xlsx"

if __name__ == "__main__":
    args = _parse(MAIN_BUILDERS, "build main_figure_data.xlsx (one sheet per main-text figure, only the data drawn)")
    want = set(args.sheets.split(",")) if args.sheets else None
    if args.statistics: build_workbook(STAT_XLSX, STAT_BUILDERS, statistics_layout(STAT_LAYOUT), want, label="statistics")
    else:
        from make_main_figs import MAIN_BLOCKS
        build_workbook(MAIN_XLSX, MAIN_BUILDERS, figure_layout(MAIN_BLOCKS, MAIN_TITLES), want, label="main")
