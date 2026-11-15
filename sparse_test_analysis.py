"""
sparse_test_analysis.py
───────────────────────
Test-set accuracy of the finetuned Qwen3.5-9B by coverage group and by seen or unseen scenario.

Groups test cases two ways using the policy-coverage sparsity report
(policy_coverage_analysis/coverage/report/policy_coverage.xlsx, sheet "Gaps",
Status == "Lower coverage" - the lower-half rule defined there):
  1. lower-coverage dimension x family cells  (a case counts if ANY of its
                                                pairs has lower coverage)
  2. lower-coverage axis x family cells       (same rule at axis level)
plus the seen and unseen scenarios: test cases whose SCENARIO NAME appears in
train70 (necessarily under other model families) and names absent from it.

Metrics per group: MAE, sMAPE, Pearson, Spearman - per-trajectory scores
(the project-wide convention: MAE/sMAPE averaged over trajectories; Pearson/
Spearman averaged over trajectories where both series are non-constant,
n_valid reported), matching score_preds.py / rag_core.

Usage (from the project root):  python3 sparse_test_analysis.py
Output: outputs/sparse_analysis_9b_test.xlsx
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve().parent
SPLIT = _HERE / "make_datasets" / "output" / "split"
PRED = _HERE / "outputs" / "finetune_preds" / \
    "ft_test_qwen3.5-9b_lr0.0005_r16_bf16_best_vllm_20260830_2009.csv"
GAPS = _HERE / "policy_coverage_analysis" / "coverage" / "report" / "policy_coverage.xlsx"
LABELS = _HERE / "policy_coverage_analysis" / "coverage" / "output" / "scenario_labels.csv"
OUT = _HERE / "outputs" / "sparse_analysis_9b_test.xlsx"

YCOLS = [str(y) for y in range(2010, 2101, 10)]

DIM_OF_KEY = {
    "stated_quantified_mitigation_constraint": "Mitigation constraint",
    "stated_non_ssp2_future_socioeconomic_driver": "Alternative socioeconomic pathway",
    "stated_supply_side_technology_setting": "New technology assumption",
    "stated_demand_side_technology_setting": "New technology assumption",
    "stated_side_unspecified_technology_setting": "New technology assumption",
    "stated_new_climate_policy": "New climate policy",
}
AXIS_OF_KEY = {
    "stated_quantified_mitigation_constraint":
        "Quantified mitigation constraint stated",
    "stated_non_ssp2_future_socioeconomic_driver":
        "Non-SSP2 future socioeconomic driver stated",
    "stated_supply_side_technology_setting":
        "Supply-side technology setting stated",
    "stated_demand_side_technology_setting":
        "Demand-side technology setting stated",
    "stated_side_unspecified_technology_setting":
        "Side-unspecified technology setting stated",
    "stated_new_climate_policy":
        "New climate policy stated (beyond what is already in force)",
}
OTHER = "None of the four stated"
LOWER_COVERAGE = "Lower coverage"


def load_sparse_sets():
    raw = pd.read_excel(GAPS, sheet_name="Gaps", header=None)
    hdr = raw.index[raw.iloc[:, 0].eq("Sheet")][0]
    df = raw.iloc[hdr + 1:].copy()
    df.columns = raw.iloc[hdr]
    df = df[df.Status.eq(LOWER_COVERAGE)]
    dxf = set(map(tuple, df.loc[df.Sheet.eq("Dimension x family"),
                                ["Class", "Model family"]].values))
    axf = set(map(tuple, df.loc[df.Sheet.eq("Axis x family"),
                                ["Class", "Model family"]].values))
    return dxf, axf


def scenario_classes():
    lab = pd.read_csv(LABELS)
    dims, axes = {}, {}
    for _, r in lab.iterrows():
        d = {DIM_OF_KEY[k] for k in DIM_OF_KEY if bool(r[k])}
        a = {AXIS_OF_KEY[k] for k in AXIS_OF_KEY if bool(r[k])}
        dims[r["scenario"]] = d if d else {OTHER}
        axes[r["scenario"]] = a if a else {OTHER}
    return dims, axes


def traj_table(gt, pred):
    """Per-trajectory mae / smape / pearson / spearman (missing pred -> 0)."""
    m = gt[["id", "model_family", "scenario", "variable"] + YCOLS].merge(
        pred[["id", "variable"] + YCOLS], on=["id", "variable"],
        how="left", suffixes=("_t", "_p"), validate="one_to_one")
    yt = m[[f"{y}_t" for y in YCOLS]].to_numpy(float)
    yp = m[[f"{y}_p" for y in YCOLS]].to_numpy(float)
    rows = []
    for i in range(len(m)):
        mask = np.isfinite(yt[i])
        if not mask.any():
            continue
        t = yt[i][mask]
        p = np.where(np.isfinite(yp[i][mask]), yp[i][mask], 0.0)
        denom = (np.abs(t) + np.abs(p)) / 2.0
        def _corr(fn):
            if t.size < 3 or np.ptp(t) == 0 or np.ptp(p) == 0:
                return np.nan
            with np.errstate(all="ignore"):
                return float(fn(t, p)[0])
        rows.append((m["id"].iloc[i], m["model_family"].iloc[i],
                     m["scenario"].iloc[i],
                     float(np.mean(np.abs(t - p))),
                     float(np.mean(np.abs(t - p) / np.maximum(denom, 1e-12)) * 100.0),
                     _corr(stats.pearsonr), _corr(stats.spearmanr)))
    return pd.DataFrame(rows, columns=["id", "model_family", "scenario",
                                       "mae", "smape_%", "pearson", "spearman"])


def group_stats(tt, case_mask_ids, name, complement=None):
    """4-metric summary for the cases in case_mask_ids and their complement."""
    out = []
    for label, part in ((name, tt[tt["id"].isin(case_mask_ids)]),
                        (complement or f"not {name}",
                         tt[~tt["id"].isin(case_mask_ids)])):
        out.append({
            "group": label,
            "n_cases": part["id"].nunique(), "n_trajectories": len(part),
            "mae": round(float(part.mae.mean()), 4) if len(part) else np.nan,
            "smape_%": round(float(part["smape_%"].mean()), 4) if len(part) else np.nan,
            "pearson": round(float(part.pearson.mean()), 4) if len(part) else np.nan,
            "n_valid_pearson": int(part.pearson.notna().sum()),
            "spearman": round(float(part.spearman.mean()), 4) if len(part) else np.nan,
            "n_valid_spearman": int(part.spearman.notna().sum()),
        })
    return out


def main():
    gt = pd.read_excel(SPLIT / "AR6_R10sd_test_gt.xlsx")
    gt.columns = [str(c) for c in gt.columns]
    pred = pd.read_csv(PRED)
    pred.columns = [str(c) for c in pred.columns]

    tt = traj_table(gt, pred)
    cases = tt.drop_duplicates("id")[["id", "model_family", "scenario"]]
    print(f"{cases.shape[0]} test cases, {len(tt)} trajectories")

    sparse_dxf, sparse_axf = load_sparse_sets()
    sdims, saxes = scenario_classes()
    print(f"lower-coverage dim x family cells: {len(sparse_dxf)}")
    print(f"lower-coverage axis x family cells: {len(sparse_axf)}")

    cd = cases["scenario"].map(lambda s: sdims.get(s, {OTHER}))
    ca = cases["scenario"].map(lambda s: saxes.get(s, {OTHER}))
    ids_dxf = set(cases.loc[[any((dd, f) in sparse_dxf for dd in d)
                             for d, f in zip(cd, cases["model_family"])], "id"])
    ids_axf = set(cases.loc[[any((aa, f) in sparse_axf for aa in a)
                             for a, f in zip(ca, cases["model_family"])], "id"])

    train = pd.read_excel(SPLIT / "AR6_R10sd_train70_ground_truth.xlsx")
    train_scens = set(train["scenario"].astype(str))
    same_fam = set(zip(train["scenario"].astype(str),
                       train["model_family"].astype(str)))
    ids_seen = set(cases.loc[cases["scenario"].astype(str).isin(train_scens), "id"])
    leak = {(s, f) for s, f in zip(cases["scenario"].astype(str),
                                   cases["model_family"].astype(str))} & same_fam
    print(f"scenario+family pairs shared with train (should be 0): {len(leak)}")

    rows = group_stats(tt, set(cases["id"]), "ALL test")[:1]
    rows[0]["group"] = "ALL test"
    rows += group_stats(tt, ids_dxf,
                        "Lower coverage: dimension x IAM family",
                        "Higher coverage: dimension x IAM family")
    rows += group_stats(tt, ids_axf,
                        "Lower coverage: axis x IAM family",
                        "Higher coverage: axis x IAM family")
    rows += group_stats(tt, ids_seen, "scenario seen in train (other family)")
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    with pd.ExcelWriter(OUT) as xw:
        summary.to_excel(xw, sheet_name="summary", index=False)
        pd.DataFrame(sorted(sparse_dxf), columns=["dimension", "model_family"]
                     ).to_excel(xw, sheet_name="lower_cov_dim_x_family", index=False)
        pd.DataFrame(sorted(sparse_axf), columns=["axis", "model_family"]
                     ).to_excel(xw, sheet_name="lower_cov_axis_x_family", index=False)
        case_tab = (tt.groupby(["id", "model_family", "scenario"])
                    .agg(mae=("mae", "mean"), smape=("smape_%", "mean"),
                         pearson=("pearson", "mean"), spearman=("spearman", "mean"),
                         n_traj=("mae", "size")).reset_index().round(4))
        case_tab["lower_coverage_dim_x_family"] = case_tab["id"].isin(ids_dxf)
        case_tab["lower_coverage_axis_x_family"] = case_tab["id"].isin(ids_axf)
        case_tab["scenario_seen_in_train"] = case_tab["id"].isin(ids_seen)
        case_tab.to_excel(xw, sheet_name="per_case", index=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
