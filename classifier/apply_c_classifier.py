"""
apply_c_classifier.py
─────────────────────
Apply the trained c-category classifiers (c_category_clf.joblib /
c_group_clf.joblib, produced by train_c_classifier.py on the CURRENT split)
to PREDICTED trajectories from any method (finetune / RAG / cVAE / KNN).

The features are built exactly as in training - 22 variables x 10 years
pivoted wide (NaN where the family does not report the variable) + region
one-hot - except the trajectory values come from the prediction CSV instead
of the ground truth. True labels (category_c / c_group) come from the GT
xlsx of the same split, joined by case id.

Usage (from the project root):
    python3 classifier/apply_c_classifier.py \
        --pred outputs/finetune_preds/ft_test_..._vllm_....csv \
        --gt   make_datasets/output/split/AR6_R10sd_test_gt.xlsx \
        --label 9b_5e-4
Output: classifier/output/<label|pred-stem>_c_classification.xlsx
    summary | per_class_category_c | per_class_c_group |
    confusion_category_c | confusion_c_group | case_predictions

Guard: refuses to run if the joblib files are OLDER than the split's GT file
(a classifier trained on a previous split must not score this split's test).
"""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

_HERE = Path(__file__).resolve().parent
OUT_DIR = _HERE / "output"

# ── constants: verbatim from train_c_classifier.py (keep in sync) ────────────
YCOLS = [str(y) for y in range(2010, 2101, 10)]
PARENTS = {
    "Primary Energy|Coal": ("Primary Energy|Coal|w/ CCS", "Primary Energy|Coal|w/o CCS"),
    "Primary Energy|Gas": ("Primary Energy|Gas|w/ CCS", "Primary Energy|Gas|w/o CCS"),
    "Primary Energy|Oil": ("Primary Energy|Oil|w/ CCS", "Primary Energy|Oil|w/o CCS"),
}
OTHER_SUPPLY = ["Primary Energy|Solar", "Primary Energy|Wind", "Primary Energy|Hydro",
                "Primary Energy|Nuclear", "Primary Energy|Biomass"]
SUPPLY_11 = [c for pair in PARENTS.values() for c in pair] + OTHER_SUPPLY
DEMAND_11 = [
    "Final Energy|Liquids", "Final Energy|Gases",
    "Final Energy|Industry", "Final Energy|Transportation",
    "Final Energy|Residential and Commercial", "Final Energy|Hydrogen",
    "Final Energy|Heat", "Final Energy|Solids",
    "Final Energy|Industry|Electricity", "Final Energy|Transportation|Electricity",
    "Final Energy|Residential and Commercial|Electricity",
]
VARIABLES_22 = SUPPLY_11 + DEMAND_11
R10_REGIONS = [
    "Countries of Latin America and the Caribbean",
    "Countries of South Asia; primarily India",
    "Countries of Sub-Saharan Africa",
    "Countries of centrally-planned Asia; primarily China",
    "Countries of the Middle East; Iran, Iraq, Israel, Saudi Arabia, Qatar, etc.",
    "Eastern and Western Europe (i.e., the EU28)",
    "North America; primarily the United States of America and Canada",
    "Other countries of Asia",
    "Pacific OECD",
    "Reforming Economies of Eastern Europe and the Former Soviet Union; primarily Russia",
]
TRAJ_COLS = [f"{v}|{y}" for v in VARIABLES_22 for y in YCOLS]
REGION_COLS = [f"region={r}" for r in R10_REGIONS]
FEATURE_COLS = TRAJ_COLS + REGION_COLS


def build_features_from_pred(pred: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """pred = long prediction frame (id, region, variable + YCOLS ...);
    meta = one row per id with region (from GT). Same pivot/one-hot as
    training, values from the prediction."""
    pred = pred.copy()
    pred.columns = [str(c) for c in pred.columns]
    for y in YCOLS:
        pred[y] = pd.to_numeric(pred[y], errors="coerce")
    wide = pred.pivot_table(index="id", columns="variable", values=YCOLS,
                            aggfunc="first")
    wide.columns = [f"{var}|{yr}" for yr, var in wide.columns]
    wide = wide.reindex(columns=TRAJ_COLS)          # fixed order, NaN-padded
    wide = wide.loc[meta["id"]]                     # GT case order

    onehot = pd.DataFrame(0.0, index=wide.index, columns=REGION_COLS)
    reg = meta.set_index("id")["region"]
    for r in R10_REGIONS:
        onehot.loc[reg == r, f"region={r}"] = 1.0
    unknown = ~reg.isin(R10_REGIONS)
    if unknown.any():
        raise ValueError(f"unknown regions: {reg[unknown].unique()}")
    return pd.concat([wide, onehot], axis=1)[FEATURE_COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="prediction CSV (long format)")
    ap.add_argument("--gt", required=True, help="GT xlsx of the SAME split (labels + meta)")
    ap.add_argument("--label", default=None, help="output name stem (default: pred file stem)")
    ap.add_argument("--clf-dir", default=str(OUT_DIR),
                    help="dir with c_category_clf.joblib / c_group_clf.joblib")
    args = ap.parse_args()

    clf_dir = Path(args.clf_dir)
    gt_path = Path(args.gt)
    for stem in ("c_category_clf", "c_group_clf"):
        jb = clf_dir / f"{stem}.joblib"
        assert jb.exists(), f"missing {jb} - run train_c_classifier.py first"
        if jb.stat().st_mtime < gt_path.stat().st_mtime:
            raise SystemExit(
                f"REFUSING: {jb.name} is OLDER than {gt_path.name} - the classifier was "
                f"trained on a previous split. Re-run train_c_classifier.py first.")

    gt = pd.read_excel(gt_path)
    gt.columns = [str(c) for c in gt.columns]
    need = {"id", "region", "scenario", "model_family", "category_c", "c_group"}
    missing = need - set(gt.columns)
    assert not missing, f"GT is missing columns: {missing}"
    meta = gt[list(need)].drop_duplicates("id").reset_index(drop=True)

    pred = pd.read_csv(args.pred)
    pred_ids = set(pred["id"].unique())
    gt_ids = set(meta["id"])
    inter = pred_ids & gt_ids
    print(f"cases: pred {len(pred_ids)} | gt {len(gt_ids)} | matched {len(inter)}")
    if len(inter) < 0.9 * len(gt_ids):
        raise SystemExit("REFUSING: <90% id overlap - pred and GT are probably from "
                         "different split versions (see split_regenerated scoring trap).")
    meta = meta[meta["id"].isin(inter)].reset_index(drop=True)

    X = build_features_from_pred(pred[pred["id"].isin(inter)], meta)

    out_rows, case = [], meta.copy()
    reports, cms = {}, {}
    for target, stem in (("category_c", "c_category_clf"), ("c_group", "c_group_clf")):
        bundle = joblib.load(clf_dir / f"{stem}.joblib")
        model = bundle["model"] if isinstance(bundle, dict) else bundle
        feats = bundle.get("feature_cols", FEATURE_COLS) if isinstance(bundle, dict) else FEATURE_COLS
        assert list(feats) == list(FEATURE_COLS), f"{stem}: feature_cols mismatch with this script"
        y_true = case[target]
        y_pred = model.predict(X)
        case[f"pred_{target}"] = y_pred
        labels = sorted(y_true.unique())
        out_rows.append({
            "target": target,
            "n": len(y_true),
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
            "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
        })
        reports[target] = (pd.DataFrame(classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0))
            .T.reset_index().rename(columns={"index": "class"}))
        cms[target] = pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=labels),
            index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
        print(f"[{target}] acc={out_rows[-1]['accuracy']}  macro_f1={out_rows[-1]['macro_f1']}")

    stem = args.label or Path(args.pred).stem
    out_path = OUT_DIR / f"{stem}_c_classification.xlsx"
    with pd.ExcelWriter(out_path) as xw:
        prov = pd.DataFrame(out_rows)
        prov["pred_csv"] = os.path.basename(args.pred)
        prov["gt_xlsx"] = os.path.basename(args.gt)
        prov.to_excel(xw, "summary", index=False)
        reports["category_c"].to_excel(xw, "per_class_category_c", index=False)
        reports["c_group"].to_excel(xw, "per_class_c_group", index=False)
        cms["category_c"].to_excel(xw, "confusion_category_c")
        cms["c_group"].to_excel(xw, "confusion_c_group")
        case.to_excel(xw, "case_predictions", index=False)
    print("saved", out_path)


if __name__ == "__main__":
    main()
