"""
train_c_classifier.py
─────────────────────
Classifiers of the AR6 climate category, trained on the training set in
make_datasets/output/split/. The generative models predict variable
trajectories only, and the category labels are used here.

Two classifiers are trained independently:
  1. category_c : fine AR6 label C1..C8   (8 classes)
  2. c_group    : coarse group c1-c4 / c5-c6 / c7-c8   (3 classes)

Input features per case (region, scenario, model_family):
  - the 22 variable trajectories (11 supply + 11 demand) x 10 years
    (2010..2100), 220 features.  Demand-only cases (GCAM/TIAM) have the
    supply half as NaN, supply-only cases (COFFEE/WITCH) the demand half.
  - the R10 region as a one-hot (10 columns; USE_REGION switches it off).
  model_family is deliberately NOT a feature.

Protocol:
  - ONE model family: RandomForestClassifier, 500 trees,
    class-weight balanced, with a standard hyper-parameter grid
    (max_features x min_samples_leaf).  NaNs are imputed with the constant
    -1.0 inside the pipeline (trees treat it as a distinct "not reported"
    value; -1 cannot collide with real data because the trajectories are
    non-negative).
  - fit on train70, select by macro-F1 on val10.
  - the winning config is refit on train70+val10 and evaluated ONCE on the
    held-out test set.

Outputs (classifier/output/):
  c_category_clf.joblib / c_group_clf.joblib
      dict(model=fitted sklearn estimator, feature_cols=list, target=str,
           classes=list)  - apply with predict() on a frame built by
      build_features() below.
  classifier_results.xlsx
      model_selection | test_metrics | per-class reports | confusion
      matrices | test_predictions

Usage (from the project root):
    python3 classifier/train_c_classifier.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.pipeline import Pipeline

_HERE = Path(__file__).resolve().parent
SPLIT_DIR = _HERE.parent / "make_datasets" / "output" / "split"
OUT_DIR = _HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
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

USE_REGION = True

TRAJ_COLS = [f"{v}|{y}" for v in VARIABLES_22 for y in YCOLS]
REGION_COLS = [f"region={r}" for r in R10_REGIONS] if USE_REGION else []
FEATURE_COLS = TRAJ_COLS + REGION_COLS


# ── feature building ─────────────────────────────────────────────────────────
def build_features(gt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """gt = long ground-truth frame (one row per id x variable) with columns
    id, region, model_family, scenario, category_c, c_group, variable + YCOLS.
    Returns (X, meta): X has one row per case with 220 trajectory features
    (NaN where the family does not report the variable) + 10 region one-hots;
    meta carries id/region/scenario/model_family/category_c/c_group."""
    gt = gt.copy()
    gt.columns = [str(c) for c in gt.columns]
    wide = (gt.pivot_table(index="id", columns="variable", values=YCOLS,
                           aggfunc="first"))
    # pivot gives (year, variable) column tuples -> flatten to "variable|year"
    wide.columns = [f"{var}|{yr}" for yr, var in wide.columns]
    wide = wide.reindex(columns=TRAJ_COLS)          # fixed order, NaN-padded

    meta_cols = ["id", "region", "scenario", "model_family",
                 "category_c", "c_group"]
    meta = (gt[meta_cols].drop_duplicates("id").set_index("id")
            .loc[wide.index])

    if USE_REGION:
        onehot = pd.DataFrame(0.0, index=wide.index, columns=REGION_COLS)
        for r in R10_REGIONS:
            onehot.loc[meta.region == r, f"region={r}"] = 1.0
        unknown = ~meta.region.isin(R10_REGIONS)
        if unknown.any():
            raise ValueError(f"unknown regions: {meta.region[unknown].unique()}")
        X = pd.concat([wide, onehot], axis=1)[FEATURE_COLS]
    else:
        X = wide[FEATURE_COLS]
    return X, meta.reset_index()


def load_split(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt = pd.read_excel(SPLIT_DIR / name)
    gt.columns = [str(c) for c in gt.columns]
    return build_features(gt)


# ── candidates ───────────────────────────────────────────────────────────────
def candidates():
    """(name, param-string, estimator) triples - a standard grid over the
    single chosen model family (RandomForest, 500 trees)."""
    out = []
    for max_features in ("sqrt", 0.3, 0.5, 0.7, 0.9):
        for min_leaf in (1, 3):
            out.append(("rf", f"max_features={max_features},min_leaf={min_leaf}",
                        Pipeline([
                            ("impute", SimpleImputer(strategy="constant",
                                                     fill_value=-1.0)),
                            ("clf", RandomForestClassifier(
                                n_estimators=500, max_features=max_features,
                                min_samples_leaf=min_leaf,
                                class_weight="balanced_subsample",
                                random_state=SEED, n_jobs=-1)),
                        ])))
    return out


# ── data ─────────────────────────────────────────────────────────────────────
print("loading splits ...")
Xtr, mtr = load_split("AR6_R10sd_train70_ground_truth.xlsx")
Xva, mva = load_split("AR6_R10sd_val10_gt.xlsx")
Xte, mte = load_split("AR6_R10sd_test_gt.xlsx")
print(f"train {len(Xtr)} | val {len(Xva)} | test {len(Xte)} cases, "
      f"{len(FEATURE_COLS)} features")

TARGETS = {"category_c": "c_category_clf", "c_group": "c_group_clf"}
sel_rows, test_rows, report_frames, cm_frames = [], [], {}, {}
pred_te = mte.copy()

for target, model_stem in TARGETS.items():
    ytr, yva, yte = mtr[target], mva[target], mte[target]
    print(f"\n===== target: {target} ({ytr.nunique()} classes) =====")

    best = None
    for fam, params, est in candidates():
        est.fit(Xtr, ytr)
        pv = est.predict(Xva)
        acc = accuracy_score(yva, pv)
        f1 = f1_score(yva, pv, average="macro")
        sel_rows.append({"target": target, "model": fam, "params": params,
                         "val_accuracy": round(acc, 4),
                         "val_macro_f1": round(f1, 4)})
        print(f"  {fam:7s} {params:38s} val acc {acc:.3f}  macroF1 {f1:.3f}")
        if best is None or f1 > best[0]:
            best = (f1, fam, params, est)

    f1b, fam, params, _ = best
    print(f"  -> selected {fam} ({params}), val macroF1 {f1b:.3f}; "
          f"refitting on train+val")
    final = [e for f, p, e in candidates() if f == fam and p == params][0]
    Xfull = pd.concat([Xtr, Xva])
    final.fit(Xfull, pd.concat([ytr, yva]))

    pt = final.predict(Xte)
    acc, f1 = accuracy_score(yte, pt), f1_score(yte, pt, average="macro")
    print(f"  TEST accuracy {acc:.3f} | macro-F1 {f1:.3f}")
    test_rows.append({"target": target, "model": fam, "params": params,
                      "n_test_cases": len(yte),
                      "test_accuracy": round(acc, 4),
                      "test_macro_f1": round(f1, 4)})

    labels = sorted(yte.unique())
    rep = pd.DataFrame(classification_report(
        yte, pt, labels=labels, output_dict=True, zero_division=0)).T
    rep.index.name = "class"
    report_frames[target] = rep.round(4).reset_index()
    cm = pd.DataFrame(confusion_matrix(yte, pt, labels=labels),
                      index=[f"true_{l}" for l in labels],
                      columns=[f"pred_{l}" for l in labels])
    cm_frames[target] = cm.reset_index(names="")
    pred_te[f"pred_{target}"] = pt

    joblib.dump({"model": final, "feature_cols": FEATURE_COLS,
                 "target": target, "classes": labels,
                 "selected": f"{fam} ({params})"},
                OUT_DIR / f"{model_stem}.joblib", compress=3)

# consistency: c_group derived from the fine C1..C8 prediction
def to_group(c):
    c = str(c).lower()
    return ("c1-c4" if c in {"c1", "c2", "c3", "c4"} else
            "c5-c6" if c in {"c5", "c6"} else "c7-c8")

pred_te["pred_c_group_from_category"] = pred_te["pred_category_c"].map(to_group)
derived_acc = accuracy_score(pred_te.c_group,
                             pred_te.pred_c_group_from_category)
derived_f1 = f1_score(pred_te.c_group, pred_te.pred_c_group_from_category,
                      average="macro")
agree = (pred_te.pred_c_group ==
         pred_te.pred_c_group_from_category).mean()
test_rows.append({"target": "c_group (derived from category_c preds)",
                  "model": "-", "params": "-", "n_test_cases": len(pred_te),
                  "test_accuracy": round(derived_acc, 4),
                  "test_macro_f1": round(derived_f1, 4)})
print(f"\nc_group via category_c classifier: acc {derived_acc:.3f} "
      f"macroF1 {derived_f1:.3f} | agreement with direct clf {agree:.3f}")

# ── results workbook ─────────────────────────────────────────────────────────
res_path = OUT_DIR / "classifier_results.xlsx"
with pd.ExcelWriter(res_path, engine="openpyxl") as w:
    pd.DataFrame(sel_rows).to_excel(w, sheet_name="model_selection", index=False)
    pd.DataFrame(test_rows).to_excel(w, sheet_name="test_metrics", index=False)
    report_frames["category_c"].to_excel(w, sheet_name="report_category_c", index=False)
    report_frames["c_group"].to_excel(w, sheet_name="report_c_group", index=False)
    cm_frames["category_c"].to_excel(w, sheet_name="cm_category_c", index=False)
    cm_frames["c_group"].to_excel(w, sheet_name="cm_c_group", index=False)
    pred_te.to_excel(w, sheet_name="test_predictions", index=False)
print(f"saved -> {res_path}")
print(f"models -> {OUT_DIR / 'c_category_clf.joblib'}, "
      f"{OUT_DIR / 'c_group_clf.joblib'}")
