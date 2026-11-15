#!/usr/bin/env python3
"""Score prediction CSVs against a GT xlsx with rag_core's exact metric code.

Every function below is copied verbatim from rag_core.py (smape,
safe_corr, prepare_arrays_gt_only, compute_metrics_flat, round_year_cols,
norm_str) and the merge/aggregation follows run_evaluation() step for step:

  pred year cols -> numeric -> round(OUTPUT_DECIMALS=2)   [GT never rounded]
  strict outer merge on (id, variable), must be all "both"
  per (id, variable) trajectory: keep GT-finite points, pred NaN -> 0
  headline = mean of the per-trajectory smape_% over all series

The point of vendoring rather than importing is that rag_core does API/client
setup at import time. --validate re-scores a pred CSV whose rag_core eval
report already exists and asserts the numbers match, so the copy is proven
faithful before any new number is trusted.
"""
import argparse
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DECIMALS = 2


# ── verbatim from rag_core.py ────────────────────────────────────────────────
def norm_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def find_year_cols(df):
    cols = []
    for c in df.columns:
        cs = str(c).strip()
        if cs.isdigit():
            y = int(cs)
            if 1900 <= y <= 2200:
                cols.append(cs)
    return sorted(cols, key=lambda s: int(s))


def smape(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(denom, eps)) * 100.0)


def safe_corr(x, y, method="pearson"):
    """x = GT, y = PRED. n<3 / constant GT / constant PRED -> nan (scipy's own
    convention; dropped from the mean). ptp==0 is scipy's exact-equality test.
    Same rule as rag_core.safe_corr; returns (corr, n)."""
    x2 = np.asarray(x, dtype=float)
    y2 = np.asarray(y, dtype=float)
    n = x2.size
    if n < 3 or np.ptp(x2) == 0 or np.ptp(y2) == 0:
        return np.nan, n
    with np.errstate(all="ignore"):
        r = (stats.pearsonr(x2, y2) if method == "pearson" else stats.spearmanr(x2, y2))
    return float(r[0]), n


def prepare_arrays_gt_only(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt)
    yt2 = yt[mask]
    yp2 = yp[mask]
    yp2 = np.where(np.isfinite(yp2), yp2, 0.0)
    return yt2, yp2


def compute_metrics_flat(y_true, y_pred):
    yt, yp = prepare_arrays_gt_only(y_true, y_pred)
    n = yt.size
    if n == 0:
        return dict(n_points=0, mae=np.nan, rmse=np.nan,
                    **{"smape_%": np.nan}, pearson=np.nan, spearman=np.nan)
    pr, _ = safe_corr(yt, yp, "pearson")
    sr, _ = safe_corr(yt, yp, "spearman")
    return {
        "n_points": int(n),
        "mae": float(np.mean(np.abs(yp - yt))),
        "rmse": float(np.sqrt(np.mean((yp - yt) ** 2))),
        "smape_%": float(smape(yt, yp)),
        "pearson": float(pr) if np.isfinite(pr) else np.nan,
        "spearman": float(sr) if np.isfinite(sr) else np.nan,
    }


def round_year_cols(df, years=None, nd=OUTPUT_DECIMALS):
    if years is None:
        years = [c for c in df.columns
                 if re.fullmatch(r"(19|20|21)\d\d", str(c).strip())]
    for y in years:
        df[y] = pd.to_numeric(df[y], errors="coerce").round(nd)
    return df


# ── run_evaluation()'s pipeline, headline row only ───────────────────────────
KEY = ["id", "variable"]
BASE = ["id", "region", "model_family", "scenario", "variable", "unit"]


def score(pred_path, gt_path):
    pred = pd.read_csv(pred_path)
    gt = pd.read_excel(gt_path)
    pred.columns = [str(c).strip() for c in pred.columns]
    gt.columns = [str(c).strip() for c in gt.columns]

    for name, d in [("PRED", pred), ("GT", gt)]:
        miss = [c for c in BASE if c not in d.columns]
        if miss:
            raise KeyError(f"{name} missing required columns: {miss}")

    for d in (pred, gt):
        d["id"] = pd.to_numeric(d["id"], errors="coerce").astype("Int64")
        for c in BASE[1:]:
            d[c] = d[c].apply(norm_str)

    years = [y for y in find_year_cols(pred) if y in find_year_cols(gt)]
    keep = BASE + years
    pred2, gt2 = pred[keep].copy(), gt[keep].copy()

    # rag_core: "Partial runs (TEST_LIMIT): evaluate only the ids that were predicted."
    _pred_ids = set(pred2["id"].astype(int).tolist())
    _gt_ids = set(gt2["id"].astype(int).tolist())
    if _pred_ids < _gt_ids:
        print(f"  [NOTE] partial run: {len(_pred_ids)}/{len(_gt_ids)} ids",
              file=sys.stderr)
        gt2 = gt2[gt2["id"].astype(int).isin(_pred_ids)].copy()

    for y in years:
        pred2[y] = pd.to_numeric(pred2[y], errors="coerce")
        gt2[y] = pd.to_numeric(gt2[y], errors="coerce")
    pred2 = round_year_cols(pred2, years)          # GT is deliberately NOT rounded

    for nm, d in [("PRED", pred2), ("GT", gt2)]:
        dup = d.duplicated(subset=KEY).sum()
        if dup:
            raise ValueError(f"{nm} has {dup} duplicate (id, variable) keys")

    merged = pred2.merge(gt2, on=KEY, how="outer",
                         suffixes=("_pred", "_true"), indicator=True)
    if not (merged["_merge"] == "both").all():
        n = int((merged["_merge"] != "both").sum())
        raise ValueError(f"{n} rows did not match on (id, variable)")
    merged = merged.drop(columns=["_merge"])

    rows = merged.apply(
        lambda r: pd.Series(compute_metrics_flat(
            [r[f"{y}_true"] for y in years], [r[f"{y}_pred"] for y in years])),
        axis=1)
    return {
        "n_series": int(len(rows)),
        "mae": float(rows["mae"].mean()),
        "rmse": float(rows["rmse"].mean()),
        "smape_%": float(rows["smape_%"].mean()),
        # correlations are means over the trajectories where they are DEFINED
        # (n>=3, GT not constant, pred not constant -> nan otherwise, dropped),
        # same convention as rag_core / knn / cvae; n_valid_* records the denominator
        "pearson": float(rows["pearson"].mean(skipna=True)),
        "n_valid_pearson": int(rows["pearson"].notna().sum()),
        "spearman": float(rows["spearman"].mean(skipna=True)),
        "n_valid_spearman": int(rows["spearman"].notna().sum()),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--label", nargs="*", default=None)
    ap.add_argument("--no-round", action="store_true",
                    help="skip the rounding of the predictions")
    ap.add_argument("--validate-against", default=None,
                    help="rag_core eval .xlsx whose first pred is re-scored here")
    a = ap.parse_args()
    if a.no_round:
        round_year_cols = lambda df, years=None, nd=None: df
        globals()['round_year_cols'] = round_year_cols

    labels = a.label or [p.split("/")[-1][:40] for p in a.pred]
    out = []
    for lab, p in zip(labels, a.pred):
        m = score(p, a.gt)
        m["label"] = lab
        out.append(m)
        print(f"scored {lab}: MAE {m['mae']:.4f}  sMAPE {m['smape_%']:.4f}%  (n={m['n_series']})",
              file=sys.stderr)

    if a.validate_against:
        ref = pd.read_excel(a.validate_against, sheet_name="overall_timeseries")
        col = next(c for c in ref.columns if str(c).strip() == "smape_%")
        expected = float(ref[col].iloc[0])
        got = out[0]["smape_%"]
        print(f"\nVALIDATION  rag_core report: {expected:.6f}   this script: {got:.6f}"
              f"   |diff| {abs(expected - got):.2e}")
        assert abs(expected - got) < 1e-9, "vendored metric code does NOT match rag_core"
        print("PASS - metric code is faithful to rag_core\n")

    # rows are sorted by MAE, the selection metric
    df = pd.DataFrame(out)[["label", "mae", "smape_%", "rmse",
                            "pearson", "n_valid_pearson", "spearman", "n_valid_spearman",
                            "n_series"]]
    print(df.sort_values("mae").to_string(index=False))
