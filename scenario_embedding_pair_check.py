"""
scenario_embedding_pair_check.py
────────────────────────────────
How similar are the DATA of the two most embedding-similar scenarios?

For every scenario, find its single nearest OTHER scenario by cosine
similarity of the scenario-DESCRIPTION embedding (text-embedding-3-large,
served from the shared cache _cache/cvae_condition_embeddings.pkl - no API
calls are made; the script refuses if a description is missing from the
cache). Then, for every (region, model_family) where BOTH scenarios report
data (train70+val10+test GT combined), score the neighbour's trajectories
as a "prediction" of the scenario's own trajectories, with the project's
per-trajectory conventions: MAE, sMAPE, Pearson, Spearman.

This is the data-level ceiling of "copy the most description-similar
scenario": if even the closest pairs differ a lot, retrieval can only help
as evidence, not as a copy source.

Usage (from the project root):  python3 scenario_embedding_pair_check.py
Output: outputs/scenario_embedding_pairs.xlsx
    summary   | overall numbers (per-trajectory pooled + per-pair means)
    pairs     | one row per scenario: its nearest neighbour, cosine sim,
                n shared (region,family) contexts, n trajectories, 4 metrics
    per_traj  : not saved (large); pooled into summary
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve().parent
SPLIT = _HERE / "make_datasets" / "output" / "split"
CACHE = _HERE / "_cache" / "cvae_condition_embeddings.pkl"
OUT = _HERE / "outputs" / "scenario_embedding_pairs.xlsx"
YEARS = [str(y) for y in range(2010, 2101, 10)]


def load_desc_dict():
    """scenario -> description text, exactly as knn_regression.load_desc_dict."""
    mixed = sorted((_HERE / "make_datasets" / "output").glob("AR6_R10_sd_mixed_*.xlsx"),
                   key=lambda p: p.stat().st_mtime)
    path = mixed[-1]                     # newest mixed workbook, as in knn_regression
    xl = pd.ExcelFile(path)
    sheet = next(s for s in xl.sheet_names
                 if "scenario_description" in
                 [str(c).strip() for c in
                  pd.read_excel(xl, sheet_name=s, nrows=1).columns])
    df = pd.read_excel(xl, sheet_name=sheet, usecols=["scenario", "scenario_description"])
    out = {}
    for s, d in zip(df["scenario"], df["scenario_description"]):
        s, d = str(s).strip(), "" if pd.isna(d) else str(d).strip()
        if s and d and s not in out:
            out[s] = d
    print("desc source:", Path(path).name)
    return out


def main():
    gt = pd.concat([pd.read_excel(SPLIT / f) for f in
                    ("AR6_R10sd_train70_ground_truth.xlsx",
                     "AR6_R10sd_val10_gt.xlsx", "AR6_R10sd_test_gt.xlsx")],
                   ignore_index=True)
    gt.columns = [str(c).strip() for c in gt.columns]
    gt = gt.drop_duplicates(subset=["region", "model_family", "scenario", "variable"])
    scenarios = sorted(gt["scenario"].astype(str).unique())
    print(f"{len(scenarios)} scenarios | {len(gt)} GT rows")

    desc = load_desc_dict()
    texts = {s: desc.get(s, s) for s in scenarios}   # fallback = name, as in knn
    cache = pickle.load(open(CACHE, "rb"))
    missing = [s for s, t in texts.items() if t not in cache]
    assert not missing, f"{len(missing)} descriptions not in embedding cache: {missing[:3]}"
    E = np.array([cache[texts[s]] for s in scenarios], float)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    sim = E @ E.T
    np.fill_diagonal(sim, -np.inf)
    nn_idx = sim.argmax(axis=1)

    # long GT indexed for fast pair lookup
    gt["scenario"] = gt["scenario"].astype(str)
    key = ["region", "model_family", "variable"]
    ymat = {s: g.set_index(key)[YEARS].astype(float)
            for s, g in gt.groupby("scenario")}

    def traj_metrics(a, b):
        """b's trajectory scored as prediction of a's (both 1d arrays)."""
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() == 0:
            return None
        t, p = a[mask], b[mask]
        denom = (np.abs(t) + np.abs(p)) / 2.0
        out = [float(np.mean(np.abs(t - p))),
               float(np.mean(np.abs(t - p) / np.maximum(denom, 1e-12)) * 100.0)]
        for fn in (stats.pearsonr, stats.spearmanr):
            if t.size < 3 or np.ptp(t) == 0 or np.ptp(p) == 0:
                out.append(np.nan)
            else:
                with np.errstate(all="ignore"):
                    out.append(float(fn(t, p)[0]))
        return out

    pair_rows, pooled = [], []
    for i, s in enumerate(scenarios):
        n = scenarios[nn_idx[i]]
        A, B = ymat[s], ymat[n]
        shared = A.index.intersection(B.index)
        mets = []
        for k in shared:
            m = traj_metrics(A.loc[k].to_numpy(), B.loc[k].to_numpy())
            if m is not None:
                mets.append(m)
                pooled.append(m)
        mets = np.array(mets, float) if mets else np.empty((0, 4))
        with np.errstate(all="ignore"):
            mm = np.nanmean(mets, axis=0) if len(mets) else [np.nan] * 4
        pair_rows.append({
            "scenario": s, "nearest_by_embedding": n,
            "cosine_sim": round(float(sim[i, nn_idx[i]]), 4),
            "n_shared_contexts": len({(r, f) for r, f, _ in shared}),
            "n_trajectories": len(mets),
            "mae": round(float(mm[0]), 4), "smape_%": round(float(mm[1]), 4),
            "pearson": round(float(mm[2]), 4), "spearman": round(float(mm[3]), 4),
        })

    pairs = pd.DataFrame(pair_rows)
    P = np.array(pooled, float)
    with np.errstate(all="ignore"):
        summary = pd.DataFrame([
            {"aggregation": "pooled over all trajectories",
             "n": len(P),
             "mae": round(float(np.nanmean(P[:, 0])), 4),
             "smape_%": round(float(np.nanmean(P[:, 1])), 4),
             "pearson": round(float(np.nanmean(P[:, 2])), 4),
             "n_valid_pearson": int(np.isfinite(P[:, 2]).sum()),
             "spearman": round(float(np.nanmean(P[:, 3])), 4),
             "n_valid_spearman": int(np.isfinite(P[:, 3]).sum())},
            {"aggregation": "mean over scenario pairs",
             "n": int(pairs.n_trajectories.gt(0).sum()),
             "mae": round(float(pairs.mae.mean()), 4),
             "smape_%": round(float(pairs["smape_%"].mean()), 4),
             "pearson": round(float(pairs.pearson.mean()), 4),
             "n_valid_pearson": int(pairs.pearson.notna().sum()),
             "spearman": round(float(pairs.spearman.mean()), 4),
             "n_valid_spearman": int(pairs.spearman.notna().sum())},
        ])
    print(summary.to_string(index=False))
    print(f"pairs with zero shared contexts: {int(pairs.n_trajectories.eq(0).sum())}")

    with pd.ExcelWriter(OUT) as xw:
        summary.to_excel(xw, sheet_name="summary", index=False)
        pairs.sort_values("cosine_sim", ascending=False).to_excel(
            xw, sheet_name="pairs", index=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
