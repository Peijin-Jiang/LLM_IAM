"""
knn_regression.py
─────────────────
KNN regression baseline for the 22 energy variables.

Condition modes (each component is embedded separately):
  1 = model_family and scenario names                 (names)
  2 = model_fingerprint and scenario_description      (content)

Method:
  - each component text is embedded (text-embedding-3-large, cache shared with
    the cVAE baseline)
  - neighbours are taken from the same region, and the similarity combines the
    two components:
        sim = alpha_model * cos_model + (1 - alpha_model) * cos_scenario
  - for a query case, the k most similar training cases of its region are found
  - the prediction is the uniform or similarity-weighted mean of the
    neighbours' trajectories

The model predicts trajectories only. The climate category columns of the
ground-truth files are metadata.

Hyperparameters are selected by validation MAE over the grid
    k in {1..5} x weighting in {uniform, similarity} x alpha_model in {0, 0.25, 0.5, 0.75, 1}
The neighbour pool is the training set. The selected setting is evaluated once
on the test set.

Case-level bootstrap (paired over case ids, B=1000, seed 42):
  - validation: every grid cell gets a 95% CI and a paired difference from the
    selected cell -> bootstrap_val.xlsx, with the CI columns merged into
    val_sweep.xlsx
  - test: the test predictions of the selected setting -> bootstrap_test.xlsx,
    with CI keys added to test_metrics.json
  - the validation predictions of the selected setting are written to
    AR6_R10sd_val10_pred_knn.csv
  Selection uses the validation MAE point estimate.

Usage (from the project root):
    python3 knn/knn_regression.py --condition 1
    python3 knn/knn_regression.py --condition 2

Env overrides:
    IAM_LLM_FP_SHEET          fingerprint sheet name (default "fingerprint")
    IAM_LLM_FINGERPRINT_PATH  fingerprint xlsx
    KNN_TEST_PATH      test ground-truth xlsx (default: the full test set)
    KNN_K_GRID         comma-separated k values        (default "1,2,3,4,5")
    KNN_ALPHA_GRID     comma-separated model-channel weights
                       (default "0.0,0.25,0.5,0.75,1.0")
    KNN_WEIGHTINGS     comma-separated weightings (default "uniform,similarity")
    KNN_BOOT_B         bootstrap replicates (default 1000; 0 disables)
"""

import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# ============================================================
# PATHS & CONFIG
# ============================================================
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_MD_OUT = _ROOT / "make_datasets" / "output"
_SPLIT = _MD_OUT / "split"
_CACHE = _ROOT / "_cache"
_CACHE.mkdir(exist_ok=True)

TRAIN_PATH = str(_SPLIT / "AR6_R10sd_train70_ground_truth.xlsx")
VAL_PATH   = str(_SPLIT / "AR6_R10sd_val10_gt.xlsx")
TEST_PATH  = os.environ.get("KNN_TEST_PATH", str(_SPLIT / "AR6_R10sd_test_gt.xlsx"))
# scenario-description source: the newest MIXED dataset
_mixed = sorted(_MD_OUT.glob("AR6_R10_sd_mixed_*.xlsx"), key=lambda p: p.stat().st_mtime)
if not _mixed:
    raise FileNotFoundError(f"no AR6_R10_sd_mixed_*.xlsx under {_MD_OUT}")
FULL_DATA_PATH = str(_mixed[-1])
# Default fingerprint: comparison-only run (same default as rag_core)
FINGERPRINT_PATH = os.environ.get(
    "IAM_LLM_FINGERPRINT_PATH",
    str(_ROOT / "outputs" / "fingerprint_from_papers" /
        "fingerprint_from_papers_comp_evidence_gpt-5.6_20260828_0525_struct.xlsx"))
FP_SHEET = os.environ.get("IAM_LLM_FP_SHEET", "fingerprint")

EMBED_MODEL = "text-embedding-3-large"
EMB_CACHE_FILE = _CACHE / "cvae_condition_embeddings.pkl"   # shared with the cVAE baseline

K_GRID = [int(x) for x in os.environ.get("KNN_K_GRID", "1,2,3,4,5").split(",") if x.strip()]
WEIGHTINGS = [w.strip() for w in
              os.environ.get("KNN_WEIGHTINGS", "uniform,similarity").split(",") if w.strip()]
ALPHA_GRID = [float(x) for x in
              os.environ.get("KNN_ALPHA_GRID", "0.0,0.25,0.5,0.75,1.0").split(",")
              if x.strip()]

BOOT_B = int(os.environ.get("KNN_BOOT_B", "1000"))
BOOT_SEED = 42
BOOT_METRICS = ("mae", "rmse", "smape_%", "pearson", "spearman")

OUTPUT_DECIMALS = 2

SUPPLY_11 = [   # fossil totals (parents) removed from the pipeline
    "Primary Energy|Coal|w/ CCS", "Primary Energy|Coal|w/o CCS",
    "Primary Energy|Gas|w/ CCS", "Primary Energy|Gas|w/o CCS",
    "Primary Energy|Oil|w/ CCS", "Primary Energy|Oil|w/o CCS",
    "Primary Energy|Solar", "Primary Energy|Wind", "Primary Energy|Hydro",
    "Primary Energy|Nuclear", "Primary Energy|Biomass",
]
DEMAND_11 = [   # Final Energy|Electricity excluded from the dataset
    "Final Energy|Liquids", "Final Energy|Gases",
    "Final Energy|Industry", "Final Energy|Transportation",
    "Final Energy|Residential and Commercial", "Final Energy|Hydrogen",
    "Final Energy|Heat", "Final Energy|Solids",
    "Final Energy|Industry|Electricity", "Final Energy|Transportation|Electricity",
    "Final Energy|Residential and Commercial|Electricity",
]
VARS_22 = SUPPLY_11 + DEMAND_11   # same canonical order as rag_core (VARSET=sd22)

# ── family variable scopes (mixed dataset: 5 both-sides + 2 supply-only + 2 demand-only) ──
CORE_FAMILIES = ["AIM/CGE", "GEM-E3", "IMAGE", "MESSAGEix-GLOBIOM", "REMIND"]
SUPPLY_ONLY_FAMILIES = ["COFFEE", "WITCH"]      # supply data only
DEMAND_ONLY_FAMILIES = ["GCAM", "TIAM"]         # demand data only


def vars_for_family(fam):
    fam = str(fam).strip()
    if fam in SUPPLY_ONLY_FAMILIES:
        return list(SUPPLY_11)
    if fam in DEMAND_ONLY_FAMILIES:
        return list(DEMAND_11)
    return list(VARS_22)


def family_group(fam):
    fam = str(fam).strip()
    if fam in SUPPLY_ONLY_FAMILIES:
        return "sup"
    if fam in DEMAND_ONLY_FAMILIES:
        return "dem"
    return "core"


# neighbor pool per query group (same grouping rule as rag_core STEP2; own
# family INCLUDED here because same-family neighbors are legitimate for
# KNN/cVAE-style retrieval):
#   core query -> the 5 core families
#   supply-only query -> COFFEE+WITCH + the core 5
#   demand-only query -> GCAM+TIAM + the core 5
# Every family in a query's pool carries ALL of the query's scoped variables.
POOL_FAMILIES = {
    "core": set(CORE_FAMILIES),
    "sup": set(CORE_FAMILIES) | set(SUPPLY_ONLY_FAMILIES),
    "dem": set(CORE_FAMILIES) | set(DEMAND_ONLY_FAMILIES),
}

YEARS = [str(y) for y in range(2010, 2110, 10)]


def load_api_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    for env_file in (_ROOT / ".env", _ROOT.parent / ".env"):
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("OPENAI_API_KEY") and "=" in line:
                    v = line.partition("=")[2].strip().strip('"').strip("'")
                    if v:
                        return v
    raise ValueError("No OpenAI API key found (environment variable or a .env file in the project root).")


# ============================================================
# DATA LOADING (same conventions as cvae/train_cvae.py)
# ============================================================
def norm_year_cols(df):
    return df.rename(columns={int(y): y for y in YEARS if int(y) in df.columns})


def read_model_fingerprints():
    df = pd.read_excel(FINGERPRINT_PATH, sheet_name=FP_SHEET)
    df.columns = [str(c).strip() for c in df.columns]
    fam_col = next(c for c in df.columns if c.lower().replace(" ", "_") == "model_family")
    pref_col = next(c for c in df.columns if "preference" in c.lower())
    label = "Energy System Preference"
    out = {}
    for _, row in df.iterrows():
        out[str(row[fam_col]).strip()] = (
            f"{label}: {row[pref_col]}"
        )
    return out


def load_desc_dict():
    xl = pd.ExcelFile(FULL_DATA_PATH)
    sheet = next(s for s in xl.sheet_names
                 if "scenario" in [str(c).strip() for c in
                                   pd.read_excel(xl, sheet_name=s, nrows=1).columns])
    df = pd.read_excel(xl, sheet_name=sheet, usecols=["scenario", "scenario_description"])
    out = {}
    for s, d in zip(df["scenario"], df["scenario_description"]):
        s, d = str(s).strip(), "" if pd.isna(d) else str(d).strip()
        if s and d and s not in out:
            out[s] = d
    return out


def read_cases(path, fingerprints, desc_dict):
    df = norm_year_cols(pd.read_excel(path))
    df.columns = [str(c).strip() for c in df.columns]
    var_rank = {v: i for i, v in enumerate(VARS_22)}

    cols_l = {str(c).strip().lower(): c for c in df.columns}
    id_col = cols_l.get("id")          # present in the eval-format gt files
    cases = []
    for (region, scenario, family), g in df.groupby(["region", "scenario", "model_family"], sort=True):
        g = g.copy()
        g["_rank"] = g["variable"].map(var_rank)
        g = g.dropna(subset=["_rank"]).sort_values("_rank")
        expected = vars_for_family(family)
        if len(g) != len(expected):
            raise ValueError(f"Case {region}|{scenario}|{family} has {len(g)} vars "
                             f"(expected {len(expected)} for this family's scope)")
        # full 22-var matrix; out-of-scope rows stay NaN (masked downstream)
        data = np.full((len(VARS_22), len(YEARS)), np.nan)
        data[g["_rank"].astype(int).to_numpy()] = g[YEARS].astype(float).to_numpy()
        desc = desc_dict.get(str(scenario).strip(), "")
        fp = fingerprints.get(str(family).strip(), "")
        cases.append({
            "case_id": (int(g[id_col].iloc[0]) if id_col else None),
            "region": str(region),
            "scenario": str(scenario),
            "model_family": str(family),
            # separate channels (region is a hard filter, not a similarity channel)
            "model_text": str(family),
            "fp_text": fp if fp else str(family),
            "scen_text": str(scenario),
            "desc_text": desc if desc else str(scenario),
            "data": data,
            "scope_idx": np.asarray(sorted(g["_rank"].astype(int))),
        })
    return cases


def channel_texts(case, mode):
    """(model channel, scenario channel) texts for the given mode."""
    if mode == 1:
        return case["model_text"], case["scen_text"]
    if mode == 2:
        return case["fp_text"], case["desc_text"]
    raise ValueError(f"Invalid condition mode: {mode}")


def compute_condition_embeddings(texts):
    cache = {}
    if EMB_CACHE_FILE.exists():
        with open(EMB_CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
    missing = sorted(set(t for t in texts if t not in cache))
    if missing:
        from openai import OpenAI
        client = OpenAI(api_key=load_api_key())
        bs = 500
        for i in tqdm(range(0, len(missing), bs), desc="Embedding conditions"):
            batch = missing[i:i + bs]
            resp = client.embeddings.create(input=batch, model=EMBED_MODEL)
            for j, d in enumerate(resp.data):
                cache[batch[j]] = d.embedding
        with open(EMB_CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
    return cache


# ============================================================
# METRICS (same as cVAE / rag_core evaluation)
# ============================================================
def smape_1d(yt, yp, eps=1e-12):
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    return np.mean(np.abs(yp - yt) / np.maximum(denom, eps)) * 100.0


def safe_corr_1d(x, y, method):
    """x = GROUND TRUTH, y = PREDICTION.

    Degenerate series follow scipy's own convention, identical across every
    method in the paper and comparable with correlations reported elsewhere:
      n < 3                -> nan (undefined, dropped from the mean)
      GT constant          -> nan (var(GT) = 0, no trend to predict)
      prediction constant  -> nan (the correlation is undefined either way)

    NB: constancy is tested with ptp (max == min), NOT std == 0.  np.std of ten
    identical 0.03 values is 6.9e-18, not 0.0, so a std-based test misses
    genuinely flat series; scipy then applies its own exact-equality check and
    returns nan.  ptp == 0 is exactly scipy's criterion, so the two agree.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return (pearsonr if method == "pearson" else spearmanr)(x, y)[0]


def trajectory_metrics(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    mask = np.isfinite(yt)
    yt, yp = yt[mask], np.where(np.isfinite(yp[mask]), yp[mask], 0.0)
    if yt.size == 0:
        return dict(n_points=0, mae=np.nan, rmse=np.nan, **{"smape_%": np.nan},
                    pearson=np.nan, spearman=np.nan)
    return {
        "n_points": int(yt.size),
        "mae": float(np.mean(np.abs(yp - yt))),
        "rmse": float(np.sqrt(np.mean((yp - yt) ** 2))),
        "smape_%": float(smape_1d(yt, yp)),
        "pearson": safe_corr_1d(yt, yp, "pearson"),
        "spearman": safe_corr_1d(yt, yp, "spearman"),
    }


# ============================================================
# KNN CORE
# ============================================================
def build_matrices(cases, emb, mode):
    """Return (M_model, M_scenario): per-channel L2-normalized embedding matrices,
    so that M_model[q] @ M_model[t] = cos_model similarity (same for scenario)."""
    def norm(v):
        v = np.asarray(v, dtype=np.float32)
        return v / max(float(np.linalg.norm(v)), 1e-12)

    Mm, Ms = [], []
    for c in cases:
        mt, st = channel_texts(c, mode)
        Mm.append(norm(emb[mt]))
        Ms.append(norm(emb[st]))
    return np.asarray(Mm, dtype=np.float32), np.asarray(Ms, dtype=np.float32)


def knn_predict(query_Ms, query_cases, train_Ms, train_cases, k, weighting, alpha):
    """Predict trajectories [Q, 22, 10] for all query cases.

    similarity = alpha * cos_model + (1 - alpha) * cos_scenario
    HARD CONSTRAINT: neighbors must come from the SAME region as the query.
    If a region has fewer than k train cases, all of them are used.
    """
    qMm, qMs = query_Ms
    tMm, tMs = train_Ms
    n_query = qMm.shape[0]

    train_data = np.stack([c["data"] for c in train_cases])   # [N, 22, 10] (NaN = out of scope)
    train_data_z = np.where(np.isfinite(train_data), train_data, 0.0)

    # pool: same region AND family group allowed for the query (rag STEP2-style
    # 5+2+2 rule, own family included). Every pool member carries all of the
    # query's scoped variables, so scoped predictions never average NaN.
    pool2idx = {}
    for i, c in enumerate(train_cases):
        for gname, fams in POOL_FAMILIES.items():
            if c["model_family"] in fams:
                pool2idx.setdefault((c["region"], gname), []).append(i)
    pool2idx = {k: np.asarray(v) for k, v in pool2idx.items()}

    preds = np.full((n_query, len(VARS_22), len(YEARS)), np.nan)
    for q in range(n_query):
        gname = family_group(query_cases[q]["model_family"])
        cand = pool2idx.get((query_cases[q]["region"], gname))
        if cand is None or len(cand) == 0:
            raise ValueError(f"No allowed train cases in region "
                             f"{query_cases[q]['region']} for group {gname}")
        sims_q = (alpha * (tMm[cand] @ qMm[q])
                  + (1.0 - alpha) * (tMs[cand] @ qMs[q]))
        kk = min(k, len(cand))
        top = np.argpartition(-sims_q, kk - 1)[:kk]
        idx = cand[top]
        s = sims_q[top]
        if weighting == "similarity":
            w = np.maximum(s, 0) + 1e-9
            w = w / w.sum()
        else:
            w = np.full(len(idx), 1.0 / len(idx))
        # NaN-safe per-variable weighted average: each variable uses only the
        # neighbors that actually have it (out-of-scope rows are NaN)
        block = train_data[idx]                       # [k, 22, 10]
        finite = np.isfinite(block[:, :, 0])          # [k, 22] scope per neighbor
        wsum = np.tensordot(w, finite.astype(float), axes=1)   # [22]
        num = np.tensordot(w, np.where(np.isfinite(block), block, 0.0), axes=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            preds[q] = num / wsum[:, None]
        sq = query_cases[q]["scope_idx"]
        assert np.isfinite(preds[q][sq]).all(), "scoped prediction has NaN"
    return round_output(preds)


def round_output(X, nd=OUTPUT_DECIMALS):
    """Round PREDICTIONS to `nd` decimals - the ground truth is left untouched.

    Applied once, at the end of knn_predict(), so the val sweep, the test
    metrics, the xlsx and the rag-schema CSV all see exactly the same numbers.
    Out-of-scope entries are NaN and stay NaN (np.round preserves them).
    """
    return np.round(np.asarray(X, dtype=np.float64), nd)


def per_traj_table(preds, cases):
    """One row per scoped (case, variable) trajectory with every metric.

    Vectorised equivalent of calling trajectory_metrics() on each scoped
    trajectory.
    Same conventions: GT-finite points only, pred NaN -> 0, correlations nan
    when n < 3 or either series is flat (max == min), Spearman = Pearson on
    average ranks.  The bootstrap resamples CASES (id), never trajectories:
    the 22 variables of one case are not independent draws, so the id column
    is what the bootstrap groups on.
    """
    from scipy.stats import rankdata
    ids, vars_, T, P = [], [], [], []
    for i, c in enumerate(cases):
        for v in c["scope_idx"]:
            ids.append(c["case_id"]); vars_.append(VARS_22[int(v)])
            T.append(c["data"][v]); P.append(preds[i, v])
    T = np.asarray(T, float); P = np.asarray(P, float)
    fin = np.isfinite(T)
    P = np.where(np.isfinite(P), P, 0.0)
    n = fin.sum(1).astype(float)
    err = np.where(fin, P - T, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mae = np.abs(err).sum(1) / n
        rmse = np.sqrt((err ** 2).sum(1) / n)
        denom = np.maximum((np.abs(T) + np.abs(P)) / 2.0, 1e-12)
        smape = np.where(fin, np.abs(err) / denom, 0.0).sum(1) / n * 100.0

    def _corr(x, y):
        # Pearson over the GT-finite points of each row; nan if n < 3 or flat
        xm = np.where(fin, x, np.nan); ym = np.where(fin, y, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            xc = xm - np.nanmean(xm, 1, keepdims=True)
            yc = ym - np.nanmean(ym, 1, keepdims=True)
            r = np.nansum(xc * yc, 1) / np.sqrt(np.nansum(xc ** 2, 1) * np.nansum(yc ** 2, 1))
            flat = ((np.nanmax(xm, 1) == np.nanmin(xm, 1))
                    | (np.nanmax(ym, 1) == np.nanmin(ym, 1)))
        return np.where((n < 3) | flat, np.nan, r)

    pear = _corr(T, P)
    # average ranks over the finite points only (out-of-scope points are
    # pushed to +inf so they never disturb the ranks of the real ones)
    rT = np.where(fin, rankdata(np.where(fin, T, np.inf), axis=1), np.nan)
    rP = np.where(fin, rankdata(np.where(fin, P, np.inf), axis=1), np.nan)
    spear = _corr(rT, rP)
    return pd.DataFrame({"id": ids, "variable": vars_, "n_points": n.astype(int),
                         "mae": mae, "rmse": rmse, "smape_%": smape,
                         "pearson": pear, "spearman": spear})


def eval_from_table(df):
    """Run-level metrics = column means of a per_traj_table()."""
    df = df[df["n_points"] > 0].reset_index(drop=True)   # drop out-of-scope rows
    return {
        "n_series": int(len(df)),
        "mae": float(df["mae"].mean()),
        "rmse": float(df["rmse"].mean()),
        "smape_%": float(df["smape_%"].mean()),
        # n_series above is the theoretical denominator; n_valid_* is how many
        # trajectories the correlation was actually defined on.  A constant GT
        # or a constant prediction leaves it undefined, so the two correlations
        # are means over a smaller set than mae/rmse/smape.
        "pearson": float(df["pearson"].mean(skipna=True)),
        "n_valid_pearson": int(df["pearson"].notna().sum()),
        "pct_valid_pearson": (round(df["pearson"].notna().sum() / len(df) * 100, 2)
                            if len(df) else np.nan),
        "spearman": float(df["spearman"].mean(skipna=True)),
        "n_valid_spearman": int(df["spearman"].notna().sum()),
        "pct_valid_spearman": (round(df["spearman"].notna().sum() / len(df) * 100, 2)
                            if len(df) else np.nan),
    }


def eval_full(preds, cases):
    return eval_from_table(per_traj_table(preds, cases))


def case_bootstrap(tables, ref=None, B=BOOT_B, seed=BOOT_SEED, metrics=BOOT_METRICS):
    """PAIRED case-level (cluster) bootstrap over per-trajectory tables.

    tables : {label: per_traj_table(...)}.  Cases = intersection of ids.
    ref    : label the paired differences are taken against (default: first).
    Every replicate resamples the SAME case ids for every label, so the
    difference between two labels is free of the val-set sampling noise they
    share.
    Correlations are nan on flat trajectories; they are excluded per replicate
    (count of finite entries), exactly as eval_full() averages them.

    Returns (summary, diffs):
      summary : one row per label; <m>, <m>_boot_sd, <m>_ci2.5, <m>_ci97.5
      diffs   : one row per (label != ref, metric); mean_diff, ci2.5_diff,
                ci97.5_diff, P(better_than_ref), excludes_zero
    """
    labels = list(tables)
    ref = ref if ref in labels else labels[0]
    ids = None
    for d in tables.values():
        ids = set(d["id"]) if ids is None else ids & set(d["id"])
    ids = np.array(sorted(ids))
    if len(ids) == 0:
        raise ValueError("bootstrap: no common case ids across tables")
    agg = {}
    for lab, d in tables.items():
        g = d[d["id"].isin(ids)].groupby("id")
        agg[lab] = {m: (g[m].count().reindex(ids).to_numpy(float),
                        g[m].sum().reindex(ids).to_numpy(float)) for m in metrics}
    rng = np.random.default_rng(seed)
    take = rng.integers(0, len(ids), (B, len(ids)))    # same resample for ALL labels
    boot = {lab: {} for lab in labels}
    for lab in labels:
        for m in metrics:
            n, s = agg[lab][m]
            with np.errstate(invalid="ignore", divide="ignore"):
                boot[lab][m] = s[take].sum(1) / n[take].sum(1)

    summary = []
    for lab in labels:
        n_traj = int(agg[lab]["mae"][0].sum())
        row = {"config": lab, "n_cases": int(len(ids)), "n_trajectories": n_traj}
        for m in metrics:
            n, s = agg[lab][m]
            row[m] = float(s.sum() / n.sum())            # = eval_full() value
            row[f"{m}_boot_sd"] = float(np.nanstd(boot[lab][m], ddof=1))
            row[f"{m}_ci2.5"] = float(np.nanpercentile(boot[lab][m], 2.5))
            row[f"{m}_ci97.5"] = float(np.nanpercentile(boot[lab][m], 97.5))
        summary.append(row)
    summary = pd.DataFrame(summary)

    diffs = []
    for lab in labels:
        if lab == ref:
            continue
        for m in metrics:
            d = boot[lab][m] - boot[ref][m]
            lo, hi = np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)
            # "better" = lower error, or higher correlation
            better = (d > 0) if m in ("pearson", "spearman") else (d < 0)
            diffs.append({"config": lab, "ref": ref, "metric": m,
                          "mean_diff": float(np.nanmean(d)),
                          "ci2.5_diff": float(lo), "ci97.5_diff": float(hi),
                          "P(better_than_ref)": float(np.nanmean(better)),
                          "excludes_zero": bool(lo > 0 or hi < 0)})
    diffs = pd.DataFrame(diffs)
    return summary, diffs


def bootstrap_run_config(B, seed, ref, n_cases, split, extra=()):
    return pd.DataFrame([
        ("method", "paired case-level (cluster) bootstrap on the saved predictions; "
                   "resampling unit = case id"),
        ("B", B), ("seed", seed), ("ref", ref), ("n_cases", n_cases), ("split", split),
        ("error_bars_cover", f"{split}-set sampling uncertainty (95% percentile CI)"),
        ("note", "KNN is deterministic: no run-to-run component"),
    ] + list(extra), columns=["key", "value"])


def pred_rows(preds, cases):
    """Per-case predictions in the rag_core pred-CSV schema (scoped vars only)."""
    rows = []
    for i, c in enumerate(cases):
        for v in c["scope_idx"]:
            row = {"id": c.get("case_id"), "region": c["region"],
                   "model_family": c["model_family"], "scenario": c["scenario"],
                   "variable": VARS_22[int(v)], "unit": "EJ/yr"}
            row.update({y: preds[i, v, j] for j, y in enumerate(YEARS)})
            rows.append(row)
    return pd.DataFrame(rows)


def cfg_label(k, weighting, alpha):
    return f"k{int(k)}_{weighting}_a{float(alpha)}"


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", type=int, choices=[1, 2], required=True,
                    help="1: names embedding | 2: fingerprint+description embedding")
    args = ap.parse_args()
    mode = args.condition

    print(f"KNN baseline | condition mode: {mode} | fp sheet: {FP_SHEET} | pool: train70")
    fingerprints = read_model_fingerprints()
    desc_dict = load_desc_dict()

    train_cases = read_cases(TRAIN_PATH, fingerprints, desc_dict)
    val_cases   = read_cases(VAL_PATH, fingerprints, desc_dict)
    test_cases  = read_cases(TEST_PATH, fingerprints, desc_dict)
    print(f"train {len(train_cases)} | val {len(val_cases)} | test {len(test_cases)}")

    texts = [t for c in train_cases + val_cases + test_cases
             for t in channel_texts(c, mode)]
    emb = compute_condition_embeddings(texts)

    train_Ms = build_matrices(train_cases, emb, mode)
    val_Ms   = build_matrices(val_cases, emb, mode)
    test_Ms  = build_matrices(test_cases, emb, mode)

    # ---- hyperparameter selection on val (by MAE) ----
    print(f"grid: k {K_GRID} x weighting {WEIGHTINGS} x alpha {ALPHA_GRID} "
          f"= {len(K_GRID) * len(WEIGHTINGS) * len(ALPHA_GRID)} cells")
    results = []
    val_tables = {}          # label -> per-trajectory table, for the bootstrap
    for k in K_GRID:
        for weighting in WEIGHTINGS:
            for alpha in ALPHA_GRID:
                preds = knn_predict(val_Ms, val_cases, train_Ms, train_cases,
                                    k, weighting, alpha)
                tab = per_traj_table(preds, val_cases)
                full = eval_from_table(tab)          # all metrics, logged
                val_tables[cfg_label(k, weighting, alpha)] = tab
                results.append({"k": k, "weighting": weighting,
                                "alpha_model": alpha,
                                "label": cfg_label(k, weighting, alpha),
                                **{f"val_{m}": full[m] for m in
                                   ("mae", "rmse", "smape_%",
                                    "pearson", "n_valid_pearson", "pct_valid_pearson",
                                    "spearman", "n_valid_spearman",
                                    "pct_valid_spearman")}})
                print(f"  k={k:<3} {weighting:<10} alpha={alpha:<5} "
                      f"val MAE {full['mae']:.4f} | sMAPE {full['smape_%']:.2f}% "
                      f"| RMSE {full['rmse']:.3f}")

    results_df = pd.DataFrame(results).sort_values("val_mae")
    best = results_df.iloc[0]
    print(f"\nBest: k={int(best['k'])}, weighting={best['weighting']}, "
          f"alpha_model={best['alpha_model']} (val MAE {best['val_mae']:.4f}, "
          f"val sMAPE {best['val_smape_%']:.2f}%, "
          f"val RMSE {best['val_rmse']:.3f})")
    print("alpha_model is the weight of the MODEL channel; "
          "1-alpha is the weight of the SCENARIO channel.")
    best_label = str(best["label"])
    n_edge = sum(1 for ax, grid in (("k", K_GRID), ("alpha_model", ALPHA_GRID))
                 if len(grid) > 1 and best[ax] in (min(grid), max(grid)))
    if n_edge:
        print(f"  NOTE: the winner sits on {n_edge} grid boundary(ies) -- consider "
              f"extending that axis via KNN_K_GRID / KNN_ALPHA_GRID.")

    # ---- paired case-level bootstrap over the whole val grid ----
    # Descriptive only: selection stays on the val MAE point estimate. The
    # paired differences say which runner-ups are statistically
    # indistinguishable from the winner (diff CI includes 0).
    val_summary = val_diffs = None
    if BOOT_B > 0:
        print(f"\nval bootstrap: {len(val_tables)} configs, B={BOOT_B}, seed={BOOT_SEED} "
              f"(paired on case ids, ref = {best_label})")
        val_summary, val_diffs = case_bootstrap(val_tables, ref=best_label)
        ci = val_summary.set_index("config")
        results_df["val_mae_ci2.5"] = results_df["label"].map(ci["mae_ci2.5"])
        results_df["val_mae_ci97.5"] = results_df["label"].map(ci["mae_ci97.5"])
        dm = val_diffs[val_diffs["metric"] == "mae"].set_index("config")
        results_df["dmae_vs_best"] = results_df["label"].map(dm["mean_diff"])
        results_df["dmae_ci2.5"] = results_df["label"].map(dm["ci2.5_diff"])
        results_df["dmae_ci97.5"] = results_df["label"].map(dm["ci97.5_diff"])
        results_df["P(better_than_best)"] = results_df["label"].map(dm["P(better_than_ref)"])
        results_df["dmae_excludes_zero"] = results_df["label"].map(dm["excludes_zero"])
        tied = results_df[(results_df["label"] != best_label)
                          & (results_df["dmae_excludes_zero"] == False)]   # noqa: E712
        print(f"  winner val MAE {ci.loc[best_label, 'mae']:.4f} "
              f"[{ci.loc[best_label, 'mae_ci2.5']:.4f}, {ci.loc[best_label, 'mae_ci97.5']:.4f}]")
        print(f"  {len(tied)} config(s) whose paired MAE difference vs the winner "
              f"includes 0 (not distinguishable at 95%):")
        for _, r in tied.head(10).iterrows():
            print(f"    {r['label']:<28} dMAE {r['dmae_vs_best']:+.4f} "
                  f"[{r['dmae_ci2.5']:+.4f}, {r['dmae_ci97.5']:+.4f}]")
        if len(tied) > 10:
            print(f"    ... and {len(tied) - 10} more")

    # ---- final test evaluation with the winning config (timed, rag-style) ----
    import time as _time
    _t0 = _time.perf_counter()
    preds = knn_predict(test_Ms, test_cases, train_Ms, train_cases,
                        int(best["k"]), best["weighting"], float(best["alpha_model"]))
    _test_wall = _time.perf_counter() - _t0
    test_tab = per_traj_table(preds, test_cases)
    metrics = eval_from_table(test_tab)
    metrics["timing"] = {
        "what": "test-set prediction with the winning config: cosine similarity vs the "
                "train pool + weighted neighbour average, vectorised over all cases. "
                "Per-case time = wall/cases; no per-case queueing latency exists. "
                "Query embeddings come from the shared cache (as for every method); "
                "embedding-API time is not included.",
        "n_test_cases": len(test_cases),
        "rpt_total_wallclock_sec": round(_test_wall, 3),
        "rpt_sec_per_case_batch": round(_test_wall / len(test_cases), 6),
        "rpt_sec_per_trajectory": round(_test_wall / len(test_cases), 6),
        "note_cost": "local compute; no per-call API cost",
    }
    test_summary = None
    if BOOT_B > 0:
        test_summary, _ = case_bootstrap({best_label: test_tab})
        ts = test_summary.iloc[0]
        for m in BOOT_METRICS:
            for suf in ("_ci2.5", "_ci97.5", "_boot_sd"):
                metrics[f"{m}{suf}"] = float(ts[f"{m}{suf}"])
        metrics["bootstrap"] = {"B": BOOT_B, "seed": BOOT_SEED,
                                "n_cases": int(ts["n_cases"]),
                                "unit": "case id"}
    print("\nTest metrics:")
    print(json.dumps(metrics, indent=2))

    # ---- save ----
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    tag = (f"knn_cond{mode}_k{int(best['k'])}_{best['weighting']}"
           f"_a{best['alpha_model']}_fp-{FP_SHEET}_{stamp}")
    out_dir = _ROOT / "outputs" / "knn" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_excel(out_dir / "val_sweep.xlsx", index=False)
    json.dump(metrics, open(out_dir / "test_metrics.json", "w"), indent=2)
    if val_summary is not None:
        with pd.ExcelWriter(out_dir / "bootstrap_val.xlsx", engine="openpyxl") as w:
            val_summary.to_excel(w, index=False, sheet_name="summary")
            val_diffs.to_excel(w, index=False, sheet_name="paired_diffs")
            bootstrap_run_config(BOOT_B, BOOT_SEED, best_label, int(val_summary["n_cases"].iloc[0]),
                                 "val", [("gt", VAL_PATH)]).to_excel(
                w, index=False, sheet_name="run_config")
        with pd.ExcelWriter(out_dir / "bootstrap_test.xlsx", engine="openpyxl") as w:
            test_summary.to_excel(w, index=False, sheet_name="summary")
            bootstrap_run_config(BOOT_B, BOOT_SEED, best_label, int(test_summary["n_cases"].iloc[0]),
                                 "test", [("gt", TEST_PATH)]).to_excel(
                w, index=False, sheet_name="run_config")
    # the validation predictions of the selected setting in the rag pred-CSV
    # schema, so that the figure scripts score KNN on the same validation cases
    val_preds = knn_predict(val_Ms, val_cases, train_Ms, train_cases,
                            int(best["k"]), best["weighting"], float(best["alpha_model"]))
    pred_rows(val_preds, val_cases).to_csv(out_dir / "AR6_R10sd_val10_pred_knn.csv", index=False)
    json.dump({
        "method": "knn_regression", "condition_mode": mode,
        "selection_metric": "val_mae", "neighbor_pool": "train70",
        "neighbor_constraint": "same_region + family-group pool "
                               "(core->core5; COFFEE/WITCH->pair+core5; GCAM/TIAM->pair+core5)",
        "variable_scope": "core x22 | COFFEE/WITCH x11 supply | GCAM/TIAM x11 demand",
        "similarity": "alpha_model * cos_model + (1 - alpha_model) * cos_scenario, "
                      "separate embeddings per channel; region is a hard filter",
        "output_postprocessing": f"predictions rounded to {OUTPUT_DECIMALS} "
                                 f"decimals (matches GT storage precision)",
        "k": int(best["k"]), "weighting": best["weighting"],
        "alpha_model": float(best["alpha_model"]),
        "val_metrics": {str(k).replace("val_", ""): float(best[k])
                for k in best.index if str(k).startswith("val_")},
        "k_grid": K_GRID, "weightings": WEIGHTINGS, "alpha_grid": ALPHA_GRID,
        "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "metrics": list(BOOT_METRICS),
                      "design": "paired case-level; val: all grid cells vs winner; "
                                "test: winner only"},
        "train_path": TRAIN_PATH, "val_path": VAL_PATH, "test_path": TEST_PATH,
        "fingerprint_sheet": FP_SHEET, "fingerprint_path": FINGERPRINT_PATH,
        "embed_model": EMBED_MODEL,
        "run_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, open(out_dir / "run_config.json", "w"), indent=2)

    # per-case predictions: xlsx for inspection and a rag-schema CSV for the
    # comparison of the methods
    pred_df = pred_rows(preds, test_cases)
    pred_df.to_excel(out_dir / "test_predictions.xlsx", index=False)
    pred_df.to_csv(out_dir / "AR6_R10sd_test_pred_knn.csv", index=False)

    print("saved to:", out_dir)


if __name__ == "__main__":
    main()
