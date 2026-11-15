"""
train_cvae.py
─────────────
Conditional VAE baseline for the 22 energy variables.

Condition modes (each component is embedded separately and the embeddings are
concatenated):
  1 = [embed(model_family); embed(scenario); embed(region)]                   (names)
  2 = [embed(model_fingerprint); embed(scenario_description); embed(region)]  (content)

Data:
  train : make_datasets/output/split/AR6_R10sd_train70_ground_truth.xlsx
  val   : make_datasets/output/split/AR6_R10sd_val10_gt.xlsx
  test  : make_datasets/output/split/AR6_R10sd_test_gt.xlsx (override with CVAE_TEST_PATH)

The model predicts trajectories only. The climate category columns of the
ground-truth files are metadata and are neither an input nor an output.

Targets are standardized before training: log1p followed by a z-score per
variable. All metrics and prediction files are computed in the original units
after the inverse transform.

Model selection:
  - within a run, the checkpoint with the lowest validation loss is kept
    (reconstruction error plus the KL term); the same loss drives early
    stopping
  - across configurations (--sweep), the best checkpoints are ranked by
    validation MAE and the lowest is selected
  - the selected configuration is evaluated on the test set with the decoder
    sampling seeds given in CVAE_EVAL_SEEDS

Grid of the reported baseline (12 configurations per condition):
    lr      in {5e-5, 1e-4, 2e-4, 5e-4}
    hidden  in {512, 1024, 2048}
    latent  = 32
    beta    = 1.0

A grid cell that already has a finished run directory under outputs/cvae
(best_cvae_model.pth and loss_curve.json) is reused, matched on (condition,
lr, hidden, latent, beta). Training is seeded (SEED=42), so cells trained at
different times are comparable. The selected configuration is chosen over the
full grid, reused cells included. --no-resume retrains every cell.

Usage (from the project root):
    python3 cvae/train_cvae.py --condition 1 --sweep            # full grid
    python3 cvae/train_cvae.py --condition 1 --sweep --dry-run  # coverage report only
    python3 cvae/train_cvae.py --condition 1 --sweep --no-resume
    python3 cvae/train_cvae.py --condition 2                    # single default configuration

Env overrides:
    IAM_LLM_FP_SHEET          fingerprint sheet name (default "fingerprint")
    IAM_LLM_FINGERPRINT_PATH  fingerprint xlsx
    CVAE_TEST_PATH     test ground-truth xlsx (default: the full test set)
    CVAE_EPOCHS        upper bound on epochs (default 1500)
    CVAE_PATIENCE      early-stopping patience on the validation loss (default 30)
    CVAE_EVAL_SEEDS    decoder sampling seeds for the final evaluation
                       (default "42"; several seeds give the spread across seeds)
    CVAE_LR_GRID       comma-separated learning rates (default "5e-5,1e-4,2e-4,5e-4")
    CVAE_HIDDEN_GRID   comma-separated hidden sizes   (default "512,1024,2048")
    CVAE_LATENT_GRID   comma-separated latent sizes   (default "32")
    CVAE_BETA_GRID     comma-separated KL weights     (default "1.0")
    CVAE_BOOT_B        case-level bootstrap replicates (default 1000; 0 = off)
    CVAE_BOOT_TOPN     number of top grid cells compared with the selected one
                       in the paired validation bootstrap (default 5)

Case-level bootstrap (resampling unit = case id, B=1000, seed 42, percentile
95% CI):
  - test: the test predictions of the selected configuration are scored per
    trajectory -> bootstrap_test.xlsx, with CI keys added to test_metrics.json
  - validation: the selected configuration is also generated on the validation
    set -> AR6_R10sd_val10_pred_cvae_seed{s}.csv, val_metrics_by_run.json and
    bootstrap_val.xlsx
  - the best CVAE_BOOT_TOPN grid cells are bootstrapped paired against the
    selected one -> sheet "paired_topn" in bootstrap_val.xlsx
  Selection uses the validation MAE point estimate.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import itertools
import json
import pickle
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

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
TEST_PATH  = os.environ.get("CVAE_TEST_PATH", str(_SPLIT / "AR6_R10sd_test_gt.xlsx"))
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
EMB_CACHE_FILE = _CACHE / "cvae_condition_embeddings.pkl"

SEED = 42
# Decoder sampling seed for the VAL metrics that rank grid cells.  Fixed so
# every config is compared on the SAME z draws (common random numbers) and so
# the number is reproducible; see evaluate_val_fixed_seed().
VAL_EVAL_SEED = int(os.environ.get("CVAE_VAL_EVAL_SEED", 42))
BATCH_SIZE = 32
EPOCHS = int(os.environ.get("CVAE_EPOCHS", "1500"))
# Early stopping: stop when val LOSS (ELBO) has not improved for PATIENCE epochs.
PATIENCE = int(os.environ.get("CVAE_PATIENCE", "30"))

BOOT_B = int(os.environ.get("CVAE_BOOT_B", "1000"))
BOOT_SEED = 42
BOOT_TOPN = int(os.environ.get("CVAE_BOOT_TOPN", "5"))
BOOT_METRICS = ("mae", "rmse", "smape_%", "pearson", "spearman")

# beta value pinned during stage 1 of the two-stage sweep (--sweep2)
BETA_REF = float(os.environ.get("CVAE_BETA_REF", "1.0"))

# default single-run configuration
DEFAULT_CONFIG = {"lr": 2e-3, "hidden": 512, "latent": 32, "beta": 1.0}

# sweep grid (identical for condition 1 and condition 2)
# Every axis is env-overridable, so a grid can be EXTENDED without editing the
# file (results are seeded, so a config re-run reproduces its earlier numbers
# exactly and separately-run batches are directly comparable on val MAE).
SWEEP_GRID = {
    "lr": [float(x) for x in
           os.environ.get("CVAE_LR_GRID", "5e-5,1e-4,2e-4,5e-4").split(",") if x.strip()],
    "hidden": [int(x) for x in
               os.environ.get("CVAE_HIDDEN_GRID", "512,1024,2048").split(",") if x.strip()],
    "latent": [int(x) for x in
               os.environ.get("CVAE_LATENT_GRID", "32").split(",") if x.strip()],
    "beta": [float(b) for b in
             os.environ.get("CVAE_BETA_GRID", "1.0").split(",") if b.strip()],
}

# val metrics logged alongside the selection loss in sweep summaries
# The correlation counts travel with the correlations: a sweep row that
# reports pearson without n_valid_pearson hides which denominator it used.
METRIC_KEYS = ("mae", "rmse", "smape_%",
               "pearson", "n_valid_pearson", "pct_valid_pearson",
               "spearman", "n_valid_spearman", "pct_valid_spearman")

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
# DATA LOADING
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
    """scenario -> description from the FULL unsplit dataset (input, not leakage)."""
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
    """Group rows into cases (region, scenario, model_family); canonical variable order."""
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
        vals = g[YEARS].astype(float).to_numpy()
        # CONTRACT: no missing cells INSIDE a scoped variable. The metrics path
        # cannot represent them: transform() zero-fills before the mask in
        # trajectory_metrics() can see a NaN, and predict() scores against the
        # round-tripped target tensor, so a within-scope hole would be silently
        # compared to expm1(mu[row]) -- a fabricated, plausible-looking GT that
        # this model tends to match -- instead of being dropped from the series.
        # (Out-of-scope variables are a different case, correctly excluded by
        # the `mv[i, v] > 0` guard in metrics_from_arrays.)
        if not np.isfinite(vals).all():
            raise ValueError(f"Case {region}|{scenario}|{family} has non-finite "
                             f"cells inside its scoped variables; the metrics "
                             f"path would score them against a fabricated GT. "
                             f"Fix the dataset.")
        data = np.full((len(VARS_22), len(YEARS)), np.nan)
        data[g["_rank"].astype(int).to_numpy()] = vals
        scope_mask = np.zeros(len(VARS_22), dtype=np.float32)
        scope_mask[g["_rank"].astype(int).to_numpy()] = 1.0
        desc = desc_dict.get(str(scenario).strip(), "")
        fp = fingerprints.get(str(family).strip(), "")
        cases.append({
            "case_id": (int(g[id_col].iloc[0]) if id_col else None),
            "region": str(region),
            "scenario": str(scenario),
            "model_family": str(family),
            # separate channels, embedded independently
            "model_text": str(family),
            "fp_text": fp if fp else str(family),
            "scen_text": str(scenario),
            "desc_text": desc if desc else str(scenario),
            "region_text": str(region),
            "data": data.tolist(),          # NaN = out of the family's scope
            "scope_mask": scope_mask.tolist(),
        })
    return cases


def condition_texts(case, mode):
    """Texts whose embeddings form the condition vector (one channel each)."""
    if mode == 1:   # name-based
        return [case["model_text"], case["scen_text"], case["region_text"]]
    if mode == 2:   # content-based
        return [case["fp_text"], case["desc_text"], case["region_text"]]
    raise ValueError(f"Invalid condition mode: {mode}")


def _load_emb_cache():
    if not EMB_CACHE_FILE.exists():
        return {}
    try:
        with open(EMB_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError, MemoryError) as e:
        print(f"  [emb cache] unreadable ({e}); re-embedding from scratch")
        return {}


def compute_condition_embeddings(texts):
    """Embed the condition texts, memoised in a shared on-disk pickle.

    CONCURRENCY: condition 1 and condition 2 are normally run as two separate
    processes and share this one cache file. A naive load-update-dump would let
    whichever finished last clobber the other's entries, and a reader could hit
    a half-written file. So the write re-reads the cache, MERGES, and swaps the
    file in with os.replace (atomic on POSIX). Worst case under a race is that
    one process re-embeds a few texts - never corruption or lost entries.
    """
    cache = _load_emb_cache()
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
        cache = {**_load_emb_cache(), **cache}      # keep the other process's
        tmp = EMB_CACHE_FILE.with_name(f"{EMB_CACHE_FILE.name}.tmp{os.getpid()}")
        with open(tmp, "wb") as f:
            pickle.dump(cache, f)
        os.replace(tmp, EMB_CACHE_FILE)             # atomic swap
    return cache


N_ROWS = len(VARS_22)              # 22 variable rows


def build_matrices(cases):
    """Stack cases into raw arrays: X[N,22,10] (NaN out of scope), M[N,22,10]."""
    X = np.full((len(cases), N_ROWS, len(YEARS)), np.nan, dtype=np.float64)
    Mk = np.zeros((len(cases), N_ROWS, len(YEARS)), dtype=np.float32)
    for i, c in enumerate(cases):
        X[i] = np.asarray(c["data"], dtype=np.float64)        # NaN = out of scope
        sm = np.asarray(c["scope_mask"], dtype=np.float32)    # [22]
        Mk[i] = sm[:, None]
    return X, Mk


def fit_standardizer(X_train):
    """Per-row mu/sd on TRAIN observed entries; all rows in log1p space."""
    Z = np.log1p(np.clip(X_train.copy(), 0, None))
    mu = np.zeros(N_ROWS)
    sd = np.ones(N_ROWS)
    for r in range(N_ROWS):
        v = Z[:, r, :][np.isfinite(Z[:, r, :])]
        mu[r] = float(v.mean())
        sd[r] = float(max(v.std(), 1e-6))
    return mu, sd


def transform(X, mu, sd):
    """Raw -> standardized; out-of-scope entries -> 0 (the row mean)."""
    Z = np.log1p(np.clip(X.copy(), 0, None))
    Z = (Z - mu[None, :, None]) / sd[None, :, None]
    return np.where(np.isfinite(Z), Z, 0.0).astype(np.float32)


def inverse_transform(Z, mu, sd):
    """Standardized -> raw; variables clipped to >= 0."""
    X = Z.astype(np.float64) * sd[None, :, None] + mu[None, :, None]
    return np.clip(np.expm1(X), 0, None)


OUTPUT_DECIMALS = 2


def round_output(X, nd=OUTPUT_DECIMALS):
    """Round reported values to `nd` decimals.

    Applied to PREDICTIONS ONLY - the ground truth is left untouched, so the
    metrics below are scored against exactly the values rag_core scores
    against.
    """
    return np.round(np.asarray(X, dtype=np.float64), nd)


# ============================================================
# MODEL
# ============================================================
class cVAEData(Dataset):
    def __init__(self, cases, emb, mode, Z, Mk):
        self.cases, self.emb, self.mode = cases, emb, mode
        self.Z, self.Mk = Z, Mk          # standardized targets + loss mask

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        item = self.cases[idx]
        vecs = [self.emb[t] for t in condition_texts(item, self.mode)]
        e = torch.tensor(np.concatenate(vecs), dtype=torch.float32)
        return e, torch.tensor(self.Z[idx]), torch.tensor(self.Mk[idx])


class Encoder(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + cond_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x, c):
        h = self.relu(self.fc1(torch.cat([x.view(x.size(0), -1), c], dim=1)))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dim, output_dim, output_shape):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.output_shape = output_shape

    def forward(self, z, c):
        h = self.relu(self.fc1(torch.cat([z, c], dim=1)))
        return self.fc2(h).view(-1, *self.output_shape)


class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim, latent_dim, output_shape):
        super().__init__()
        self.encoder = Encoder(input_dim, cond_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, cond_dim, hidden_dim, input_dim, output_shape)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, c), mu, logvar

    def generate(self, c, latent_dim):
        z = torch.randn(c.size(0), latent_dim).to(c.device)
        return self.decoder(z, c)


def loss_function(recon_x, x, mu, logvar, mask, beta=1.0):
    """Masked beta-ELBO: reconstruction error counted ONLY on entries the case
    actually has (its family's variable scope). beta weights the KL term;
    beta=1 is the plain ELBO."""
    mse = torch.sum(((recon_x - x) ** 2) * mask)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kld


# ============================================================
# METRICS (aligned with rag_core evaluation)
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


def predict(model, loader, device, latent_dim, mu, sd, X_raw=None):
    """Generate, then map back to RAW space so every metric below (and the pred
    CSV) is computed on physical values. inverse_transform also enforces the
    post-processing: variables clipped to >= 0. Predictions are then rounded to
    OUTPUT_DECIMALS decimals; the truths are NOT, so metrics stay comparable to
    rag_core's."""
    model.eval()
    preds, trues, masks = [], [], []
    with torch.no_grad():
        for e, x, m in loader:
            preds.append(model.generate(e.to(device), latent_dim).cpu().numpy())
            trues.append(x.numpy())
            masks.append(m.numpy())
    preds = round_output(inverse_transform(np.concatenate(preds), mu, sd))

    trues_rt = inverse_transform(np.concatenate(trues), mu, sd)
    if X_raw is None:
        trues = trues_rt
    else:
        trues = np.asarray(X_raw, dtype=np.float64)
        if trues.shape != trues_rt.shape:
            raise ValueError(f"X_raw shape {trues.shape} != loader output "
                             f"{trues_rt.shape}; the raw targets and the loader "
                             f"are out of step (is the loader shuffling?)")
        # alignment guard: the round-tripped truth must still agree with the raw
        # one to float32 precision, otherwise the rows do not correspond.
        fin = np.isfinite(trues)
        dev = np.abs(trues_rt[fin] - trues[fin]) / np.maximum(np.abs(trues[fin]), 1.0)
        if dev.size and float(dev.max()) > 1e-4:
            raise ValueError(f"raw targets and loader targets disagree "
                             f"(max rel dev {float(dev.max()):.2e}) -- refusing "
                             f"to score against a mismatched ground truth")
    return preds, trues, np.concatenate(masks)


def val_smape(model, loader, device, latent_dim, mu, sd, X_raw=None):
    """Fast objective used for checkpoint + hyperparameter selection."""
    preds, trues, masks = predict(model, loader, device, latent_dim, mu, sd, X_raw)
    mv = masks[:, :, 0]
    vals = [smape_1d(trues[i, v], np.where(np.isfinite(preds[i, v]), preds[i, v], 0.0))
            for i in range(trues.shape[0]) for v in range(trues.shape[1])
            if mv[i, v] > 0]
    return float(np.mean(vals))


def per_traj_table(preds, trues, masks, cases=None):
    """One row per scoped (case, variable) trajectory with every metric.

    Vectorised equivalent of calling trajectory_metrics() on every scoped
    trajectory.  Same conventions: GT-finite points only, pred NaN
    -> 0, correlations nan when n < 3 or either series is flat (max == min),
    Spearman = Pearson on average ranks.  The `id` column (case id when
    `cases` is given, else the row index) is what the case-level bootstrap
    resamples: the 22 variables of one case are not independent draws.
    """
    from scipy.stats import rankdata
    mv = masks[:, :, 0] > 0
    ci, vi = np.nonzero(mv)                      # scoped (case, variable) pairs
    T = np.asarray(trues, float)[ci, vi]         # [S, 10]
    P = np.asarray(preds, float)[ci, vi]
    ids = ([cases[i].get("case_id") for i in ci] if cases is not None else ci.tolist())
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
        xm = np.where(fin, x, np.nan); ym = np.where(fin, y, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            xc = xm - np.nanmean(xm, 1, keepdims=True)
            yc = ym - np.nanmean(ym, 1, keepdims=True)
            r = np.nansum(xc * yc, 1) / np.sqrt(np.nansum(xc ** 2, 1) * np.nansum(yc ** 2, 1))
            flat = ((np.nanmax(xm, 1) == np.nanmin(xm, 1))
                    | (np.nanmax(ym, 1) == np.nanmin(ym, 1)))
        return np.where((n < 3) | flat, np.nan, r)

    pear = _corr(T, P)
    rT = np.where(fin, rankdata(np.where(fin, T, np.inf), axis=1), np.nan)
    rP = np.where(fin, rankdata(np.where(fin, P, np.inf), axis=1), np.nan)
    spear = _corr(rT, rP)
    return pd.DataFrame({"id": ids, "variable": [VARS_22[int(v)] for v in vi],
                         "n_points": n.astype(int),
                         "mae": mae, "rmse": rmse, "smape_%": smape,
                         "pearson": pear, "spearman": spear})


def eval_from_table(df):
    """Run-level metrics = column means of a per_traj_table()."""
    df = df[df["n_points"] > 0].reset_index(drop=True)
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


def metrics_from_arrays(preds, trues, masks):
    return eval_from_table(per_traj_table(preds, trues, masks))


def evaluate(model, loader, device, latent_dim, mu, sd, X_raw=None):
    return metrics_from_arrays(*predict(model, loader, device, latent_dim, mu, sd,
                                        X_raw))


def case_bootstrap(tables, ref=None, B=BOOT_B, seed=BOOT_SEED, metrics=BOOT_METRICS):
    """PAIRED case-level (cluster) bootstrap over per-trajectory tables.

    tables : {label: per_traj_table(...)}.  Cases = intersection of ids.
    ref    : label the paired differences are taken against (default: first).
    Every replicate resamples the SAME case ids for every label, so the
    difference between two labels is free of the val-set sampling noise they
    share.
    Correlations are nan on flat trajectories; they are excluded per replicate
    (count of finite entries), exactly as eval_from_table() averages them.

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
        row = {"config": lab, "n_cases": int(len(ids)),
               "n_trajectories": int(agg[lab]["mae"][0].sum())}
        for m in metrics:
            n, s = agg[lab][m]
            row[m] = float(s.sum() / n.sum())            # = eval_from_table() value
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
            better = (d > 0) if m in ("pearson", "spearman") else (d < 0)
            diffs.append({"config": lab, "ref": ref, "metric": m,
                          "mean_diff": float(np.nanmean(d)),
                          "ci2.5_diff": float(lo), "ci97.5_diff": float(hi),
                          "P(better_than_ref)": float(np.nanmean(better)),
                          "excludes_zero": bool(lo > 0 or hi < 0)})
    return summary, pd.DataFrame(diffs)


def mean_of_runs(tables):
    """Per-trajectory mean over repeated runs (nan-aware)."""
    cat = pd.concat(list(tables.values()), ignore_index=True)
    return (cat.groupby(["id", "variable"], as_index=False)
               .agg({"n_points": "first", "mae": "mean", "rmse": "mean",
                     "smape_%": "mean", "pearson": "mean", "spearman": "mean"}))


def bootstrap_run_config(B, seed, ref, n_cases, split, extra=()):
    return pd.DataFrame([
        ("method", "paired case-level (cluster) bootstrap on the saved predictions; "
                   "resampling unit = case id"),
        ("B", B), ("seed", seed), ("ref", ref), ("n_cases", n_cases), ("split", split),
        ("error_bars_cover", f"{split}-set sampling uncertainty (95% percentile CI)"),
        ("run_rows_cover", "decoder sampling stochasticity (one row per sampling seed; "
                           "mean_of_seeds = per-trajectory mean, the cross-method representation)"),
    ] + list(extra), columns=["key", "value"])


def generate_runs(model, loader, device, latent_dim, mu, sd, X_raw, cases, seeds,
                  out_dir, split, csv_stem):
    """Generate `split` predictions once per sampling seed: per-run metrics,
    per-run per-trajectory tables, and a rag-schema pred CSV per seed."""
    per_run, tables = [], {}
    for s_ in seeds:
        torch.manual_seed(s_)               # controls the decoder's z draws
        preds, trues, masks = predict(model, loader, device, latent_dim, mu, sd, X_raw)
        tab = per_traj_table(preds, trues, masks, cases)
        m = eval_from_table(tab)
        pred_csv = out_dir / f"{csv_stem}_seed{s_}.csv"
        save_pred_csv(preds, cases, pred_csv)
        print(f"  {split} sampling seed {s_}: MAE {m['mae']:.4f} | sMAPE {m['smape_%']:.2f}% "
              f"| pred -> {pred_csv.name}")
        per_run.append({"seed": s_, **m})
        tables[f"seed{s_}"] = tab
    return per_run, tables


def bootstrap_runs(tables, seeds):
    """Bootstrap every seed run AND the per-trajectory mean of the seeds.
    Returns (summary, mean-of-seeds row as a dict)."""
    all_tabs = dict(tables)
    if len(tables) > 1:
        all_tabs["mean_of_seeds"] = mean_of_runs(tables)
    rep_lab = "mean_of_seeds" if len(tables) > 1 else next(iter(tables))
    summary, _ = case_bootstrap(all_tabs, ref=rep_lab)
    return summary, summary[summary["config"] == rep_lab].iloc[0].to_dict()


def save_pred_csv(preds, cases, path):
    """Per-case predictions in the rag_core pred-CSV schema (scoped vars only),
    so the cross-method bootstrap can consume them directly.
    Trajectories only - no category column is produced by this model."""
    rows = []
    for i, c in enumerate(cases):
        for v_idx in np.where(np.asarray(c["scope_mask"]) > 0)[0]:
            row = {"id": c.get("case_id"), "region": c["region"],
                   "model_family": c["model_family"], "scenario": c["scenario"],
                   "variable": VARS_22[int(v_idx)], "unit": "EJ/yr"}
            for j, y in enumerate(YEARS):
                row[y] = float(preds[i, int(v_idx), j])
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def evaluate_val_fixed_seed(model, loader, device, latent_dim, mu, sd, X_raw,
                            seed=VAL_EVAL_SEED):
    """Validation metrics with a fixed, isolated decoder sampling seed.

    The decoder draws z ~ N(0,1) at generation time. A fixed seed scores every
    configuration on the same draws (within a latent_dim), so the ranking of
    the grid cells does not depend on the sampling noise.

    The global RNG state is saved before the call and restored after it, so
    that training proceeds as if this call had not happened.
    """
    cpu_state = torch.get_rng_state()
    dev_state = None
    if device.type == "cuda":
        dev_state = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        try:
            dev_state = torch.mps.get_rng_state()
        except Exception:
            dev_state = None
    try:
        torch.manual_seed(seed)
        return evaluate(model, loader, device, latent_dim, mu, sd, X_raw)
    finally:
        torch.set_rng_state(cpu_state)
        if dev_state is not None:
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(dev_state)
            else:
                torch.mps.set_rng_state(dev_state)


# ============================================================
# TRAINING
# ============================================================
def train_one(config, mode, loaders, dims, device, save_dir, epochs, mu, sd,
              seed=SEED):
    """Train one config; checkpoint selection by VAL loss. Returns best val loss."""
    tr_loader, va_loader, X_va_raw = loaders
    input_dim, cond_dim, out_shape = dims

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CVAE(input_dim, cond_dim, config["hidden"], config["latent"], out_shape).to(device)
    opt = optim.Adam(model.parameters(), lr=config["lr"])
    save_dir.mkdir(parents=True, exist_ok=True)

    import time as _time
    _t_train0 = _time.perf_counter()
    best_loss, best_epoch, best_metrics = float("inf"), -1, {}
    tr_losses, va_losses, va_history = [], [], []
    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for e, x, m in tr_loader:
            e, x, m = e.to(device), x.to(device), m.to(device)
            opt.zero_grad()
            recon, q_mu, q_logvar = model(x, e)
            loss = loss_function(recon, x, q_mu, q_logvar, m, config["beta"])
            loss.backward()
            tl += loss.item()
            opt.step()
        tl /= len(tr_loader.dataset)

        # validation ELBO loss (natural objective) - used for selection
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for e, x, m in va_loader:
                e, x, m = e.to(device), x.to(device), m.to(device)
                recon, q_mu, q_logvar = model(x, e)
                vl += loss_function(recon, x, q_mu, q_logvar, m,
                                    config["beta"]).item()
        vl /= len(va_loader.dataset)
        # full val metrics (logged every epoch; selection stays on val loss)
        vm = evaluate_val_fixed_seed(model, va_loader, device,
                                     config["latent"], mu, sd, X_va_raw)
        tr_losses.append(tl)
        va_losses.append(vl)
        va_history.append({"epoch": epoch, "val_loss": vl, **vm})

        if vl < best_loss:
            best_loss, best_epoch, best_metrics = vl, epoch, vm
            torch.save(model.state_dict(), save_dir / "best_cvae_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch}/{epochs} | train loss {tl:.4f} | val loss {vl:.4f} "
                  f"| val MAE {vm['mae']:.4f} | val sMAPE {vm['smape_%']:.2f}% "
                  f"(best loss {best_loss:.4f} @ ep{best_epoch})")

        if epochs_no_improve >= PATIENCE:
            print(f"  early stop at epoch {epoch} "
                  f"(no val LOSS improvement for {PATIENCE} epochs; best @ ep{best_epoch})")
            break

    _train_sec = _time.perf_counter() - _t_train0
    print(f"  train wall clock: {_train_sec:.1f}s "
          f"({len(tr_losses)} epochs on {device})")
    json.dump({"train_loss": tr_losses, "val_loss": va_losses,
               "val_metrics_per_epoch": va_history,
               "best_val_loss": best_loss, "best_epoch": best_epoch,
               "best_val_metrics": best_metrics,
               "train_sec": round(_train_sec, 1),
               "epochs_trained": len(tr_losses),
               "device": str(device)},
              open(save_dir / "loss_curve.json", "w"), indent=2)
    if HAS_MPL:
        fig, ax1 = plt.subplots()
        ax1.plot(tr_losses, label="train loss", color="tab:blue")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss")
        ax2 = ax1.twinx()
        ax2.plot([h["smape_%"] for h in va_history], label="val sMAPE", color="tab:red")
        ax2.set_ylabel("val sMAPE (%)")
        fig.legend(loc="upper right")
        plt.savefig(save_dir / "loss_curve.png"); plt.close()

    return best_loss, best_metrics, best_epoch, model


# ── resume: reuse finished run directories instead of retraining ──────────────
# A run directory is identified by its tag, which encodes the full config:
#   cond{mode}_lr{lr}_hid{hidden}_lat{latent}_beta{beta}_fp-{sheet}_{stamp}
# Resume matches on (mode, lr, hidden, latent, beta, fp sheet) and IGNORES the
# timestamp, so cells trained in different batches all count as covered.
TAG_RE = re.compile(
    r"^cond(?P<mode>\d+)_lr(?P<lr>[0-9eE.+-]+)_hid(?P<hidden>\d+)"
    r"_lat(?P<latent>\d+)_beta(?P<beta>[0-9eE.+-]+)_fp-(?P<fp>.+)_"
    r"(?P<stamp>\d{8}_\d{4})$"
)


def config_key(cfg):
    """Hashable identity of a grid cell (floats normalised so 5e-4 == 0.0005)."""
    return (round(float(cfg["lr"]), 12), int(cfg["hidden"]),
            int(cfg["latent"]), round(float(cfg["beta"]), 12))


def scan_finished_runs(root, mode):
    """Index every COMPLETE run dir for this condition -> {config_key: row}.

    Complete = has both best_cvae_model.pth (needed for the final test eval)
    and loss_curve.json (holds the selection metrics). Incomplete/crashed dirs
    are ignored, so they simply get retrained. If several timestamps exist for
    the same cell the NEWEST is used and the others are reported.
    """
    found = {}
    if not root.exists():
        return found
    for d in sorted(root.glob(f"cond{mode}_lr*_fp-*")):
        if not d.is_dir():
            continue
        m = TAG_RE.match(d.name)
        if not m or m.group("fp") != FP_SHEET:
            continue
        curve, ckpt = d / "loss_curve.json", d / "best_cvae_model.pth"
        if not (curve.exists() and ckpt.exists()):
            print(f"  [resume] incomplete, will retrain: {d.name}")
            continue
        try:
            lc = json.load(open(curve))
            vm = lc["best_val_metrics"]
            row = {
                "lr": float(m.group("lr")), "hidden": int(m.group("hidden")),
                "latent": int(m.group("latent")), "beta": float(m.group("beta")),
                "stage": "cached", "tag": d.name,
                "val_loss": float(lc["best_val_loss"]),
                "best_epoch": int(lc["best_epoch"]),
                **{f"val_{k}": v for k, v in vm.items() if k in METRIC_KEYS},
                # split fingerprint: see the SPLIT GUARD in main()
                "_val_n_series": int(vm.get("n_series", -1)),
                "stamp": m.group("stamp"),
            }
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            print(f"  [resume] unreadable, will retrain: {d.name} ({e})")
            continue
        k = config_key(row)
        prev = found.get(k)
        # same cell trained more than once -> keep the newest run (the tag
        # stamp is YYYYmmdd_HHMM, so a plain string compare is chronological)
        if prev is None or row["stamp"] > prev["stamp"]:
            if prev is not None:
                print(f"  [resume] duplicate cell, using newer {d.name} "
                      f"(ignoring {prev['tag']})")
            found[k] = row
        else:
            print(f"  [resume] duplicate cell, keeping {prev['tag']} "
                  f"(ignoring {d.name})")
    return {k: {kk: vv for kk, vv in r.items() if kk != "stamp"}
            for k, r in found.items()}


def print_coverage(configs, cache, mode):
    """Grid coverage matrix: which cells are cached and which need training."""
    print(f"\n--- grid coverage, condition {mode} "
          f"({len(configs)} cells: * = cached, . = to train) ---")
    df = pd.DataFrame([{**c, "have": "*" if config_key(c) in cache else "."}
                       for c in configs])
    for beta, sub in df.groupby("beta"):
        print(f"beta = {beta}")
        print(sub.pivot_table(index="lr", columns=["hidden", "latent"],
                              values="have", aggfunc="first").to_string())
    n_have = sum(1 for c in configs if config_key(c) in cache)
    print(f"cached {n_have}/{len(configs)} | to train {len(configs) - n_have}")
    return n_have


def run_configs(configs, mode, loaders, dims, device, root, stamp, mu, sd, stage,
                cache=None):
    """Return one result row per config, training only the cells not in cache."""
    cache = cache or {}
    rows, todo = [], []
    for cfg in configs:
        hit = cache.get(config_key(cfg))
        (rows if hit else todo).append(hit if hit else cfg)
    for i, cfg in enumerate(todo, 1):
        tag = (f"cond{mode}_lr{cfg['lr']}_hid{cfg['hidden']}_lat{cfg['latent']}"
               f"_beta{cfg['beta']}_fp-{FP_SHEET}_{stamp}")
        print(f"\n=== [{stage}] ({i}/{len(todo)}) {tag} ===")
        best_loss, best_vm, best_epoch, _ = train_one(
            cfg, mode, loaders, dims, device, root / tag, EPOCHS, mu, sd)
        row = {**cfg, "stage": stage, "tag": tag, "val_loss": best_loss,
               "best_epoch": best_epoch,
               **{f"val_{k}": v for k, v in best_vm.items()
                  if k in METRIC_KEYS}}
        rows.append(row)
        cache[config_key(row)] = row       # so a later stage can reuse it too
    return rows


def clear_stale_test_eval(root, mode, winner_tag):
    """Move test artefacts out of non-winner dirs of this condition.

    A previous batch may have written test predictions into whatever config
    won THAT batch. Once the full-grid winner is known those files are stale
    and must not be picked up by downstream analysis, so they are moved to
    outputs/cvae/_stale_test_eval/<tag>/ rather than deleted.
    """
    names = ["test_metrics.json", "test_metrics_by_run.json",
             "training_hyperparameters.json",
             "val_metrics_by_run.json", "bootstrap_test.xlsx", "bootstrap_val.xlsx"]
    for d in sorted(root.glob(f"cond{mode}_lr*_fp-*")):
        if not d.is_dir() or d.name == winner_tag:
            continue
        stale = [p for p in d.iterdir()
                 if p.name in names or p.name.startswith("AR6_")]
        if not stale:
            continue
        dest = root / "_stale_test_eval" / d.name
        dest.mkdir(parents=True, exist_ok=True)
        for p in stale:
            shutil.move(str(p), str(dest / p.name))
        print(f"  [cleanup] moved {len(stale)} stale test file(s) from "
              f"{d.name} -> _stale_test_eval/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", type=int, choices=[1, 2], required=True,
                    help="1: [family; scenario; region] embeddings | "
                         "2: [fingerprint; description; region] embeddings")
    ap.add_argument("--sweep", action="store_true",
                    help="FULL rectangular grid over lr x hidden x latent x "
                         "beta; winner picked over the whole grid by val "
                         "MAE. THIS IS THE REPORTED PROTOCOL.")
    ap.add_argument("--sweep2", action="store_true",
                    help="Two-stage sweep: stage 1 searches "
                         "lr/hidden/latent with beta fixed at CVAE_BETA_REF, "
                         "stage 2 searches the remaining CVAE_BETA_GRID "
                         "values on the stage-1 winner. With "
                         "the default beta grid (1.0 only) stage 2 is empty.")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="retrain every cell even if a finished run directory "
                         "for it already exists (default: reuse it)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the grid coverage report and exit without "
                         "training or evaluating anything")
    args = ap.parse_args()
    mode = args.condition

    root = _ROOT / "outputs" / "cvae"
    root.mkdir(parents=True, exist_ok=True)   # may not exist after an archive
    grid_configs = [dict(zip(SWEEP_GRID, v))
                    for v in itertools.product(*SWEEP_GRID.values())]

    print(f"Condition mode: {mode} | fp sheet: {FP_SHEET} | "
          f"selection: checkpoint by val loss (ELBO), config by val MAE")
    print(f"grid: lr {SWEEP_GRID['lr']} x hidden {SWEEP_GRID['hidden']} "
          f"x latent {SWEEP_GRID['latent']} x beta {SWEEP_GRID['beta']} "
          f"= {len(grid_configs)} cells")

    # scan finished runs first so --dry-run costs nothing (no data, no API)
    cache = scan_finished_runs(root, mode) if args.resume else {}
    if not args.resume:
        print("resume DISABLED (--no-resume): every cell will be retrained")
    if args.sweep or args.dry_run:
        print_coverage(grid_configs, cache, mode)
    if args.dry_run:
        print("\n--dry-run: nothing trained, nothing evaluated.")
        return

    fingerprints = read_model_fingerprints()
    desc_dict = load_desc_dict()

    train_cases = read_cases(TRAIN_PATH, fingerprints, desc_dict)
    val_cases   = read_cases(VAL_PATH, fingerprints, desc_dict)
    test_cases  = read_cases(TEST_PATH, fingerprints, desc_dict)
    print(f"train {len(train_cases)} | val {len(val_cases)} | test {len(test_cases)}")

    texts = [t for c in train_cases + val_cases + test_cases
             for t in condition_texts(c, mode)]
    emb = compute_condition_embeddings(texts)

    # standardization fitted on TRAIN only, applied to all three splits
    X_tr, M_tr = build_matrices(train_cases)
    X_va, M_va = build_matrices(val_cases)

    n_series_now = int((M_va[:, :, 0] > 0).sum())
    stale = {k: r for k, r in cache.items()
             if r.get("_val_n_series", -1) != n_series_now}
    if stale:
        print(f"\n  [resume] VAL SPLIT MISMATCH: {len(stale)} cached cell(s) were "
              f"scored on a different val split (this split has "
              f"n_series={n_series_now}); they will be retrained:")
        for r in list(stale.values())[:5]:
            print(f"    {r['tag']}  (cached n_series={r.get('_val_n_series')})")
        if len(stale) > 5:
            print(f"    ... and {len(stale) - 5} more")
        for k in list(stale):
            del cache[k]
    # --------------------------------------------------------------------
    X_te, M_te = build_matrices(test_cases)
    mu, sd = fit_standardizer(X_tr)
    Z_tr, Z_va, Z_te = (transform(X_tr, mu, sd), transform(X_va, mu, sd),
                        transform(X_te, mu, sd))
    print(f"standardized targets: {N_ROWS} variable rows, all unit variance "
          f"(log1p space)")

    tr_ds = cVAEData(train_cases, emb, mode, Z_tr, M_tr)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(cVAEData(val_cases, emb, mode, Z_va, M_va),
                           batch_size=BATCH_SIZE)
    te_loader = DataLoader(cVAEData(test_cases, emb, mode, Z_te, M_te),
                           batch_size=BATCH_SIZE)

    e0, x0, _m0 = tr_ds[0]
    cond_dim, out_shape = e0.shape[0], x0.shape
    input_dim = int(np.prod(out_shape))
    dims = (input_dim, cond_dim, out_shape)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    stamp = f"{datetime.now():%Y%m%d_%H%M}"

    loaders = (tr_loader, va_loader, X_va)   # X_va: raw float64 val targets
    common = (mode, loaders, dims, device, root, stamp, mu, sd)

    if args.sweep2:
        # ---- stage 1: architecture / optimiser, beta pinned ----
        arch = {k: SWEEP_GRID[k] for k in ("lr", "hidden", "latent")}
        s1 = [dict(zip(arch, v), beta=BETA_REF)
              for v in itertools.product(*arch.values())]
        print(f"\nTWO-STAGE sweep | stage 1: {len(s1)} configs "
              f"(lr x hidden x latent, beta pinned at {BETA_REF})")
        results = run_configs(s1, *common, stage="1-arch", cache=cache)

        # ---- stage 2: beta, on the stage-1 winner ----
        w = pd.DataFrame(results).sort_values("val_mae").iloc[0]
        s2 = [{"lr": float(w["lr"]), "hidden": int(w["hidden"]),
               "latent": int(w["latent"]), "beta": b}
              for b in SWEEP_GRID["beta"] if abs(b - BETA_REF) > 1e-12]
        print(f"\nstage 1 winner: lr={w['lr']} hidden={int(w['hidden'])} "
              f"latent={int(w['latent'])} (val MAE {w['val_mae']:.4f}, "
              f"val sMAPE {w['val_smape_%']:.2f}%)")
        print(f"stage 2: {len(s2)} configs (beta in "
              f"{[c['beta'] for c in s2]}, architecture pinned)")
        results += run_configs(s2, *common, stage="2-beta", cache=cache)
    elif args.sweep:
        print(f"\nFULL grid sweep: {len(grid_configs)} cells "
              f"({len(grid_configs) - sum(1 for c in grid_configs if config_key(c) in cache)}"
              f" to train, the rest reused)")
        results = run_configs(grid_configs, *common, stage="grid", cache=cache)
    else:
        results = run_configs([DEFAULT_CONFIG], *common, stage="single",
                              cache=cache)

    results_df = (pd.DataFrame(results)
                  .drop(columns=["_val_n_series"], errors="ignore")
                  .sort_values("val_mae"))
    n_cached = int((results_df["stage"] == "cached").sum())
    print(f"\nConfig ranking by val MAE (checkpoints selected by val loss) "
          f"| {len(results_df)} configs, {n_cached} reused from earlier runs:")
    print(results_df.to_string(index=False))

    # winner: train ONCE (already done in the selection loop); evaluate its
    # best checkpoint on test N times with different SAMPLING seeds (the
    # decoder draws z each run) and report mean +/- SD
    best = results_df.iloc[0]
    best_dir = root / best["tag"]
    cfg_best = {"lr": float(best["lr"]), "hidden": int(best["hidden"]),
                "latent": int(best["latent"]), "beta": float(best["beta"])}
    EVAL_SEEDS = [int(s) for s in
                  os.environ.get("CVAE_EVAL_SEEDS", "42").split(",")]

    model = CVAE(input_dim, cond_dim, cfg_best["hidden"], cfg_best["latent"],
                 out_shape).to(device)
    model.load_state_dict(torch.load(best_dir / "best_cvae_model.pth",
                                     map_location=device))

    print(f"\nBest config over the FULL grid ({len(results_df)} cells): "
          f"{best['tag']} (val MAE {best['val_mae']:.4f}, "
          f"val loss {best['val_loss']:.4f}, "
          f"val sMAPE {best['val_smape_%']:.2f}%, stage={best['stage']})")
    # any test artefacts sitting in a NON-winner dir are from an earlier
    # batch's internal winner and would be misleading -> park them
    clear_stale_test_eval(root, mode, best["tag"])
    print(f"FINAL: evaluating the best checkpoint on test {len(EVAL_SEEDS)} "
          f"times (sampling seeds {EVAL_SEEDS})...")

    import time as _time
    _train_sec = None
    _curve = best_dir / "loss_curve.json"
    if _curve.exists():
        _train_sec = json.load(open(_curve)).get("train_sec")
    if os.environ.get("CVAE_TIME_FINAL") == "1":
        print("\n[CVAE_TIME_FINAL] fresh timing-retrain of the winner (weights discarded)...")
        _tdir = best_dir / "_timing_retrain"
        train_one(cfg_best, mode, loaders, dims, device, _tdir, EPOCHS, mu, sd)
        _train_sec = json.load(open(_tdir / "loss_curve.json"))["train_sec"]

    _t_test0 = _time.perf_counter()
    per_run, test_tables = generate_runs(
        model, te_loader, device, cfg_best["latent"], mu, sd, X_te, test_cases,
        EVAL_SEEDS, best_dir, "test", "AR6_R10sd_test_pred_cvae")
    _test_wall = _time.perf_counter() - _t_test0
    _n_te = len(test_cases)
    _timing = {
        "what": "cVAE winner: training wall clock + test-set generation wall clock. "
                "Generation is one batched forward pass, so per-case time = wall/cases "
                "(there is no per-case queueing latency to report separately).",
        "config": best["tag"], "device": str(device),
        "n_test_cases": _n_te, "sampling_seeds": EVAL_SEEDS,
        "rpt_train_wallclock_sec": _train_sec,
        "rpt_total_wallclock_sec": round(_test_wall, 2),
        "rpt_sec_per_case_batch": round(_test_wall / (_n_te * len(EVAL_SEEDS)), 5),
        "rpt_sec_per_trajectory": round(_test_wall / (_n_te * len(EVAL_SEEDS)), 5),
        "measured": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    json.dump(_timing, open(best_dir / "TEST_TIMING.json", "w"), indent=2)
    print("[timing]", json.dumps(_timing, indent=2))

    num_keys = [k for k in per_run[0] if k != "seed"
                and isinstance(per_run[0][k], (int, float))]
    metrics = {
        "n_runs": len(per_run),
        "sampling_seeds": EVAL_SEEDS,
        **{k: float(np.mean([d[k] for d in per_run])) for k in num_keys},
        **{f"{k}_sd": (float(np.std([d[k] for d in per_run], ddof=1))
                       if len(per_run) > 1 else 0.0) for k in num_keys},
    }
    # ---- case-level bootstrap on test (each seed + mean of seeds) ----
    # Two different uncertainties, reported side by side: <m>_sd = decoder
    # sampling (across the seed runs); <m>_ci2.5/97.5 = test-set sampling
    # (case bootstrap of the mean-of-seeds per-trajectory table).
    test_boot = None
    if BOOT_B > 0:
        test_boot, mrow = bootstrap_runs(test_tables, EVAL_SEEDS)
        for k in BOOT_METRICS:
            for suf in ("_ci2.5", "_ci97.5", "_boot_sd"):
                metrics[f"{k}{suf}"] = float(mrow[f"{k}{suf}"])
        metrics["bootstrap"] = {"B": BOOT_B, "seed": BOOT_SEED,
                                "n_cases": int(mrow["n_cases"]),
                                "represented_by": ("per-trajectory mean of the sampling-seed runs"
                                                   if len(EVAL_SEEDS) > 1 else
                                                   f"single sampling seed {EVAL_SEEDS[0]}"),
                                "unit": "case id"}
    print("\nTest metrics (mean over runs, _sd = across-run SD, _ci = case bootstrap):")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                      for k, v in metrics.items()}, indent=2))

    json.dump(per_run, open(best_dir / "test_metrics_by_run.json", "w"), indent=2)
    json.dump(metrics, open(best_dir / "test_metrics.json", "w"), indent=2)
    if test_boot is not None:
        with pd.ExcelWriter(best_dir / "bootstrap_test.xlsx", engine="openpyxl") as w:
            test_boot.to_excel(w, index=False, sheet_name="summary")
            bootstrap_run_config(BOOT_B, BOOT_SEED, "mean_of_seeds", int(mrow["n_cases"]),
                                 "test", [("gt", TEST_PATH), ("seeds", str(EVAL_SEEDS))]
                                 ).to_excel(w, index=False, sheet_name="run_config")

    print(f"\nVAL: generating with the same {len(EVAL_SEEDS)} sampling seed(s) ...")
    val_per_run, val_tables = generate_runs(
        model, va_loader, device, cfg_best["latent"], mu, sd, X_va, val_cases,
        EVAL_SEEDS, best_dir, "val", "AR6_R10sd_val10_pred_cvae")
    json.dump(val_per_run, open(best_dir / "val_metrics_by_run.json", "w"), indent=2)
    if BOOT_B > 0:
        val_boot, vrow = bootstrap_runs(val_tables, EVAL_SEEDS)
        print(f"  val MAE ({'mean of seeds' if len(EVAL_SEEDS) > 1 else 'seed ' + str(EVAL_SEEDS[0])}) {vrow['mae']:.4f} "
              f"[{vrow['mae_ci2.5']:.4f}, {vrow['mae_ci97.5']:.4f}]  "
              f"(sweep val_mae at seed {VAL_EVAL_SEED}: {best['val_mae']:.4f})")

        # top-N grid cells, re-generated at the FIXED sweep seed so each
        # reproduces its own sweep val_mae, then bootstrapped PAIRED against
        # the winner: this is the only way to tell a 0.01 gap from noise.
        topn_summary = topn_diffs = None
        topn = results_df.head(max(1, BOOT_TOPN))
        if len(topn) > 1:
            tabs, missing = {}, []
            for _, r in topn.iterrows():
                d = root / r["tag"]
                if not (d / "best_cvae_model.pth").exists():
                    missing.append(r["tag"]); continue
                m_ = CVAE(input_dim, cond_dim, int(r["hidden"]), int(r["latent"]),
                          out_shape).to(device)
                m_.load_state_dict(torch.load(d / "best_cvae_model.pth",
                                              map_location=device))
                torch.manual_seed(VAL_EVAL_SEED)
                p_, t_, k_ = predict(m_, va_loader, device, int(r["latent"]), mu, sd, X_va)
                tab = per_traj_table(p_, t_, k_, val_cases)
                got = eval_from_table(tab)["mae"]
                flag = "" if abs(got - float(r["val_mae"])) < 1e-6 else \
                       f"   <-- WARNING: sweep says {float(r['val_mae']):.6f}"
                print(f"  top-N {r['tag']}: val MAE {got:.6f}{flag}")
                tabs[r["tag"]] = tab
            if missing:
                print(f"  top-N: {len(missing)} cell(s) have no checkpoint, skipped: {missing}")
            if len(tabs) > 1:
                topn_summary, topn_diffs = case_bootstrap(tabs, ref=best["tag"])
                dm = topn_diffs[topn_diffs["metric"] == "mae"]
                print(f"  paired vs winner (B={BOOT_B}):")
                for _, r in dm.iterrows():
                    print(f"    {r['config'][:60]:<60} dMAE {r['mean_diff']:+.4f} "
                          f"[{r['ci2.5_diff']:+.4f}, {r['ci97.5_diff']:+.4f}] "
                          f"{'distinct' if r['excludes_zero'] else 'NOT distinguishable'}")
        with pd.ExcelWriter(best_dir / "bootstrap_val.xlsx", engine="openpyxl") as w:
            val_boot.to_excel(w, index=False, sheet_name="summary")
            if topn_summary is not None:
                topn_summary.to_excel(w, index=False, sheet_name="topn_summary")
                topn_diffs.to_excel(w, index=False, sheet_name="paired_topn")
            bootstrap_run_config(BOOT_B, BOOT_SEED, "mean_of_seeds", int(vrow["n_cases"]),
                                 "val", [("gt", VAL_PATH), ("seeds", str(EVAL_SEEDS)),
                                         ("topn_seed", VAL_EVAL_SEED),
                                         ("topn_ref", best["tag"])]
                                 ).to_excel(w, index=False, sheet_name="run_config")
    json.dump({
        "condition_mode": mode,
        "selection_metric": "checkpoint: val_loss_elbo | config: val_mae",
        "config": {k: (float(best[k]) if k in ("lr", "beta") else int(best[k]))
                   for k in ["lr", "hidden", "latent", "beta"]},
        "standardization": ("all 22 variable rows log1p + per-row z-score; "
                            "train stats"),
        "postprocessing": (f"predictions clipped to >= 0, then rounded to "
                           f"{OUTPUT_DECIMALS} decimals"),
        "standardizer_mu": [float(v) for v in mu],
        "standardizer_sd": [float(v) for v in sd],
        "val_loss": float(best["val_loss"]), "val_metrics": {k.replace("val_", ""): float(best[k]) for k in best.index if str(k).startswith("val_") and k != "val_loss"}, "best_epoch": int(best["best_epoch"]),
        "batch_size": BATCH_SIZE, "epochs": EPOCHS, "seed": SEED,
        "eval_sampling_seeds": EVAL_SEEDS,
        "final_protocol": "single trained model; test generation repeated per sampling seed",
        "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "topn": BOOT_TOPN,
                      "metrics": list(BOOT_METRICS),
                      "design": "paired case-level; test & val: seed runs + mean of seeds; "
                                "val top-N cells paired vs winner at VAL_EVAL_SEED"},
        "epochs_cap": EPOCHS, "patience": PATIENCE,
        "train_path": TRAIN_PATH, "val_path": VAL_PATH, "test_path": TEST_PATH,
        "variable_scope": "core x22 | COFFEE/WITCH x11 supply | GCAM/TIAM x11 demand (masked ELBO)",
        "fingerprint_sheet": FP_SHEET, "fingerprint_path": FINGERPRINT_PATH,
        "embed_model": EMBED_MODEL,
        "sweep": ("two_stage" if args.sweep2 else
                  "full_grid" if args.sweep else "single_config"),
        "sweep_grid": {k: list(v) for k, v in SWEEP_GRID.items()},
        "beta_ref_stage1": BETA_REF if args.sweep2 else None,
        "resume_enabled": bool(args.resume),
        "n_configs_ranked": len(results_df),
        "n_configs_trained_this_run": int((results_df["stage"] != "cached").sum()),
        "n_configs_reused": n_cached,
        "selection_scope": ("full grid incl. cached cells" if args.resume
                            else "configs trained in this run"),
        "run_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, open(best_dir / "training_hyperparameters.json", "w"), indent=2)

    if args.sweep or args.sweep2:
        # timestamped copy (history) + stable name (always the current full grid)
        results_df.to_excel(root / f"sweep_summary_cond{mode}_{stamp}.xlsx",
                            index=False)
        results_df.to_excel(root / f"sweep_summary_cond{mode}_FULL.xlsx",
                            index=False)
        print(f"grid summary -> sweep_summary_cond{mode}_FULL.xlsx "
              f"({len(results_df)} configs)")
    print("saved to:", best_dir)


if __name__ == "__main__":
    main()
