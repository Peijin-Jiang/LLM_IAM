

import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# LOCAL SETUP (no Colab). API keys come from .env or environment.
# ============================================================
_HERE = Path(__file__).resolve().parent


def _load_dotenv(path=None):
    """Minimal .env loader (KEY=VALUE lines); does not override existing env vars."""
    path = path or (_HERE / ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v

_load_dotenv(_HERE / ".env")
_load_dotenv(_HERE.parent / ".env")

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError(
        "Missing OPENAI_API_KEY. Put it in a .env file in the project root (OPENAI_API_KEY=sk-...) "
        "or export it before running."
    )

from openai import OpenAI
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
client = openai_client

# Gemini is optional: only initialized if GEMINI_API_KEY is set.
gemini_client = None
if os.environ.get("GEMINI_API_KEY"):
    try:
        from google import genai
        gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    except ImportError:
        pass

# dataset built by make_datasets/
_MD_OUT = _HERE / "make_datasets" / "output"
_SPLIT = _MD_OUT / "split"

# text-embedding-3-large is used for every embedding in this project (rag_core,
# cvae, knn). The cached vectors in _cache/*.json are 3072-d. The cache key is
# text-only, so embed_texts stamps the model into the cache and refuses a
# mismatch (see the guard there).
EMBED_MODEL = "text-embedding-3-large"
FINGERPRINT_PATH = os.environ.get(
    "IAM_LLM_FINGERPRINT_PATH",
    str(_HERE / "outputs" / "fingerprint_from_papers" /
        "fingerprint_from_papers_comp_evidence_gpt-5.6_20260828_0525_struct.xlsx"),
)

CACHE_DIR = os.environ.get(
    "IAM_LLM_CACHE_DIR",
    str(_HERE / "_cache"),
)
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
SCEN_EMB_CACHE = os.path.join(CACHE_DIR, "cache_scenario_desc_embeddings.json")
FP_EMB_CACHE   = os.path.join(CACHE_DIR, "cache_model_fingerprint_embeddings.json")

TOPK_SCEN  = int(os.environ.get("IAM_LLM_TOPK_SCEN",  "2"))
TOPK_MODEL = int(os.environ.get("IAM_LLM_TOPK_MODEL", "1"))
TOPK_CROSS = int(os.environ.get("IAM_LLM_TOPK_CROSS", "3"))
RUN_STEP3  = False

# Which sheet of model_fingerprint.xlsx to use (recorded in the eval report)
FP_SHEET_NAME = os.environ.get("IAM_LLM_FP_SHEET", "fingerprint")
# Whether the TARGET's scenario description text is given to the LLM
INCLUDE_SCENARIO_DESC = True

# 11 supply variables - fossil totals (parents) removed from the pipeline.
VARS_14 = [
    "Primary Energy|Coal|w/ CCS",
    "Primary Energy|Coal|w/o CCS",
    "Primary Energy|Gas|w/ CCS",
    "Primary Energy|Gas|w/o CCS",
    "Primary Energy|Oil|w/ CCS",
    "Primary Energy|Oil|w/o CCS",
    "Primary Energy|Solar",
    "Primary Energy|Wind",
    "Primary Energy|Hydro",
    "Primary Energy|Nuclear",
    "Primary Energy|Biomass",
]

DEMAND_11 = [   # Final Energy|Electricity excluded from the dataset
    "Final Energy|Liquids", "Final Energy|Gases",
    "Final Energy|Industry", "Final Energy|Transportation",
    "Final Energy|Residential and Commercial", "Final Energy|Hydrogen",
    "Final Energy|Heat", "Final Energy|Solids",
    "Final Energy|Industry|Electricity", "Final Energy|Transportation|Electricity",
    "Final Energy|Residential and Commercial|Electricity",
]


# ── family variable scopes (mixed dataset: 5 both-sides + 2 supply-only + 2 demand-only) ──
SUPPLY_VARS_11 = list(VARS_14)          # the 11 supply variables (captured pre-extension)
CORE_FAMILIES = ["AIM/CGE", "GEM-E3", "IMAGE", "MESSAGEix-GLOBIOM", "REMIND"]
SUPPLY_ONLY_FAMILIES = ["COFFEE", "WITCH"]      # supply data only
DEMAND_ONLY_FAMILIES = ["GCAM", "TIAM"]         # demand data only


def vars_for_family(fam) -> list:
    """Variable scope of a family: generation, output and evaluation all happen
    ONLY on these variables (GT files are already scoped the same way)."""
    fam = str(fam).strip()
    if fam in SUPPLY_ONLY_FAMILIES:
        return list(SUPPLY_VARS_11)
    if fam in DEMAND_ONLY_FAMILIES:
        return list(DEMAND_11)
    return list(VARS_14)                 # both sides (after the VARSET extension)

# IAM_LLM_VARSET=sd22 -> supply+demand task (22 variables, new R10sd dataset)
VARSET = os.environ.get("IAM_LLM_VARSET", "sd22")
if VARSET == "sd22":
    VARS_14 = VARS_14 + DEMAND_11
    print(f"[NOTE] VARSET=sd22: {len(VARS_14)} variables (supply + demand)")
YEARS = [str(y) for y in range(2010, 2110, 10)]  # 2010..2100

"""Set target"""

TARGET = {
    "region": "Countries of centrally-planned Asia; primarily China",          # must exactly match the region value in sheet
    "model_family": "IMAGE",                # must exactly match the model value in sheet
    "scenario": "CO_BAU",     # must exactly match the scenario value in sheet
}

# Training (train70), validation (val10) and test sets. The test set is also written as the subsets sub01..sub05.
BATCH_SHEET_PATH = os.environ.get(
    "IAM_LLM_BATCH_SHEET_PATH",
    str(_SPLIT / "AR6_R10sd_val10_list.xlsx"),
)
def _gt_for_list(list_path: str) -> str:
    """The eval GT that PAIRS with a case list: *_list.xlsx -> *_gt.xlsx.
    Holds for val10_list, test_list and sub01..05_test_list alike."""
    q = Path(list_path)
    return str(q.with_name(q.stem.replace("_list", "_gt") + q.suffix))


GT_PATH = os.environ.get("IAM_LLM_GT_PATH") or _gt_for_list(BATCH_SHEET_PATH)
# embedded neighbour pool: the training set, as for the cVAE and KNN baselines
# (the environment variable below overrides the file)
EXCEL_PATH = os.environ.get(
    "IAM_LLM_TRAIN_GT_PATH",
    str(_SPLIT / "AR6_R10sd_train70_ground_truth.xlsx"),
)
EXCEL_SHEET = int(os.environ.get("IAM_LLM_EXCEL_SHEET", "0"))

MODE = "batch"  # "single" or "batch"
# Paths
OUTPUTS_DIR = str(_HERE / "outputs")
Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
# Single-mode outputs (distinct names: OUT_XLSX is reused later for the eval report)
OUT_SINGLE_JSON = os.path.join(OUTPUTS_DIR, "single_target_output.json")
OUT_SINGLE_XLSX = os.path.join(OUTPUTS_DIR, "single_target_output.xlsx")
OUT_JSON = OUT_SINGLE_JSON
OUT_BATCH_CSV = os.path.join(OUTPUTS_DIR, "batch_pred.csv")  # placeholder; replaced by the RUN_TAG name below

OUTPUT_DECIMALS = int(os.environ.get("IAM_LLM_OUTPUT_DECIMALS", "2"))


def round_year_cols(df, years=None, nd=None):
    """Round the year columns of a prediction frame to `nd` decimals, in place.

    PREDICTIONS ONLY - ground truth frames are never passed through this.
    """
    import re as _re
    nd = OUTPUT_DECIMALS if nd is None else nd
    if nd is None or nd < 0:
        return df
    if years is None:
        years = [c for c in df.columns
                 if _re.fullmatch(r"(19|20|21)\d\d", str(c).strip())]
    for y in years:
        df[y] = pd.to_numeric(df[y], errors="coerce").round(nd)
    return df

# Batch config (env-overridable; lower concurrency if you hit 429
# spend-based rate limits, e.g. IAM_LLM_MAX_CONCURRENCY=4).
MAX_CONCURRENCY = int(os.environ.get("IAM_LLM_MAX_CONCURRENCY", "60"))
RETRY_MAX = int(os.environ.get("IAM_LLM_RETRY_MAX", "16"))
RETRY_BACKOFF_BASE = 1.8

"""function define"""

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _norm_col(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"\s+", "_", c)
    c = c.replace("-", "_")
    return c

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_norm_col(c) for c in df.columns]
    return df

def find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_str(x):
    return "" if pd.isna(x) else str(x)

def embed_texts(texts, cache: dict, prefix: str, batch_size: int = 128):
    """
    Embed texts with caching (JSON).
    Cache key = f"{prefix}:{text}" - deliberately text-only, so an edited
    fingerprint or description re-embeds automatically. It does NOT include
    EMBED_MODEL, so a cache built with another model would be reused at a 100%
    hit rate and the run would quietly use the wrong vectors. The stamp below
    turns that silent failure into a loud one.
    """
    _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    stamped = cache.get("__embed_model__")
    if stamped is None:
        cache["__embed_model__"] = EMBED_MODEL
    elif stamped != EMBED_MODEL:
        raise RuntimeError(
            f"embedding cache was built with {stamped!r} but EMBED_MODEL is "
            f"{EMBED_MODEL!r}. The cache key does not encode the model, so the "
            f"old vectors would be reused silently. Point IAM_LLM_CACHE_DIR at "
            f"a fresh directory (or move the *.json out of {CACHE_DIR}) before "
            f"switching embedding models.")
    keys = [f"{prefix}:{t}" for t in texts]
    missing_texts = []
    missing_keys = []
    for t, k in zip(texts, keys):
        if k not in cache:
            missing_texts.append(t)
            missing_keys.append(k)

    if missing_texts:
        for i in tqdm(range(0, len(missing_texts), batch_size), desc=f"Embedding {prefix}"):
            batch = missing_texts[i:i+batch_size]
            resp = _client.embeddings.create(model=EMBED_MODEL, input=batch)
            for emb_obj, k in zip(resp.data, missing_keys[i:i+batch_size]):
                cache[k] = emb_obj.embedding

    dims = {len(cache[k]) for k in keys}
    if len(dims) > 1:
        raise RuntimeError(
            f"{prefix} embeddings have mixed dimensions {sorted(dims)} - the "
            f"cache in {CACHE_DIR} mixes two embedding models; start from a "
            f"clean IAM_LLM_CACHE_DIR.")
    embs = np.array([cache[k] for k in keys], dtype=np.float32)
    return embs

def cosine_rank(query_emb: np.ndarray, cand_embs: np.ndarray):
    sims = cosine_similarity(query_emb.reshape(1, -1), cand_embs).ravel()
    order = np.argsort(-sims)
    return order, sims

def build_fp_text(row: pd.Series) -> str:
    pref = safe_str(row.get('energy_system_preference', ''))
    return f"{FP_PREF_LABEL}: {pref}".strip()

"""Load data"""

# Read scenario data
df = pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_SHEET)
df = normalize_columns(df)

# Read model fingerprint data
fp = pd.read_excel(FINGERPRINT_PATH, sheet_name=FP_SHEET_NAME)
fp = normalize_columns(fp)

col_model      = find_col(df, ["model"])
col_scenario   = find_col(df, ["scenario"])
col_region     = find_col(df, ["region"])
col_variable   = find_col(df, ["variable"])
col_unit       = find_col(df, ["unit"])
col_family     = find_col(df, ["model_family"])
col_desc       = find_col(df, ["scenario_description"])
col_c_category = find_col(df, ["category_c"])

# Eval targets table
df_gt = pd.read_excel(GT_PATH, sheet_name=EXCEL_SHEET)
scenario2desc = dict(zip(df[col_scenario], df[col_desc]))
family2model = dict(zip(df[col_family], df[col_model]))

# Supplement scenario descriptions from the FULL (unsplit) dataset, so test
# scenarios that never appear in train still get a description.
# (Descriptions are model INPUT, not ground truth -> not leakage.)
_cleaned = sorted(_MD_OUT.glob("AR6_R10_sd_mixed_*.xlsx"))
DESC_SOURCE_PATH = os.environ.get(
    "IAM_LLM_DESC_SOURCE_PATH",
    str(_cleaned[-1]) if _cleaned else "",
)
if not DESC_SOURCE_PATH:
    raise FileNotFoundError(
        "No AR6_R10_sd_mixed_*.xlsx found in make_datasets/output/ "
        "and IAM_LLM_DESC_SOURCE_PATH is not set."
    )
_desc_supplement = {}
if os.path.exists(DESC_SOURCE_PATH):
    _dxl = pd.ExcelFile(DESC_SOURCE_PATH)
    _dsheet = 0
    for _sn in _dxl.sheet_names:   # file may contain a stray empty sheet
        _probe = pd.read_excel(_dxl, sheet_name=_sn, nrows=1)
        if "scenario" in [str(c).strip() for c in _probe.columns]:
            _dsheet = _sn
            break
    _ddf = normalize_columns(pd.read_excel(_dxl, sheet_name=_dsheet))
    _ds = find_col(_ddf, ["scenario"])
    _dd = find_col(_ddf, ["scenario_description"])
    if _ds and _dd:
        for _s, _d in zip(_ddf[_ds], _ddf[_dd]):
            _s, _d = str(_s).strip(), safe_str(_d).strip()
            if _s and _d:
                _desc_supplement.setdefault(_s, _d)
    del _ddf
for _s, _d in _desc_supplement.items():
    scenario2desc.setdefault(_s, _d)

df_gt[col_desc] = df_gt[col_scenario].map(scenario2desc)
df_gt[col_model] = df_gt[col_scenario].map(family2model)

# Neighbour retrieval pool: the training set only. If a test case has no
# same-scenario neighbour in the training set, that neighbour category is absent.
INCLUDE_EVAL_GT_IN_NEIGHBOR_POOL = False
if INCLUDE_EVAL_GT_IN_NEIGHBOR_POOL:
    df = pd.concat(
       [df, df_gt.reindex(columns=df.columns)],
       axis=0
    )

req = {
    "model": col_model,
    "scenario": col_scenario,
    "region": col_region,
    "variable": col_variable,
    "unit": col_unit,
    "model_family": col_family,
    "scenario_description": col_desc,
    "category_c": col_c_category,
}
missing = [k for k, v in req.items() if v is None]
if missing:
    raise KeyError(
        f"Missing columns in Excel sheet '{EXCEL_SHEET}': {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

# Convert year columns if read as int
for y in YEARS:
    yi = int(y)
    if y not in df.columns and yi in df.columns:
        df = df.rename(columns={yi: y})

FP_PREF_LABEL_EXPECTED = "Energy System Preference"

fp_col_model = find_col(fp, ["model_family"])
fp_col_pref  = find_col(fp, ["energy_system_preference"])
if fp_col_pref is None:
    raise KeyError(
        f"Fingerprint sheet is missing the column: {FP_PREF_LABEL_EXPECTED}. "
        f"Found: {list(fp.columns)}")

FP_PREF_LABEL = FP_PREF_LABEL_EXPECTED
# Variable scope enumerated in prompts (demand side only in the sd22 task).
_SUPPLY_SCOPE = ("coal w/ CCS, coal w/o CCS, gas w/ CCS, gas w/o CCS, oil w/ CCS, "
                 "oil w/o CCS, solar, wind, hydro, nuclear, and biomass")
_DEMAND_SCOPE = ("final-energy liquids, gases, solids, heat, hydrogen; "
                 "total final energy in industry, transportation, and residential and "
                 "commercial; and the electricity level within each of these three sectors")
FP_SCOPE = (_SUPPLY_SCOPE + ("; and on the demand side: " + _DEMAND_SCOPE
                             if VARSET == "sd22" else ""))


def fp_scope_for(vars_out) -> str:
    """FP_SCOPE restricted to the sides actually present in vars_out
    (supply-only / demand-only families get only their own side)."""
    vs = {str(v).strip() for v in vars_out}
    has_sup = any(v in vs for v in SUPPLY_VARS_11)
    has_dem = any(v in vs for v in DEMAND_11)
    if has_sup and has_dem:
        return _SUPPLY_SCOPE + "; and on the demand side: " + _DEMAND_SCOPE
    if has_dem:
        return _DEMAND_SCOPE
    return _SUPPLY_SCOPE

if fp_col_model is None:
    raise KeyError(
        f"Fingerprint sheet missing 'model_family' column. Available: {list(fp.columns)}"
    )

# Build fp_text (must exist before you embed fingerprints)
fp["fp_text"] = fp.apply(build_fp_text, axis=1)
FP_FAMILY_COL = "model_family"
if FP_FAMILY_COL not in fp.columns:
    raise KeyError(f"fp missing '{FP_FAMILY_COL}'. Available: {list(fp.columns)}")
if "fp_text" not in fp.columns:
    raise KeyError(
        "fp missing 'fp_text'. Did you run fp['fp_text'] = fp.apply(build_fp_text, axis=1)?"
    )

_fp2 = fp[[FP_FAMILY_COL, "fp_text"]].copy()
_fp2[FP_FAMILY_COL] = _fp2[FP_FAMILY_COL].fillna("").astype(str).str.strip()
_fp2["fp_text"]     = _fp2["fp_text"].fillna("").astype(str)

model_family2fp_text = dict(zip(_fp2[FP_FAMILY_COL], _fp2["fp_text"]))

# Ensure TARGET family exists
tf = str(TARGET["model_family"]).strip()
if tf not in model_family2fp_text or not str(model_family2fp_text[tf]).strip():
    raise KeyError(
        f"TARGET model_family '{tf}' not found or has empty fp_text in model_family2fp_text."
    )
print("model_family2fp_text size:", len(model_family2fp_text))
print("TARGET fp_text preview:", str(model_family2fp_text[tf])[:200], "...")

df_gt.shape

"""Embeddings (with caches)"""

# JSON caches of the embeddings (the cache key includes the full text, so edited
# fingerprints or descriptions are embedded again).
scen_cache = load_json(SCEN_EMB_CACHE)
fp_cache   = load_json(FP_EMB_CACHE)

df_desc_global = df[[col_scenario, col_desc]].copy()
df_desc_global[col_scenario] = df_desc_global[col_scenario].astype(str).str.strip()
df_desc_global[col_desc]     = df_desc_global[col_desc].fillna("").astype(str)

df_desc_global = (
    df_desc_global
    .sort_values([col_scenario])
    .groupby(col_scenario, as_index=False)
    .agg({col_desc: lambda x: next((t for t in x if str(t).strip()), "")})
)

scenario2desc = dict(zip(df_desc_global[col_scenario], df_desc_global[col_desc]))
# Re-apply full-dataset description supplement (test-only scenarios etc.)
for _s, _d in _desc_supplement.items():
    if not str(scenario2desc.get(_s, "")).strip():
        scenario2desc[_s] = _d
print("scenario2desc size:", len(scenario2desc))


all_desc = sorted({str(v).strip() for v in scenario2desc.values() if str(v).strip()})
desc_embs = embed_texts(all_desc, scen_cache, prefix="SCEN_DESC")
save_json(scen_cache, SCEN_EMB_CACHE)
desc2emb = {t: desc_embs[i] for i, t in enumerate(all_desc)}


FP_FAMILY_COL = "model_family"
fp_families = fp[FP_FAMILY_COL].fillna("").astype(str).str.strip().tolist()
fp_texts    = fp["fp_text"].fillna("").astype(str).tolist()

fp_embs = embed_texts(fp_texts, fp_cache, prefix="MODEL_FP")
save_json(fp_cache, FP_EMB_CACHE)


family2fp = {fam: emb for fam, emb in zip(fp_families, fp_embs) if fam}
print("family2fp keys:", list(family2fp.keys()))

def infer_family_for_model(model_name: str) -> str | None:
    fams = (
        df[df[col_model].astype(str) == str(model_name)][col_family]
        .dropna().astype(str).unique().tolist()
    )
    return fams[0] if fams else None

"""Retrieval"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


YEARS = [str(y) for y in range(2010, 2101, 10)]
VARS_14_NORM = [str(v).strip() for v in VARS_14]
VARS_14_SET  = set(VARS_14_NORM)

def ensure_year_cols(df: pd.DataFrame, years=YEARS) -> pd.DataFrame:
    df = df.copy()
    for y in years:
        yi = int(y)
        if y not in df.columns and yi in df.columns:
            df = df.rename(columns={yi: y})
    return df

df = ensure_year_cols(df, YEARS)

def get_scenario_desc(scenario: str) -> str | None:
    d = scenario2desc.get(str(scenario))
    if d is None:
        return None
    d = str(d).strip()
    return d if d else None

# Step1: rank scenarios by desc similarity within same region+family
def _gemini_client():
    """Module-level cached client, so the rerank calls do not depend on the one
    nested inside synthesize_one_target."""
    global gemini_client
    if globals().get("gemini_client") is not None:
        return gemini_client
    if not os.environ.get("GEMINI_API_KEY", ""):
        raise ValueError("Missing GEMINI_API_KEY in env.")
    from google import genai
    gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return gemini_client


def llm_rank_json(prompt: str, n_items: int) -> list[int] | None:
    """Ask the LLM for a ranking of n_items candidates; returns 0-based indices
    best-first, or None if the call fails (callers then keep the embedding
    order, so a failed rerank call does not stop the run)."""
    schema = {
        "type": "object",
        "properties": {
            "ranking": {
                "type": "array",
                "items": {"type": "integer"},
                "description": ("Candidate numbers, best first, most relevant "
                                "to the TARGET. Every candidate exactly once."),
            }
        },
        "required": ["ranking"],
    }
    try:
        if PROVIDER == "gemini":
            from google.genai import types
            gc = _gemini_client()
            cfg = types.GenerateContentConfig(
                thinking_config=(types.ThinkingConfig(thinking_level="minimal")
                                 if GEMINI_MODEL_NAME.startswith("gemini-3")
                                 else types.ThinkingConfig(thinking_budget=0)),
                response_mime_type="application/json",
                response_json_schema=schema,
                temperature=TEMPERATURE,
                max_output_tokens=2000,
            )
            resp = gc.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents="Return JSON only.\n\n" + prompt, config=cfg)
            um = getattr(resp, "usage_metadata", None)
            if um is not None:
                _track_usage(getattr(um, "prompt_token_count", 0),
                             getattr(um, "candidates_token_count", 0),
                             getattr(um, "thoughts_token_count", 0))
            out = extract_json((resp.text or "").strip())
        else:
            from openai import OpenAI
            resp = OpenAI().chat.completions.create(
                model=MODEL_NAME, temperature=TEMPERATURE,
                messages=[{"role": "system", "content": "Return JSON only."},
                          {"role": "user", "content": prompt}])
            if getattr(resp, "usage", None):
                _track_usage(resp.usage.prompt_tokens, resp.usage.completion_tokens)
            out = extract_json(resp.choices[0].message.content)
    except Exception as e:
        print(f"[RERANK] call failed ({e}); keeping embedding order")
        return None

    raw = out.get("ranking") if isinstance(out, dict) else None
    if not isinstance(raw, list):
        return None
    # Be forgiving: drop anything out of range or duplicated, then append the
    # candidates the model omitted, in their original embedding order. The
    # result is always a full permutation.
    seen, order = set(), []
    for v in raw:
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n_items and i not in seen:
            seen.add(i); order.append(i)
    order += [i for i in range(n_items) if i not in seen]
    return order


_SCEN_FEAT_CACHE: dict[str, dict] = {}

_KW_FLAGS = [
    ("baseline",  r"baseline|no climate|without climate|reference"),
    ("ndc",       r"\bndc|indc"),
    ("netzero",   r"net[- ]zero"),
    ("cprice",    r"carbon (price|tax)"),
    ("cbudget",   r"carbon budget|cumulative (?:co2 |carbon )?emissions|\bcb[-_ ]?\d+|\d[\d,.]*\s*gt\s*co2"),
    ("renew",     r"renewable|high re\b"),
    ("demand",    r"demand[- ]side|lifestyle|behavio"),
    ("effgap",    r"efficiency"),
]


def scenario_features(scenario: str) -> dict:
    """Keyword features of a scenario description, cached per scenario.

    Deliberately plain regex over the description text - no model in the loop,
    so the ranking is reproducible run to run."""
    key = str(scenario).strip()
    if key in _SCEN_FEAT_CACHE:
        return _SCEN_FEAT_CACHE[key]
    t = (get_scenario_desc(key) or "").lower()
    f = {}
    b = re.findall(r"(\d[\d,\.]*)\s*gtco", t)
    f["budget"] = float(b[0].replace(",", "")) if b else float("nan")
    tm = [float(x) for x in re.findall(r"(\d(?:\.\d)?)\s*°?\s*c\b", t)
          if 1.0 <= float(x) <= 6.0]
    f["temp"] = min(tm) if tm else float("nan")
    sp_ = re.findall(r"ssp(\d)", t)
    f["ssp"] = float(sp_[0]) if sp_ else float("nan")
    for name, pat in _KW_FLAGS:
        f[name] = 1.0 if re.search(pat, t) else 0.0
    _SCEN_FEAT_CACHE[key] = f
    return f


def keyword_distance(a: dict, b: dict) -> float:
    """Mean per-feature distance, averaged over the features both sides can
    speak to. A numeric feature present on one side and missing on the other
    counts as a full mismatch; missing on both is simply not counted, so
    scenarios that state nothing do not get an artificial head start."""
    tot = 0.0
    n = 0
    for k, x in a.items():
        y = b.get(k, float("nan"))
        if k in ("budget", "temp", "ssp"):
            ok_x, ok_y = np.isfinite(x), np.isfinite(y)
            if ok_x and ok_y:
                # budgets span 200-8000 GtCO2, so compare them on a log scale
                tot += (abs(np.log1p(x) - np.log1p(y)) / 2.0 if k == "budget"
                        else abs(x - y) / 2.0)
                n += 1
            elif ok_x != ok_y:
                tot += 1.0
                n += 1
        else:
            tot += abs(x - y)
            n += 1
    return tot / max(n, 1)


def rerank_scenarios_keyword(target_scenario: str, ranked_scen: list) -> list:
    """Deterministic STEP1 re-rank. ranked_scen arrives in embedding order and
    carries (scenario, cosine); in hybrid mode that cosine is blended back in
    with weight RERANK_BLEND."""
    pool = ranked_scen[:RERANK_POOL_SCEN]
    if len(pool) <= 1:
        return ranked_scen
    qf = scenario_features(target_scenario)
    scored = []
    for scen, sim in pool:
        d = keyword_distance(qf, scenario_features(scen))
        if RERANK_MODE == "hybrid":
            d -= RERANK_BLEND * float(sim)
        scored.append((d, scen, sim))
    scored.sort(key=lambda x: x[0])
    return [(sc, sim) for _d, sc, sim in scored] + ranked_scen[RERANK_POOL_SCEN:]


def rerank_scenarios(target_scenario: str, ranked_scen: list) -> list:
    """Reorder STEP1 candidates (same region+family, different scenarios) by
    scenario description alone."""
    pool = ranked_scen[:RERANK_POOL_SCEN]
    if len(pool) <= 1:
        return ranked_scen
    tdesc = get_scenario_desc(target_scenario) or ""
    lines = []
    for i, (scen, _sim) in enumerate(pool):
        lines.append(f"[{i}] scenario: {scen}\n    description: "
                     f"{(get_scenario_desc(scen) or '').strip()}")
    prompt = f"""
You are ranking candidate reference scenarios for an integrated assessment model (IAM) synthetic data generation task.

The TARGET and every candidate come from the SAME model and the SAME region, so they differ only in the scenario. The most relevant candidate is the one whose energy trajectories are expected to be most similar to the TARGET’s because its mitigation stringency and pathway characteristics match most closely, including the carbon budget, temperature goal, policy timing, overshoot treatment, and technology constraints.

TARGET scenario: {target_scenario}
TARGET description: {tdesc}

CANDIDATES:
{chr(10).join(lines)}

Rank ALL {len(pool)} candidates from most to least useful as references for the TARGET based on how closely their mitigation pathways match the TARGET across these dimensions. Output every candidate number exactly once.

OUTPUT JSON FORMAT (exactly):
{{"ranking": [<candidate number>, ...]}}
""".strip()
    order = llm_rank_json(prompt, len(pool))
    if order is None:
        return ranked_scen
    return [pool[i] for i in order] + ranked_scen[RERANK_POOL_SCEN:]


def rerank_models(target_family: str, step2_models: list) -> list:
    """Reorder STEP2 candidates (same region+scenario, different family) by
    model fingerprint alone."""
    pool = step2_models[:RERANK_POOL_MODEL]
    if len(pool) <= 1:
        return step2_models
    tfp = get_model_family_fp_text(str(target_family))
    lines = [f"[{i}] model_family: {fam}\n    fingerprint: {get_model_family_fp_text(str(fam))}"
             for i, (_m, fam, _s) in enumerate(pool)]
    prompt = f"""
You are ranking candidate reference models for an integrated assessment model (IAM) emulation task.

The TARGET and every candidate ran the SAME scenario in the SAME region, so they differ only in the model. The candidate whose energy trajectories will look most like the TARGET's is the one whose structural behaviour matches best - similar technology preferences across the energy system, that is, sitting on the same side of the other models on the same variables.

A fingerprint may read "Not Available", or may simply not mention a variable. Either means no cross-model evidence exists for it, never that the model is average or unremarkable. Compare the models only on the variables both fingerprints actually state. Two fingerprints that both read "Not Available" are not thereby similar - do not read agreement into a missing value; rank such a candidate neutrally rather than treating the absence as a match.

TARGET model_family: {target_family}
TARGET fingerprint: {tfp}

CANDIDATES:
{chr(10).join(lines)}

Rank ALL {len(pool)} candidates from most to least useful as a reference for the TARGET, by how closely their structural behaviour matches. Output every candidate number exactly once.

OUTPUT JSON FORMAT (exactly):
{{"ranking": [<candidate number>, ...]}}
""".strip()
    order = llm_rank_json(prompt, len(pool))
    if order is None:
        return step2_models
    return [pool[i] for i in order] + step2_models[RERANK_POOL_MODEL:]


def step1_rank_scenarios_region_family(target_region: str, target_family: str, target_scenario: str):
    tdesc = get_scenario_desc(target_scenario)
    if not tdesc or tdesc not in desc2emb:
        raise ValueError(f"Target scenario description missing/not embedded: {target_scenario}")

    q = desc2emb[tdesc].reshape(1, -1)

    cand_s = (
        df[
            (df[col_region].astype(str) == str(target_region)) &
            (df[col_family].astype(str) == str(target_family))
        ][col_scenario]
        .astype(str).unique().tolist()
    )
    keep = []
    for s in cand_s:
        if s == str(target_scenario):
            continue
        d = get_scenario_desc(s)
        if not d or d not in desc2emb:
            continue
        keep.append((str(s), d))

    if not keep:
        return []

    cand_emb = np.vstack([desc2emb[d] for _, d in keep])
    sims = cosine_similarity(q, cand_emb).ravel()
    order = np.argsort(-sims)
    print("Step 1: Target Family ({}) Target Region ({}) Target Scenario ({})".format(target_family, target_region, target_scenario), [(keep[i][0], float(sims[i])) for i in order[: 3]])
    return [(keep[i][0], float(sims[i])) for i in order]


def extract_single_model_18vars(region: str, family: str, model: str, scenario: str,
                               step_tag: str, similarity_score: float,
                               years=YEARS) -> pd.DataFrame:
    sub = df[
        (df[col_region].astype(str) == str(region)) &
        (df[col_model].astype(str) == str(model)) &
        (df[col_scenario].astype(str) == str(scenario)) &
        (df[col_variable].astype(str).str.strip().isin(VARS_14_SET))
    ].copy()

    sub["_var"] = sub[col_variable].astype(str).str.strip()

    for y in years:
        if y in sub.columns:
            sub[y] = pd.to_numeric(sub[y], errors="coerce")

    if not sub.empty:
        sub = sub.sort_values(["_var"]).drop_duplicates(subset=["_var"], keep="first")

    lookup = {str(r["_var"]): r for _, r in sub.iterrows()}

    rows = []
    for v in VARS_14_NORM:
        r0 = lookup.get(v)
        out = {
            "step": step_tag,
            "similarity": float(similarity_score),
            "region": str(region),
            "model_family": str(family),
            "model": str(model),
            "scenario": str(scenario),
            "category_c": None,
            "variable": str(v),
            "unit": None
        }

        if r0 is not None:
            # category_c (source column may be category_c or category_C)
            cat = r0.get(col_c_category, None)
            if isinstance(cat, pd.Series):
                cat = cat.iloc[0] if not cat.empty else None
            out["category_c"] = None if (cat is None or pd.isna(cat)) else str(cat)

            # unit
            unit = r0.get(col_unit, None)
            if isinstance(unit, pd.Series):
                unit = unit.iloc[0] if not unit.empty else None
            out["unit"] = None if (unit is None or pd.isna(unit)) else unit

            # years
            for y in years:
                if y in sub.columns:
                    val = r0.get(y, None)
                else:
                    yi = int(y)
                    val = r0.get(yi, None) if yi in sub.columns else None
                out[y] = None if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val)
        else:
            for y in years:
                out[y] = None

        rows.append(out)

    return pd.DataFrame(rows)

def extract_family_models_for_scenario_limited(region: str, family: str, scenario: str,
                                               step_tag: str, similarity_score: float,
                                               remaining_budget: int):
    models = (
        df[
            (df[col_region].astype(str) == str(region)) &
            (df[col_family].astype(str) == str(family)) &
            (df[col_scenario].astype(str) == str(scenario))
        ][col_model]
        .astype(str).unique().tolist()
    )

    tables = []
    used = 0
    for m in models:
        if used >= remaining_budget:
            break
        tables.append(
            extract_single_model_18vars(
                region=region,
                family=family,
                model=str(m),
                scenario=str(scenario),
                step_tag=step_tag,
                similarity_score=similarity_score
            )
        )
        used += 1

    out_df = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return out_df, used

def allowed_neighbor_families(target_family: str) -> set:
    """STEP2 (same-scenario / different-family) candidate pool, per the
    mixed-dataset rule:
      - core-5 target                 -> ONLY the other core-5 families
      - supply-only target (COFFEE/WITCH) -> the other supply-only family + the core-5
      - demand-only target (GCAM/TIAM)    -> the other demand-only family + the core-5
    When a supply-/demand-only target borrows a core-5 neighbor, the evidence
    builder already exposes ONLY the target-scope variables of that neighbor
    (vars_out is the target family's scope), so the borrowed neighbor is seen
    through the relevant variables only."""
    fam = str(target_family).strip()
    if fam in SUPPLY_ONLY_FAMILIES:
        return set(CORE_FAMILIES) | (set(SUPPLY_ONLY_FAMILIES) - {fam})
    if fam in DEMAND_ONLY_FAMILIES:
        return set(CORE_FAMILIES) | (set(DEMAND_ONLY_FAMILIES) - {fam})
    return set(CORE_FAMILIES) - {fam}


# Step2: choose top-k model×scenario using family similarity (family2fp)
def topk_models_same_region_scenario_by_family_similarity(
    target_region: str,
    target_scenario: str,
    target_family: str,
    k_models: int = 3,
    exclude_target_family: bool = True
):
    if target_family not in family2fp:
        raise ValueError(f"target_family not in family2fp: {target_family}")

    sub = df[
        (df[col_region].astype(str) == str(target_region)) &
        (df[col_scenario].astype(str) == str(target_scenario))
    ][[col_model, col_family]].dropna().copy()

    if sub.empty:
        return []

    # model -> family (first)
    m2f = {}
    for m, f in zip(sub[col_model].astype(str).tolist(), sub[col_family].astype(str).tolist()):
        if m not in m2f:
            m2f[str(m)] = str(f)

    q = family2fp[target_family].reshape(1, -1)

    allowed = allowed_neighbor_families(target_family)
    fam2sim = {}
    for fam in sorted(set(m2f.values())):
        if exclude_target_family and fam == str(target_family):
            continue
        if fam not in allowed:      # family-group pool restriction (see helper)
            continue
        if fam not in family2fp:
            continue
        fam2sim[fam] = float(cosine_similarity(q, family2fp[fam].reshape(1, -1))[0, 0])

    if not fam2sim:
        return []

    scored = [(m, fam, fam2sim[fam]) for m, fam in m2f.items() if fam in fam2sim]
    scored.sort(key=lambda x: x[2], reverse=True)
    print("Step 2: Target Family ({}) Target Region ({}) Target Scenario ({})".format(target_family, target_region, target_scenario), scored[: 3])
    return scored[:k_models]  # (model, family, fam_sim)

def step3_cross_with_budget(
    target_region: str,
    step1_used_scenarios: list[tuple[str, float]],     # [(scenario, scen_sim)] in order
    step2_models: list[tuple[str, str, float]],        # [(model, family, fam_sim)]
    topk_cross: int
):
    # unique families from step2, keep best similarity per family
    fam_best = {}
    for _, fam, fam_sim in step2_models:
        fam = str(fam)
        fam_best[fam] = max(fam_best.get(fam, -1e9), float(fam_sim))

    remaining = int(topk_cross)
    tables = []

    if remaining <= 0 or not fam_best or not step1_used_scenarios:
        return pd.DataFrame()

    for scen, scen_sim in step1_used_scenarios:
        if remaining <= 0:
            break
        scen = str(scen)
        scen_sim = float(scen_sim)

        for fam, fam_sim in sorted(fam_best.items(), key=lambda x: x[1], reverse=True):
            if remaining <= 0:
                break

            exists = df[
                (df[col_region].astype(str) == str(target_region)) &
                (df[col_family].astype(str) == str(fam)) &
                (df[col_scenario].astype(str) == str(scen))
            ]
            if exists.empty:
                continue

            combo_sim = 0.5 * float(fam_sim) + 0.5 * float(scen_sim)

            block_df, used = extract_family_models_for_scenario_limited(
                region=target_region,
                family=fam,
                scenario=scen,
                step_tag="STEP3_CROSS",
                similarity_score=combo_sim,
                remaining_budget=remaining
            )

            if not block_df.empty and used > 0:
                tables.append(block_df)
                remaining -= used

    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

def assign_combo_id_sequential(df_in: pd.DataFrame, start_id: int):
    """
    Returns (df_out, next_id)
    """
    if df_in is None or df_in.empty:
        return df_in, start_id

    df_out = df_in.copy()

    combos = df_out[["model", "scenario"]].drop_duplicates(keep="first").reset_index(drop=True)
    combos["combo_id"] = range(start_id, start_id + len(combos))

    df_out = df_out.merge(combos, on=["model", "scenario"], how="left")

    cols = ["combo_id"] + [c for c in df_out.columns if c != "combo_id"]
    df_out = df_out[cols]

    next_id = start_id + len(combos)
    return df_out, next_id

def run_retrieval_with_stepwise_ids(
    target_region: str,
    target_family: str,
    target_scenario: str,
    topk_scen: int,
    topk_model: int,
    run_step3: bool,
    topk_cross: int,
    exclude_target_family_in_step2: bool = True
):
    # ----- Step1 (budgeted by combos) -----
    ranked_scen = step1_rank_scenarios_region_family(target_region, target_family, target_scenario)
    if RERANK_MODE == "llm":
        ranked_scen = rerank_scenarios(target_scenario, ranked_scen)
    elif RERANK_MODE in ("keyword", "hybrid"):
        ranked_scen = rerank_scenarios_keyword(target_scenario, ranked_scen)

    remaining = int(topk_scen)
    step1_tables = []
    step1_used_scenarios = []  # [(scenario, sim)] ONLY those used (in order)

    for scen, sim in ranked_scen:
        if remaining <= 0:
            break

        block_df, used = extract_family_models_for_scenario_limited(
            region=target_region,
            family=target_family,
            scenario=scen,
            step_tag="STEP1_SCEN",
            similarity_score=float(sim),
            remaining_budget=remaining
        )

        if not block_df.empty and used > 0:
            step1_tables.append(block_df)
            step1_used_scenarios.append((str(scen), float(sim)))
            remaining -= used

    step1_df = pd.concat(step1_tables, ignore_index=True) if step1_tables else pd.DataFrame()

    # ----- Step2 (budgeted by combos directly) -----
    # With RERANK on, pull a wider pool first so the LLM has something to choose
    # between - at the default topk_model=1 the embedding ranking is otherwise
    # the only thing deciding which model family is shown.
    step2_models = topk_models_same_region_scenario_by_family_similarity(
        target_region=target_region,
        target_scenario=target_scenario,
        target_family=target_family,
        k_models=(max(int(topk_model), RERANK_POOL_MODEL)
                  if RERANK_MODE == "llm" else int(topk_model)),
        exclude_target_family=exclude_target_family_in_step2
    )
    if RERANK_MODE == "llm":
        step2_models = rerank_models(target_family, step2_models)[:int(topk_model)]

    step2_tables = []
    for m, fam, sim in step2_models:
        step2_tables.append(
            extract_single_model_18vars(
                region=target_region,
                family=fam,
                model=m,
                scenario=target_scenario,
                step_tag="STEP2_MODEL",
                similarity_score=float(sim)
            )
        )
    step2_df = pd.concat(step2_tables, ignore_index=True) if step2_tables else pd.DataFrame()

    # ----- Step3 (optional) -----
    if run_step3:
        step3_df = step3_cross_with_budget(
            target_region=target_region,
            step1_used_scenarios=step1_used_scenarios,
            step2_models=step2_models,
            topk_cross=int(topk_cross)
        )
    else:
        step3_df = pd.DataFrame()

    next_id = 1
    step1_df, next_id = assign_combo_id_sequential(step1_df, next_id)
    step2_df, next_id = assign_combo_id_sequential(step2_df, next_id)
    step3_df, next_id = assign_combo_id_sequential(step3_df, next_id)

    return step1_df, step2_df, step3_df, next_id


def compute_steps_for_target(target: dict) -> dict:
    """
    Build STEP1_DF / STEP2_DF / STEP3_DF for a given target combo.
    Requires globals: TOPK_SCEN, TOPK_MODEL, RUN_STEP3, TOPK_CROSS
    """
    step1_df, step2_df, step3_df, _ = run_retrieval_with_stepwise_ids(
        target_region=target["region"],
        target_family=target["model_family"],
        target_scenario=target["scenario"],
        topk_scen=TOPK_SCEN,
        topk_model=TOPK_MODEL,
        run_step3=RUN_STEP3,
        topk_cross=TOPK_CROSS,
        exclude_target_family_in_step2=True
    )
    return {"STEP1_DF": step1_df, "STEP2_DF": step2_df, "STEP3_DF": step3_df}


"""# Executor"""

import os
import json
import re
import ast
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from openai import OpenAI

# ============================================================
# CONFIG (HARD MODE SWITCHES)
# ============================================================
# Default LLM: Gemini 3.5 Flash-Lite. Override per-run without editing this file:
#   IAM_LLM_PROVIDER=openai IAM_LLM_MODEL=gpt-4.1-mini python3 rag_core.py
PROVIDER = os.environ.get("IAM_LLM_PROVIDER", "gemini")   # "openai" | "gemini"

# ---- OpenAI models ----
MODEL_NAME = os.environ.get("IAM_LLM_MODEL", "gpt-4.1-mini")

# ---- Gemini models ----
GEMINI_MODEL_NAME = os.environ.get("IAM_LLM_GEMINI_MODEL", "gemini-3.5-flash-lite")

# ============================================================
# RUN TAG: date + model + topk + fp sheet + scenario-desc on/off.
# Used in output filenames so runs never overwrite each other.
# ============================================================
from datetime import datetime as _dt
# fp tag: sheet name by default; if a custom fingerprint file is supplied via env,
# use its filename so runs with different fingerprint sources are distinguishable
_fp_env = os.environ.get("IAM_LLM_FINGERPRINT_PATH", "")
_FP_TAG = (Path(_fp_env).stem.replace("fingerprint_from_papers_", "papers-")
           if _fp_env
           else (Path(FINGERPRINT_PATH).stem.replace("fingerprint_from_papers_", "papers-")
                 if "fingerprint_from_papers" in FINGERPRINT_PATH else FP_SHEET_NAME))
# ============================================================
# OPTIONAL SWITCHES (all off by default)
# Every one of these lands in RUN_TAG, so output files never collide and a
# run's config is readable from its filename.
# ============================================================
# Evaluate on a deterministic stratified subsample of the batch list.
# Stratified by (model_family, region) so the family mix matches the full
# list. Always compare variants on the same seed.
def _env_int(name: str, default: str) -> int:
    """int() an env var, with a readable error for the usual shell mistake.

    zsh (the macOS default) does NOT word-split an unquoted parameter
    expansion, so `E="A=1 B=2"; env $E cmd` passes ONE argument and A ends up
    holding "1 B=2". bash splits and works. The bare int() traceback for that
    is unhelpful, so name the cause.
    """
    raw = os.environ.get(name, default)
    try:
        return int(str(raw).strip())
    except ValueError:
        hint = ""
        if "=" in str(raw) or " " in str(raw).strip():
            hint = ("\n  This looks like a shell word-splitting problem. In zsh an "
                    "unquoted\n  $VAR expands to a SINGLE word, so `env $E ...` passes "
                    "everything as one\n  argument. Use `env ${=E} ...`, make E a zsh "
                    "array `E=(A=1 B=2)`, or just\n  export the variables once before "
                    "the runs.")
        raise ValueError(f"{name} must be an integer, got {raw!r}.{hint}") from None


SAMPLE_N    = _env_int("IAM_LLM_SAMPLE_N", "0")
SAMPLE_SEED = _env_int("IAM_LLM_SAMPLE_SEED", "42")


# ============================================================
# LLM RERANK (two-stage retrieval)
# ============================================================
RERANK = os.environ.get("IAM_LLM_RERANK", "1") == "1"

RERANK_MODE = os.environ.get("IAM_LLM_RERANK_MODE", "llm").strip().lower()
if RERANK_MODE not in ("off", "keyword", "hybrid", "llm"):
    raise ValueError(f"IAM_LLM_RERANK_MODE must be off|keyword|hybrid|llm, "
                     f"got {RERANK_MODE!r}")
if not RERANK:
    RERANK_MODE = "off"
# weight of the embedding cosine when blending it with the keyword distance
RERANK_BLEND = float(os.environ.get("IAM_LLM_RERANK_BLEND", "0.15"))
# The STEP2 pool is small by construction: for many cases no candidate from the
# same region and scenario under a different family exists after the family-pool
# rule, so RERANK_POOL_MODEL and TOPK_MODEL have no effect for those cases.
RERANK_POOL_SCEN  = _env_int("IAM_LLM_RERANK_POOL_SCEN", "15")
RERANK_POOL_MODEL = _env_int("IAM_LLM_RERANK_POOL_MODEL", "3")


RUN_TAG = (
    f"{_dt.now():%Y%m%d_%H%M}"
    f"_{(MODEL_NAME if PROVIDER == 'openai' else GEMINI_MODEL_NAME).replace('/', '-')}"
    f"_topk{TOPK_SCEN}{TOPK_MODEL}{TOPK_CROSS if RUN_STEP3 else 0}"
    f"_fp-{_FP_TAG}"
    f"_desc-{'on' if INCLUDE_SCENARIO_DESC else 'off'}"
    + (f"_n{SAMPLE_N}s{SAMPLE_SEED}" if SAMPLE_N else "")
    + ("" if RERANK_MODE == "off" else
       f"_rr{RERANK_MODE[:2]}{RERANK_POOL_SCEN}"
       + (f"x{RERANK_POOL_MODEL}" if RERANK_MODE == "llm" else ""))
    + (f"_{VARSET}" if VARSET != "supply11" else "")
)
# Dataset tag derived from the test list filename
# (AR6_R10sd_val10 / AR6_R10sd_sub01_test / ...)
DATASET_TAG = Path(BATCH_SHEET_PATH).stem.replace("_list", "")
OUT_BATCH_CSV = os.path.join(OUTPUTS_DIR, f"{DATASET_TAG}_pred_{RUN_TAG}.csv")
# prompts go to a sidecar (one line per case) instead of a column repeated on
# every row; set IAM_LLM_SAVE_PROMPTS=0 to skip it entirely
OUT_PROMPTS_JSONL = os.path.join(OUTPUTS_DIR, f"{DATASET_TAG}_prompts_{RUN_TAG}.jsonl")
SAVE_PROMPTS = os.environ.get("IAM_LLM_SAVE_PROMPTS", "1") == "1"


def print_resolved_config() -> None:
    """Print which files this process actually resolved, and whether each came
    from an env var or a default.

    Every path below is an independent env var read once at import. Printing
    the resolved files makes a wrong or missing variable visible before the run
    starts.
    """
    def src(var, value):
        return f"env {var}" if os.environ.get(var) else "default"
    rows = [
        ("batch list", BATCH_SHEET_PATH, src("IAM_LLM_BATCH_SHEET_PATH", BATCH_SHEET_PATH)),
        ("eval GT", GT_PATH,
         src("IAM_LLM_GT_PATH", GT_PATH) if os.environ.get("IAM_LLM_GT_PATH")
         else "derived from batch list"),
        ("neighbour pool", EXCEL_PATH, src("IAM_LLM_TRAIN_GT_PATH", EXCEL_PATH)),
        ("fingerprint", FINGERPRINT_PATH, src("IAM_LLM_FINGERPRINT_PATH", FINGERPRINT_PATH)),
        ("pred CSV out", OUT_BATCH_CSV, "derived"),
        ("eval xlsx out", globals().get("OUT_XLSX", "(set later)"), "derived"),
    ]
    w = max(len(r[0]) for r in rows)
    print("\n[CONFIG] resolved paths for this run")
    for name, path, how in rows:
        print(f"  {name:<{w}} : {Path(path).name}   ({how})")
    print(f"  {'provider':<{w}} : {PROVIDER} / "
          f"{GEMINI_MODEL_NAME if PROVIDER == 'gemini' else MODEL_NAME}")
    print(f"  {'retrieval':<{w}} : topk_scen={TOPK_SCEN} topk_model={TOPK_MODEL} "
          f"step3={RUN_STEP3}{f' topk_cross={TOPK_CROSS}' if RUN_STEP3 else ''}")
    _v = [n for n, on in [(f"rerank[{RERANK_MODE}] pool {RERANK_POOL_SCEN}"
                           + (f"/{RERANK_POOL_MODEL}" if RERANK_MODE == "llm" else "")
                           + (f" blend {RERANK_BLEND}" if RERANK_MODE == "hybrid" else ""),
                           RERANK_MODE != "off")] if on]
    print(f"  {'variant':<{w}} : {', '.join(_v) if _v else 'baseline (no switches)'}"
          f"{f'  |  sample n={SAMPLE_N} seed={SAMPLE_SEED}' if SAMPLE_N else ''}")
    print()


TEMPERATURE = 0
N_RUNS = 1

# ── usage accounting (tokens across all LLM calls; thread-safe enough via GIL) ──
USAGE = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "tokens_thinking": 0,
         # calls that raised BEFORE returning usage_metadata (not in est cost):
         "failed_calls": 0, "failed_503": 0, "failed_disconnect": 0,
         "failed_other": 0, "tokens_lost_est": 0}
RUN_STATS = {}   # filled by __main__ (elapsed seconds, cost estimate)
# unit prices in USD per 1M tokens (default: gemini-3.5-flash-lite paid tier,
# input $0.30 / output INCLUDING thinking tokens $2.50, per the official
# pricing page); override via env for other models
PRICE_IN  = float(os.environ.get("IAM_LLM_PRICE_IN",  "0.30"))
PRICE_OUT = float(os.environ.get("IAM_LLM_PRICE_OUT", "2.50"))


def _track_usage(n_in, n_out, n_think=0):
    USAGE["calls"] += 1
    USAGE["tokens_in"] += int(n_in or 0)
    USAGE["tokens_out"] += int(n_out or 0)
    USAGE["tokens_thinking"] += int(n_think or 0)


def usage_cost():
    return (USAGE["tokens_in"] * PRICE_IN
            + (USAGE["tokens_out"] + USAGE["tokens_thinking"]) * PRICE_OUT) / 1e6


def _classify_error(e) -> str:
    s = str(e).lower()
    if "503" in s or "unavailable" in s or "429" in s or "resource_exhausted" in s:
        return "failed_503"          # rejected up front -> not billed
    if "disconnect" in s or "timeout" in s or "timed out" in s or "deadline" in s:
        return "failed_disconnect"   # possibly processed & billed, response lost
    return "failed_other"


def _track_failed_call(gc, prompt_text, err):
    """Account an LLM call that raised before usage_metadata came back.
    These calls are invisible to est cost but MAY have been billed by Google
    (esp. server-disconnect cases). tokens_lost_est measures the prompt with
    the free count_tokens endpoint -> upper bound for the un-tracked input."""
    USAGE["failed_calls"] += 1
    kind = _classify_error(err)
    USAGE[kind] += 1
    if kind == "failed_503":         # rejected before processing: nothing billed
        return
    try:
        n_in = int(gc.models.count_tokens(
            model=GEMINI_MODEL_NAME, contents=prompt_text).total_tokens or 0)
    except Exception:
        n_in = len(prompt_text) // 4   # rough fallback if count_tokens also fails
    USAGE["tokens_lost_est"] += n_in


# typical JSON answer length, used only for the lost-cost upper bound
_LOST_OUT_EST_TOKENS = 2000


def lost_cost_upper_bound():
    """Max USD Google may have billed for failed (un-tracked) calls."""
    n_maybe_billed = USAGE["failed_disconnect"] + USAGE["failed_other"]
    return (USAGE["tokens_lost_est"] * PRICE_IN
            + n_maybe_billed * _LOST_OUT_EST_TOKENS * PRICE_OUT) / 1e6

INCLUDE_PER_VAR_RATIONALE = True

# HARD MODE: choose what to expose to the LLM (raw values OR curve features)
# - "raw"       => pass ONLY decade values (2010..2100)
# - "features"  => pass ONLY curve features (requires FEATURES_DF)
ELECTRICITY_EVIDENCE_MODE = "raw"      # "raw" | "features"

# For NON-electricity variables, choose what to expose.
OTHER_EVIDENCE_MODE = "raw"            # "raw" | "features"

# Years
YEARS = [str(y) for y in range(2010, 2101, 10)]

# ============================================================
# HARD CONSTRAINT VARIABLES (must match your exact variable strings)
# ============================================================
TOTAL_ELECTRICITY_VAR = None   # No single total variable for Primary Energy

TECH_ELECTRICITY_VARS = []     # Not used (no total balance constraint for Primary Energy)

AGG_IDENTITIES = []   # parents removed; no aggregation identities to enforce

# All primary energy variables share the same evidence mode
PARENT2PARTS = {it["parent"]: list(it["parts"]) for it in AGG_IDENTITIES}
ELECTRICITY_SET = set(VARS_14)
for parent, parts in PARENT2PARTS.items():
    ELECTRICITY_SET.add(parent)
    for p in parts:
        ELECTRICITY_SET.add(p)

# ============================================================
# REQUIRED INPUTS
# - VARS_14
# - TARGET dict with keys: region, model_family, scenario
# - scenario2desc
# - model_family2fp_text
# - STEP1_DF, STEP2_DF, (optional STEP3_DF)
# - (optional) FEATURES_DF if you choose any *MODE="features"
# ============================================================
VARS_OUT = [str(v).strip() for v in VARS_14]
VARS_SET = set(VARS_OUT)


TARGET_SCENARIO = str(TARGET["scenario"])
TARGET_FAMILY   = str(TARGET["model_family"]).strip()
TARGET_REGION   = str(TARGET["region"])

if "CACHE_DIR" not in globals():
    CACHE_DIR = "."

# ============================================================
# Basic getters
# ============================================================
def get_model_family_fp_text(model_family: str) -> str:
    if "model_family2fp_text" not in globals():
        raise KeyError("Missing global dict: model_family2fp_text (keyed by model_family).")
    t = model_family2fp_text.get(str(model_family))
    if t is None:
        raise KeyError(f"No fingerprint found for model_family='{model_family}'.")
    t = str(t).strip()
    if not t:
        raise ValueError(f"Empty fingerprint text for model_family='{model_family}'.")
    return t

def get_scenario_desc_text(scenario: str) -> str | None:
    if "scenario2desc" not in globals():
        return None
    t = scenario2desc.get(str(scenario))
    if t is None:
        return None
    t = str(t).strip()
    return t if t else None

def _nan_to_none(x):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    return x

# ============================================================
# Consistency checks
# ============================================================
def assert_constraints_consistent(vars_set: set[str]):
    missing = []
    for it in AGG_IDENTITIES:
        if it["parent"] not in vars_set:
            missing.append(it["parent"])
        for p in it["parts"]:
            if p not in vars_set:
                missing.append(p)
    missing = sorted(set(missing))
    if missing:
        raise ValueError(
            "Constraint variables missing from VARS_14 (NO implicit residuals).\n"
            + "\n".join([f" - {m}" for m in missing])
        )

assert_constraints_consistent(VARS_SET)

# ============================================================
# Neighbor DF assembly (raw decade data)
# ============================================================
def ensure_year_cols(df: pd.DataFrame, years=YEARS) -> pd.DataFrame:
    df = df.copy()
    for y in years:
        yi = int(y)
        if y not in df.columns and yi in df.columns:
            df = df.rename(columns={yi: y})
    return df

def concat_neighbor_raw(step1_df: pd.DataFrame, step2_df: pd.DataFrame, step3_df: pd.DataFrame | None):
    frames = []
    for x in [step1_df, step2_df, step3_df]:
        if x is not None and not x.empty:
            frames.append(x.copy())
    if not frames:
        # All neighbor categories empty is ALLOWED: the target may have no
        # same-scenario / same-family train neighbors under the new split.
        cols = ["combo_id", "step", "similarity", "region", "model_family",
                "scenario", "category_c", "variable", "unit"] + list(YEARS)
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    out = ensure_year_cols(out, YEARS)
    return out

# ============================================================
# Feature extraction helper (from FEATURES_DF) – only used if MODE="features"
# ============================================================
def require_features_df():
    if (
        ELECTRICITY_EVIDENCE_MODE == "features"
        or OTHER_EVIDENCE_MODE == "features"
    ):
        if "FEATURES_DF" not in globals() or FEATURES_DF is None or FEATURES_DF.empty:
            raise KeyError("You selected evidence_mode='features' but FEATURES_DF is missing/empty.")

def get_features_for_combo_variable(features_df: pd.DataFrame, combo_id: int, variable: str) -> dict:
    sub = features_df[
        (features_df["combo_id"].astype(int) == int(combo_id)) &
        (features_df["variable"].astype(str).str.strip() == str(variable).strip())
    ]
    if sub.empty:
        return {}
    r = sub.iloc[0].to_dict()
    drop = {
        "combo_id", "step", "similarity", "region", "model_family", "scenario",
        "category_c", "variable", "unit"
    }
    feats = {}
    for k, v in r.items():
        if k in drop:
            continue
        feats[k] = _nan_to_none(v)
    return feats

# ============================================================
# HARD MODE: build neighbors evidence – ONLY expose what mode requires
# ============================================================
def var_mode(variable: str) -> str:
    v = str(variable).strip()
    if v in ELECTRICITY_SET:
        return ELECTRICITY_EVIDENCE_MODE
    return OTHER_EVIDENCE_MODE

def build_neighbors_all_evidence_hard(raw_df: pd.DataFrame, vars_out: list[str]) -> list[dict]:
    """
    NEIGHBOR_ALL:
    - per variable: either "values" OR "features" (hard mode)
    - keep STEP1 / STEP2 / STEP3 separated in ordering because similarity meanings differ
    """
    require_features_df()
    neighbors = []

    if raw_df is None or raw_df.empty:
        return neighbors   # no neighbors available for this target - allowed

    need_cols = {"combo_id", "region", "model_family", "scenario", "similarity", "category_c", "variable", "unit"}
    miss = [c for c in need_cols if c not in raw_df.columns]
    if miss:
        raise KeyError(f"NEIGHBOR_RAW_DF missing columns: {miss}")

    # Label shown to the LLM. The strings match the neighbour roles described in the
    # prompt's VARIABLE TRAJECTORY INFERENCE RULES verbatim, so the model does not have to
    # re-derive the class by string-matching scenario / model_family against the TARGET.
    _NEIGHBOR_TYPE = {
        1: "SAME-MODEL-FAMILY / DIFFERENT-SCENARIO",
        2: "SAME-SCENARIO / DIFFERENT-MODEL-FAMILY",
        3: "CROSS (different model family AND different scenario)",
    }

    def _step_rank(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 999
        s = str(x).strip().lower()
        if s in {"1", "step1", "step_1", "step1_scen"}:
            return 1
        if s in {"2", "step2", "step_2", "step2_model"}:
            return 2
        if s in {"3", "step3", "step_3", "step3_cross"}:
            return 3
        return 999

    raw_df = raw_df.copy()

    # group by step + combo_id if step exists, otherwise keep old behavior
    if "step" in raw_df.columns:
        group_iter = raw_df.groupby([raw_df["step"], raw_df["combo_id"].astype(int)], dropna=False)
    else:
        group_iter = raw_df.groupby(raw_df["combo_id"].astype(int), dropna=False)

    for group_key, g in group_iter:
        meta = g.iloc[0]

        if isinstance(group_key, tuple):
            step_val, combo_id = group_key
        else:
            step_val, combo_id = None, group_key

        scenario = str(meta["scenario"])
        fam = str(meta["model_family"]).strip()

        case = {
            "combo_id": int(combo_id),
            "similarity": _nan_to_none(float(meta["similarity"])) if pd.notna(meta["similarity"]) else None,
            "neighbor_type": _NEIGHBOR_TYPE.get(_step_rank(step_val), "UNSPECIFIED"),
            "category_c": (lambda v: None if (v is None or (not isinstance(v, pd.Series) and pd.isna(v)))
                           else (str(v.iloc[0]) if isinstance(v, pd.Series) and not v.empty else str(v)))
                          (meta.get("category_c", None)),
            "region": str(meta["region"]),
            "model_family": fam,
            "scenario": scenario,
            "model_fingerprint_text": get_model_family_fp_text(fam),
            "scenario_description_text": get_scenario_desc_text(scenario),
            "series": [],
        }

        g2 = g.copy()
        g2["_var"] = g2["variable"].astype(str).str.strip()
        lookup = {str(r["_var"]): r for _, r in g2.iterrows()}

        for v in vars_out:
            v = str(v).strip()

            r0 = lookup.get(v)

            unit = None
            values = None
            feats = None

            mode = var_mode(v)

            if r0 is not None:
                unit = _nan_to_none(r0.get("unit", None))
                if mode == "raw":
                    vv = {}
                    for y in YEARS:
                        val = r0.get(y, None)
                        # raw mode: missing -> 0
                        vv[y] = 0.0 if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val)
                    values = vv
                elif mode == "features":
                    feats = get_features_for_combo_variable(FEATURES_DF, int(combo_id), v)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
            else:
                # Mixed dataset: families have different variable scopes, so a
                # cross-family neighbor may legitimately lack this variable.
                # OMIT it from this neighbor's evidence instead of showing
                # misleading all-zero values.
                continue

            item = {"variable": v, "unit": unit}
            if mode == "raw":
                item["values"] = values
            else:
                item["features"] = feats

            case["series"].append(item)

        # internal sort helper only; removed before return
        case["_step_rank"] = _step_rank(step_val)
        neighbors.append(case)

    # separate ordering by step first, then by similarity within each step
    neighbors.sort(
        key=lambda d: (
            d["_step_rank"],
            -(d["similarity"] if d["similarity"] is not None else -1e9)
        )
    )

    for d in neighbors:
        d.pop("_step_rank", None)

    return neighbors

# ============================================================
# TARGET block
# ============================================================
def build_target_block(target_family: str, target_scenario: str,
                       target_region: str | None = None) -> dict:
    """target_region is optional so existing 2-arg callers keep working. When given it is
    emitted FIRST: the task is defined per (region, model_family, scenario) and the
    trajectories are absolute levels whose magnitude depends on the region."""
    block = {}
    if target_region is not None and str(target_region).strip():
        block["region"] = str(target_region).strip()
    block.update({
        "model_family": str(target_family),
        "scenario": str(target_scenario),
        "model_fingerprint_text": get_model_family_fp_text(str(target_family)),
        "scenario_description_text": (
            get_scenario_desc_text(str(target_scenario))
            if INCLUDE_SCENARIO_DESC else None
        ),
    })
    return block

# ============================================================
# PROMPT BUILDER
# ============================================================
def build_json_schema(
    vars_out: list[str],
    include_per_var_rationale: bool,
) -> Dict[str, Any]:
    """Schema for the trajectory call. Trajectories are the only output. The AR6
    climate category is predicted by a separate classifier (classifier/)."""
    # Field order matters: Gemini emits fields in schema order. The per-variable
    # rationale is placed BEFORE the trajectories it justifies so it is actual
    # reasoning rather than post-hoc text.
    props: Dict[str, Any] = {}

    if include_per_var_rationale:
        props["per_variable_rationales"] = {
            "type": "object",
            "description": (
                "Mapping from exact variable name to a detailed 3-5 sentence evidence-grounded "
                "rationale. Write these BEFORE the trajectories."
            ),
            "additionalProperties": {"type": "string"},
        }

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "trajectories": {
                "type": "array",
                # Pin the variable set at DECODING time via the "variable" enum below.
                # NOTE: do NOT add minItems/maxItems here. Gemini's response_json_schema
                # answers 400 INVALID_ARGUMENT for a length-constrained array whose items
                # are an object this large. The exact variable count is covered by the
                # prompt's VARIABLE LIST.
                "items": {
                    "type": "object",
                    "properties": {
                        "variable": {
                            "type": "string",
                            "enum": [str(v) for v in vars_out],
                            "description": "Exact variable name",
                        },
                        "unit": {
                            # "nullable" is an OpenAPI keyword, not JSON Schema; under
                            # response_json_schema it is ignored and null would be rejected.
                            "type": ["string", "null"],
                            "description": "Unit, or null if unknown",
                        },
                        "values": {
                            "type": "object",
                            "properties": {
                                "2010": {"type": "number"},
                                "2020": {"type": "number"},
                                "2030": {"type": "number"},
                                "2040": {"type": "number"},
                                "2050": {"type": "number"},
                                "2060": {"type": "number"},
                                "2070": {"type": "number"},
                                "2080": {"type": "number"},
                                "2090": {"type": "number"},
                                "2100": {"type": "number"},
                            },
                            "required": [
                                "2010", "2020", "2030", "2040", "2050",
                                "2060", "2070", "2080", "2090", "2100",
                            ],
                        },
                    },
                    "required": ["variable", "unit", "values"],
                },
            },
        },
    }

    props["trajectories"] = schema["properties"]["trajectories"]

    return {"type": "object", "properties": props, "required": list(props.keys())}


def build_prompt(
    target_block: dict,
    neighbors_all: list[dict],
    vars_out: list[str],
    total_var: str,
    tech_vars: list[str],
    include_per_var_rationale: bool,
) -> str:
    """Trajectory prompt. Trajectories are the only output. The climate
    category is predicted by a separate classifier (classifier/)."""

    fp_scope = fp_scope_for(vars_out)

    per_var_rationale_rules = (
        "Provide a concise, evidence-grounded rationale for EACH variable trajectory in 3–5 sentences per variable. "
        "The rationale MUST follow this structure: "
        "(1) Describe the quantitative pattern inferred from the neighbors, "
        "such as the onset of change, peak timing, decline rate, long-run level, or stabilization behavior. "
        "(2) Explain how the TARGET scenario_description_text modifies the inferred trajectory in terms of HOW FAST and HOW FAR "
        "the variable evolves. For example, stronger mitigation may imply a steeper decline and a lower long-run level. "
        "Describe these modifications relative to the relevant same-model, different-scenario neighbor (if available). "
        "(3) WHERE THE FINGERPRINT SPEAKS TO A VARIABLE, explain how the TARGET model_fingerprint_text shifts "
        "that variable's LEVEL up or down relative to the relevant same-scenario, different-model neighbor if such a "
        "neighbor is available. The fingerprint covers only a few variables; for any variable it does not mention, "
        "say so in one clause and rely on the neighbor evidence instead - do not invent a fingerprint claim; "
        f"the {FP_PREF_LABEL} states where this model sits RELATIVE TO OTHER MODELS on the variables it does cover, "
        f"drawn from: {fp_scope}. "
        "The rationale must be specific and grounded in observable patterns from the provided inputs. "
        "Do not produce vague statements, generic reasoning, or assumptions that are not directly supported by the inputs."
    )


    soft_section = ""
    _vs_set = {str(v).strip() for v in vars_out}
    if VARSET == "sd22" and "Final Energy|Industry|Electricity" in _vs_set:
        soft_section = (
            "NUMERIC CONSISTENCY CONSTRAINTS:\n\n"
            "Sectoral consistency (must hold for each year y):\n"
            ' - "Final Energy|Industry|Electricity"[y] <= "Final Energy|Industry"[y]\n'
            ' - "Final Energy|Transportation|Electricity"[y] <= "Final Energy|Transportation"[y]\n'
            ' - "Final Energy|Residential and Commercial|Electricity"[y] <= '
            '"Final Energy|Residential and Commercial"[y]\n\n')

    neighbors_json_all = json.dumps(neighbors_all, ensure_ascii=False, indent=2)
    target_json = json.dumps(target_block, ensure_ascii=False, indent=2)

    vars_json = json.dumps([str(v) for v in vars_out], ensure_ascii=False)
    tech_json = json.dumps([str(v) for v in tech_vars], ensure_ascii=False)

    # Leading newline lives INSIDE this block so an empty block collapses cleanly
    # instead of leaving a stray blank line in the rendered prompt.
    per_var_block = ""
    if include_per_var_rationale:
        per_var_block = f"""

PER-VARIABLE RATIONALES REQUIREMENT:
- {per_var_rationale_rules}
""".rstrip()

    # This block MUST mirror the field order of build_json_schema exactly:
    #   [per_variable_rationales] -> trajectories
    pre_fields = ""
    if include_per_var_rationale:
        pre_fields = ('\n  "per_variable_rationales": { "<exact variable name>": '
                      '"<evidence-grounded rationale; write these FIRST>", "...": "..." },')

    prompt = f"""
You are an expert in integrated assessment model (IAM) scenario synthesis for IPCC AR6-style databases.
Your task is to generate decade-step numeric trajectories for a TARGET model_family × scenario using the provided TARGET information and NEIGHBOR evidence.

You must reason carefully and step by step internally, but you must output JSON only.
Do NOT include any extra text before or after the JSON.

TASK:
Generate, for the TARGET model_family × scenario, full trajectories (no interpolation) for ALL required variables, for years: {", ".join(YEARS)}

EVIDENCE PROVIDED:
- for each NEIGHBOR: its variable trajectories, its scenario_description_text and its model_fingerprint_text;
- for the TARGET: its scenario_description_text and its model_fingerprint_text (no trajectories).

DECISION PROCESS (FOLLOW INTERNALLY IN THIS ORDER):

VARIABLE TRAJECTORY INFERENCE RULES
Step 1: Infer variable trajectories using quantitative neighbor patterns.
- Use neighbors as quantitative references for trajectory shape, timing, slope, peak timing, decline timing, long-run level, and stabilization behavior.
- Read a neighbor's numbers through its OWN scenario_description_text and model_fingerprint_text (what produced them), then move them by how the TARGET differs from that neighbor on those same two texts.
- Each neighbor in NEIGHBOR_ALL carries a "neighbor_type" field naming its class; use it to decide how to read that neighbor.
- SAME-MODEL-FAMILY / DIFFERENT-SCENARIO neighbors are the primary references for the structural behavior of the TARGET model family. Use the neighbors' and the TARGET's scenario_description_text to translate the neighbor patterns into the TARGET trajectory.
- SAME-SCENARIO / DIFFERENT-MODEL-FAMILY neighbors are scenario-level references. Use them to understand how the target scenario responds under different models, and then translate these patterns to the TARGET model family by comparing the neighbors' and the TARGET model's {FP_PREF_LABEL} fingerprints.
- Use both the neighbors' and the TARGET's scenario_description_text to determine their relative mitigation stringency and constraints, such as a tighter carbon budget or a more ambitious climate target. These factors influence HOW FAST and HOW FAR the TARGET system should change relative to the neighbors.
- Use both the neighbors' and the TARGET's model_fingerprint_text to determine their relative LEVEL positioning - which model sits higher or lower than the other on a given variable:
  - {FP_PREF_LABEL} states, for a few of the predicted variables ({fp_scope}), whether this model tends to sit at a higher or lower LEVEL than other models.
  - The fingerprint may read "Not Available", or may simply not mention a variable. That means no cross-model evidence exists for it: the information is missing, and this is never itself a property of the model. Do not treat it as average, neutral or mid-range - leave that variable's fingerprint-based adjustment at zero and rely on the neighbor evidence instead. Never treat two models as alike because both read "Not Available".

Step 2: Fix the long-horizon anchors before writing any numbers.
- For each variable, decide the 2100 LEVEL implied by the neighbour evidence and the TARGET's mitigation stringency, then the TURNING POINT (the decade where the variable peaks, or where it begins its terminal decline, or "monotone" if it has neither).
- Anchor the 2100 level on the neighbours' own 2100 values: pick the neighbour that is closest in mitigation stringency and decide whether the TARGET should end above, below or at that level.

Step 3: Generate the decade path subject to all consistency constraints.
- Fill in the decades so the path passes through the turning point from Step 2 and lands on its 2100 level.
- Ensure non-negativity.
- Avoid unrealistic decade-to-decade oscillations.

{soft_section}NON-NEGATIVITY:
- Every value must be >= 0 for every year.

GENERAL OUTPUT RULES:
- Output MUST be valid JSON.
- Values must be numeric only.
- Do not output NaN.
- Do not output strings for numeric values.
- Avoid null whenever possible.
- Do not omit required variables.
- Use exact variable names.
- Use only the provided evidence.
- If evidence is weak, stay conservative and rely on the closest visible reference pattern rather than inventing unsupported detail.{per_var_block}

VARIABLE LIST (must include EXACTLY these variables, each exactly once):
{vars_json}

OUTPUT JSON FORMAT (STRICT; FOLLOW EXACTLY, INCLUDING FIELD ORDER):
{{{pre_fields}
  "trajectories": [
    {{
      "variable": "<exact variable name>",
      "unit": "<unit or null>",
      "values": {{
        "2010": <number>,
        "2020": <number>,
        "2030": <number>,
        "2040": <number>,
        "2050": <number>,
        "2060": <number>,
        "2070": <number>,
        "2080": <number>,
        "2090": <number>,
        "2100": <number>
      }}
    }}
  ]
}}

TARGET (JSON):
{target_json}

NEIGHBOR_ALL (JSON; "neighbor_type" gives each neighbor's class):
{neighbors_json_all}

Return JSON only.
""".strip()

    return prompt

# ============================================================
# JSON extraction (robust)
# ============================================================
def extract_json(text: str) -> dict:
    s = (text or "").strip()

    s = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s.strip())

    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        raise ValueError(f"Model output does not contain a JSON object.\nRAW:\n{s[:800]}")
    s = s[i:j+1].strip()

    s = re.sub(r",\s*([}\]])", r"\1", s)

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(s)
            return json.loads(json.dumps(obj, ensure_ascii=False))
        except Exception as e:
            raise ValueError(
                "Failed to parse JSON even after repairs.\n"
                f"Original JSONDecodeError: {repr(e)}\n"
                f"RAW (first 1200 chars):\n{s[:1200]}"
            )

# ============================================================
# Projection to satisfy constraints exactly
# ============================================================
def project_constraints_inplace(traj_map: dict):
    # 0) Non-negativity + fill missing
    for var, item in traj_map.items():
        for y in YEARS:
            v = item["values"].get(y, None)

            if v is None or (isinstance(v, float) and np.isnan(v)):
                item["values"][y] = 0.0
                continue

            vv = float(v)
            item["values"][y] = max(vv, 0.0)

    # 1) Enforce identities: parent = sum(parts)
    for y in YEARS:
        for parent, parts in PARENT2PARTS.items():
            P = float(traj_map[parent]["values"][y])
            s_parts = float(sum(float(traj_map[p]["values"][y]) for p in parts))

            if s_parts > 0:
                traj_map[parent]["values"][y] = s_parts
            else:
                if P > 0:
                    n = len(parts)
                    for p in parts:
                        traj_map[p]["values"][y] = P / n if n > 0 else 0.0
                    traj_map[parent]["values"][y] = float(
                        sum(float(traj_map[p]["values"][y]) for p in parts)
                    )
                else:
                    for p in parts:
                        traj_map[p]["values"][y] = 0.0
                    traj_map[parent]["values"][y] = 0.0

    # 2) Final exact identities again
    for y in YEARS:
        for parent, parts in PARENT2PARTS.items():
            traj_map[parent]["values"][y] = float(
                sum(float(traj_map[p]["values"][y]) for p in parts)
            )

# ============================================================
# Helpers: output <-> map
# ============================================================
def trajectories_to_map(out_obj: dict, vars_out: list | None = None) -> dict:
    """
    Parse model output to internal map.
    All variables must have numeric values for every year.
    """
    m = {}
    trajs = out_obj.get("trajectories", [])
    if not isinstance(trajs, list):
        raise ValueError("Output 'trajectories' must be a list.")

    for item in trajs:
        var = str(item.get("variable", "")).strip()
        if not var:
            raise ValueError("Trajectory item missing variable.")

        unit = item.get("unit", None)
        values_obj = item.get("values", {}) or {}
        if not isinstance(values_obj, dict):
            raise ValueError(f"Invalid values dict for var={var}")

        vals = {}

        for y in YEARS:
            if y not in values_obj:
                raise KeyError(f"Missing year '{y}' for variable '{var}'")

            v = values_obj.get(y, None)

            if v is None:
                raise ValueError(f"Null value not allowed for variable '{var}' year {y}")

            try:
                vals[y] = float(v)
            except Exception:
                raise ValueError(f"Non-numeric value for variable '{var}' year {y}: {v}")

        m[var] = {"unit": unit, "values": vals}

    for var in (vars_out if vars_out is not None else VARS_OUT):
        if var not in m:
            raise KeyError(f"Model output missing variable '{var}' entirely.")

    return m

def map_to_output(
    avg_map: dict,
    per_var_rationales: dict | None,
    prompt: str | None = None,
    vars_out: list | None = None,
) -> dict:

    def _json_safe_number(x):
        if x is None:
            return None
        if isinstance(x, float) and np.isnan(x):
            return None
        return float(x)

    out = {
        "prompt": prompt,
        "trajectories": [
            {
                "variable": var,
                "unit": avg_map[var]["unit"],
                "values": {y: _json_safe_number(avg_map[var]["values"][y]) for y in YEARS},
            }
            for var in (vars_out if vars_out is not None else VARS_OUT)
        ],
    }

    if INCLUDE_PER_VAR_RATIONALE:
        out["per_variable_rationales"] = per_var_rationales or {}

    return out


"""Run"""

# ============================================================
# 1) Core function: synthesize_one_target (KEEP THIS)
# ============================================================

from decimal import Decimal, ROUND_HALF_UP
import math

def round_sig(x, sig=3):
    if x == 0:
        return 0.0
    return float(
        Decimal(x).quantize(
            Decimal(f'1e{int(math.floor(math.log10(abs(x)))) - sig + 1}'),
            rounding=ROUND_HALF_UP
        )
    )

def round_dict_floats(d, sig=3):
    if isinstance(d, dict):
        return {k: round_dict_floats(v, sig) for k, v in d.items()}
    elif isinstance(d, list):
        return [round_dict_floats(v, sig) for v in d]
    elif isinstance(d, tuple):
        return tuple(round_dict_floats(v, sig) for v in d)
    elif isinstance(d, float):
        return round_sig(d, sig)
    else:
        return d

def synthesize_one_target(target: dict, include_rationales: bool = True) -> dict:
    """
    Synthesize trajectories for one target combo.
    Batch-safe: does NOT rely on global TARGET_* variables.
    """

    global INCLUDE_PER_VAR_RATIONALE

    target_region = str(target["region"]).strip()
    target_family = str(target["model_family"]).strip()
    target_scenario = str(target["scenario"]).strip()
    # per-family variable scope (core=22, COFFEE/WITCH=supply11, GCAM/TIAM=demand11)
    vars_out_t = vars_for_family(target_family)

    # Ensure TARGET scenario has a description (only if we intend to use it)
    if INCLUDE_SCENARIO_DESC:
        td = get_scenario_desc_text(target_scenario)
        if td is None or not str(td).strip():
            raise KeyError(f"TARGET scenario '{target_scenario}' has no scenario_description in scenario2desc.")

    # Compute retrieval steps for THIS target
    steps = compute_steps_for_target({
        "region": target_region,
        "model_family": target_family,
        "scenario": target_scenario
    })
    step1_df = steps["STEP1_DF"]
    step2_df = steps["STEP2_DF"]
    step3_df = steps["STEP3_DF"]

    # Neighbor raw df
    neighbor_raw_df = concat_neighbor_raw(
        step1_df=step1_df,
        step2_df=step2_df,
        step3_df=step3_df if (step3_df is not None and not step3_df.empty) else None
    )

    # Build hard-mode neighbor evidence
    neighbors_all = build_neighbors_all_evidence_hard(neighbor_raw_df, vars_out_t)
    target_block = build_target_block(target_family, target_scenario, target_region)
    print("Neighbors All: ", neighbors_all)

    prompt = build_prompt(
        target_block=target_block,
        neighbors_all=neighbors_all,
        vars_out=vars_out_t,
        total_var=TOTAL_ELECTRICITY_VAR,
        tech_vars=TECH_ELECTRICITY_VARS,
        include_per_var_rationale=INCLUDE_PER_VAR_RATIONALE,
    )
    gemini_json_schema = build_json_schema(
        vars_out_t,
        INCLUDE_PER_VAR_RATIONALE,
    )

    # ----------------------------
    # LLM call (OpenAI / Gemini)
    # ----------------------------
    from openai import OpenAI
    openai_client = OpenAI()  # uses OPENAI_API_KEY from env

    def _ensure_gemini_client():
        """
        New SDK: google-genai (google.genai)
        Returns a cached client stored in global 'gemini_client'.
        """
        global gemini_client
        if "gemini_client" in globals() and gemini_client is not None:
            return gemini_client

        if not os.environ.get("GEMINI_API_KEY", ""):
            raise ValueError("Missing GEMINI_API_KEY in env.")

        from google import genai
        gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return gemini_client

    def _ensure_hf_model():
        global hf_model, hf_tokenizer
        if "hf_model" in globals() and hf_model is not None:
            return hf_model, hf_tokenizer

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        # Use HF_MODEL_NAME if available, else MODEL_NAME
        model_name = globals().get("HF_MODEL_NAME", globals().get("MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf"))

        hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        hf_model.eval()
        return hf_model, hf_tokenizer

    def _run_once(p: str | None = None, schema: dict | None = None) -> dict:
        if p is None:
            p = prompt
        if schema is None:
            schema = gemini_json_schema
        if PROVIDER == "openai":
            resp = openai_client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                seed=globals().get("SEED", 0),
                messages=[
                    {"role": "system", "content": "Return JSON only."},
                    {"role": "user", "content": p},
                ],
            )
            if getattr(resp, "usage", None):
                _track_usage(resp.usage.prompt_tokens, resp.usage.completion_tokens)
            return extract_json(resp.choices[0].message.content)

        if PROVIDER == "gemini":
            gc = _ensure_gemini_client()
            gemini_prompt = "Return JSON only.\n\n" + p
            from google.genai import types
            # Thinking is ON by default (billed as output tokens - the hidden cost
            # driver). Gemini 3.x cannot fully disable it and uses thinking_level
            # ("minimal" is the floor); Gemini 2.5 uses thinking_budget=0.
            if GEMINI_MODEL_NAME.startswith("gemini-3"):
                thinking_cfg = types.ThinkingConfig(thinking_level="minimal")
            else:
                thinking_cfg = types.ThinkingConfig(thinking_budget=0)
            try:
                resp = gc.models.generate_content(
                    model=GEMINI_MODEL_NAME,
                    contents=gemini_prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=thinking_cfg,
                        response_mime_type="application/json",
                        response_json_schema=schema,
                        temperature=TEMPERATURE,
                        max_output_tokens=40000,
                    ),
                )
            except Exception as e:
                _track_failed_call(gc, gemini_prompt, e)
                raise
            um = getattr(resp, "usage_metadata", None)
            if um is not None:
                _track_usage(getattr(um, "prompt_token_count", 0),
                             getattr(um, "candidates_token_count", 0),
                             getattr(um, "thoughts_token_count", 0))
            return extract_json((resp.text or "").strip())

        if PROVIDER == "huggingface":
            model, tokenizer = _ensure_hf_model()
            messages = [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": p},
            ]

            try:
                input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
            except Exception:
                # Fallback if chat template is unsupported
                hf_prompt = "System: Return JSON only.\nUser: " + p + "\nAssistant:"
                input_ids = tokenizer(hf_prompt, return_tensors="pt").input_ids.to(model.device)

            gen_kwargs = {
                "max_new_tokens": 4000,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if getattr(globals(), "TEMPERATURE", 0) > 0.0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = globals()["TEMPERATURE"]
            else:
                gen_kwargs["do_sample"] = False

            import torch
            print(input_ids.shape)
            with torch.no_grad():
                outputs = model.generate(input_ids, **gen_kwargs)

            input_length = input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            text_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return extract_json(text_output.strip())

        raise ValueError(f"Unknown PROVIDER: {PROVIDER}")

    outs = [_run_once() for _ in range(N_RUNS)]

    # Average trajectories across runs
    maps = [trajectories_to_map(o, vars_out_t) for o in outs]
    avg_map = {}

    for var in vars_out_t:
        unit = next((m[var]["unit"] for m in maps if m[var].get("unit", None) is not None), None)
        avg_vals = {y: float(np.mean([m[var]["values"][y] for m in maps])) for y in YEARS}
        avg_map[var] = {"unit": unit, "values": avg_vals}

    # Enforce constraints exactly
    project_constraints_inplace(avg_map)

    # Rationales (optional)
    per_var_rationales = None
    if INCLUDE_PER_VAR_RATIONALE:
        prv = outs[0].get("per_variable_rationales", {})
        per_var_rationales = prv if isinstance(prv, dict) else {}

    out_obj = map_to_output(
        avg_map=avg_map,
        per_var_rationales=per_var_rationales,
        prompt=prompt,
        vars_out=vars_out_t,
    )

    return out_obj


# ============================================================
# 2) MODE SWITCH (ONLY SINGLE or ONLY BATCH)
# ============================================================

import os
import json
import asyncio
import time
import numpy as np
import pandas as pd


def load_validation_xlsx(path: str) -> pd.DataFrame:
    dfv = pd.read_excel(path)
    required = ["id", "region", "model_family", "scenario"]
    missing = [c for c in required if c not in dfv.columns]
    if missing:
        raise KeyError(f"Missing required columns in validation sheet: {missing}")

    dfv = dfv.copy()
    dfv["id"] = dfv["id"].astype(int)
    for c in ["region", "model_family", "scenario"]:
        dfv[c] = dfv[c].astype(str).str.strip()
    return dfv


def stratified_sample(dfv: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """A deterministic subsample of the case list.

    Stratified by (model_family, region): the error differs between families,
    so an unstratified draw would move the metric by shifting the family mix.
    NOT the same as TEST_LIMIT, which takes the head of a list sorted by
    region/scenario/family and therefore covers only a few regions.

    The same (n, seed) always yields the same cases, so variants are compared
    on paired cases.
    """
    if n <= 0 or n >= len(dfv):
        return dfv
    strata = (dfv["model_family"].astype(str) + "||" + dfv["region"].astype(str))
    rng = np.random.default_rng(seed)
    counts = strata.value_counts().sort_index()
    quota = (counts * (n / counts.sum())).astype(int)
    rem = n - int(quota.sum())
    for k in (counts * (n / counts.sum()) - quota).sort_values(ascending=False).index:
        if rem <= 0:
            break
        if quota[k] < counts[k]:
            quota[k] += 1
            rem -= 1
    keep = []
    for k in counts.index:
        idx = np.array(dfv.index[strata == k])   # .to_numpy() on an Index is read-only
        rng.shuffle(idx)
        keep += idx[: int(quota[k])].tolist()
    out = dfv.loc[sorted(keep)].reset_index(drop=True)
    print(f"[SAMPLE] stratified subsample: {len(out)}/{len(dfv)} cases "
          f"(seed {seed}, {strata.nunique()} strata) - compare variants on this "
          f"same (n, seed)")
    return out


def assert_batch_gt_consistent(dfv: pd.DataFrame, gt_path: str) -> None:
    """Verify the case list and the eval GT describe the SAME split, before a
    single API call is made.

    Without this, predicting one split while scoring against another (easy to
    do: BATCH_SHEET_PATH and GT_PATH are independent env vars) would surface only
    in assert_exact_key_match at the very end, after the whole batch had been
    generated and billed.
    """
    try:
        gt = pd.read_excel(gt_path, usecols=["id", "region", "model_family", "scenario"])
    except Exception as e:
        raise RuntimeError(f"Cannot read GT_PATH for the pre-flight check: "
                           f"{gt_path}\n{e}") from e

    gt["id"] = pd.to_numeric(gt["id"], errors="coerce").astype("Int64")
    list_ids, gt_ids = set(dfv["id"].astype(int)), set(gt["id"].dropna().astype(int))
    missing = sorted(list_ids - gt_ids)
    if missing:
        overlap = len(list_ids & gt_ids)
        raise ValueError(
            "[PRE-FLIGHT] The batch list and the eval GT are not the same split - "
            "nothing was generated.\n"
            f"  batch list : {BATCH_SHEET_PATH}\n"
            f"               {len(list_ids)} cases, ids {min(list_ids)}..{max(list_ids)}\n"
            f"  eval GT    : {gt_path}\n"
            f"               {len(gt_ids)} cases, ids {min(gt_ids)}..{max(gt_ids)}\n"
            f"  overlap    : {overlap} case(s); {len(missing)} list id(s) absent "
            f"from the GT, e.g. {missing[:5]}\n"
            "  Set IAM_LLM_BATCH_SHEET_PATH only - GT_PATH now defaults to the "
            "matching *_gt.xlsx - or point IAM_LLM_GT_PATH at the same split."
        )

    # ids can line up while the rows describe different cases (e.g. a list and a
    # GT regenerated from different splits); the keys are right there, so check.
    keys = ["region", "model_family", "scenario"]
    g = gt.dropna(subset=["id"]).copy()
    for c in keys:
        g[c] = g[c].astype(str).str.strip()
    g = g.drop_duplicates(subset=["id"]).set_index("id")
    bad = []
    for r in dfv.itertuples(index=False):
        row = g.loc[int(r.id)] if int(r.id) in g.index else None
        if row is None:
            continue
        if (row["region"], row["model_family"], row["scenario"]) != (
                str(r.region).strip(), str(r.model_family).strip(), str(r.scenario).strip()):
            bad.append(int(r.id))
    if bad:
        raise ValueError(
            f"[PRE-FLIGHT] {len(bad)} case id(s) describe different "
            f"(region, model_family, scenario) in the list vs the GT, e.g. {bad[:5]}. "
            f"Re-run make_datasets/split_r10_sd.py so the two files match.\n"
            f"  batch list : {BATCH_SHEET_PATH}\n  eval GT    : {gt_path}"
        )

    print(f"[PRE-FLIGHT] ok: {len(list_ids)} cases, list and GT agree "
          f"({Path(BATCH_SHEET_PATH).name} <-> {Path(gt_path).name})")


def run_single() -> dict:
    """Run exactly one TARGET and save JSON/XLSX (prints rationales)."""
    out_obj = synthesize_one_target(TARGET, include_rationales=True)

    if INCLUDE_PER_VAR_RATIONALE:
        print("\nPer-variable rationales (FULL):")
        for v in vars_for_family(TARGET["model_family"]):
            txt = out_obj.get("per_variable_rationales", {}).get(v, "")
            print(f"\n- {v}:\n{txt}")

    # Save JSON
    with open(OUT_SINGLE_JSON, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print("\nSaved JSON:", OUT_SINGLE_JSON)

    # Save XLSX (variables × years)
    rows = []
    for item in out_obj["trajectories"]:
        row = {"variable": item["variable"], "unit": item.get("unit", None)}
        for y in YEARS:
            row[y] = item["values"][y]
        row['prompt'] = out_obj.get('prompt', '')
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_excel(OUT_SINGLE_XLSX, index=False)
    print("Saved XLSX:", OUT_SINGLE_XLSX)

    return out_obj


def run_one_target_no_rationale(target: dict) -> dict:
    """Batch mode: one target run with rationales disabled."""
    return synthesize_one_target(target, include_rationales=False)


def run_with_retry_blocking(target: dict) -> dict:
    """Blocking retry wrapper (used inside asyncio.to_thread).
    429 / RESOURCE_EXHAUSTED (spend-based rate limits) get a much longer,
    linearly growing wait so the quota window can actually replenish."""
    last_err = None
    for k in range(RETRY_MAX):
        try:
            return run_one_target_no_rationale(target)
        except Exception as e:
            last_err = e
            s = str(e).lower()
            if "429" in s or "resource_exhausted" in s or "quota" in s:
                # spend-rate window: wait 30s, 60s, 90s ... capped at 180s
                wait = min(30.0 * (k + 1) + 10 * np.random.random(), 180.0)
            else:
                # transient error: exponential backoff with jitter, capped at 30s
                wait = min((RETRY_BACKOFF_BASE ** k) + np.random.random(), 30.0)
            print(f"Retry {k + 1}/{RETRY_MAX} in {wait:.0f}s due to error: {e}")
            time.sleep(float(wait))
    raise RuntimeError(f"Failed after retries. Last error: {repr(last_err)}")


# id -> prompt, filled by run_batch. Kept out of the prediction rows: the
# prompt is long and identical for all rows of a case.
PROMPT_STORE: dict[int, str] = {}

# Per-case wall time in seconds, measured from the moment a case ACQUIRES a
# concurrency slot to the moment it returns (so queue wait is excluded and
# retry sleeps are included). Filled by run_batch; averaged into RUN_STATS as
# sec_per_case_latency_measured, which _timing_rows() reports as the
# sec_per_case_latency. The derived form (throughput x concurrency) assumes a
# saturated pool and is reported as sec_per_case_latency_derived only when no
# measurement exists.
CASE_SECS: list[float] = []

# (case id, repr(error)) for cases that exhausted RETRY_MAX. Filled by
# run_batch. Non-empty means the pred CSV is INCOMPLETE: __main__ still writes
# the successful rows (so a billed batch is never thrown away) and then exits
# non-zero BEFORE run_evaluation, because scoring a short batch would silently
# change the denominator and make this run incomparable to the others.
FAILED_CASES: list[tuple] = []


async def run_batch(dfv: pd.DataFrame) -> pd.DataFrame:
    """Run batch synthesis with bounded concurrency."""
    import concurrent.futures
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)
    loop = asyncio.get_running_loop()
    print(f"  [run_batch] semaphore {MAX_CONCURRENCY} | thread pool "
          f"{pool._max_workers} (default would have been "
          f"{min(32, (os.cpu_count() or 1) + 4)})")

    async def _one(row: pd.Series) -> pd.DataFrame:
        target = {
            "region": row["region"],
            "model_family": row["model_family"],
            "scenario": row["scenario"],
        }

        async with sem:
            _t_case = time.time()
            out_obj = await loop.run_in_executor(pool, run_with_retry_blocking,
                                                 target)
            CASE_SECS.append(time.time() - _t_case)

        PROMPT_STORE[int(row["id"])] = out_obj.get("prompt", "") or ""

        rows = []
        for item in out_obj["trajectories"]:
            r = {
                "id": int(row["id"]),
                "region": row["region"],
                "model_family": row["model_family"],
                "scenario": row["scenario"],
                "variable": item["variable"],
                "unit": item.get("unit", None),
            }
            for y in YEARS:
                r[y] = item["values"][y]
            rows.append(r)

        return pd.DataFrame(rows)

    try:
        parts = await asyncio.gather(*[_one(dfv.iloc[i])
                                       for i in range(len(dfv))],
                                     return_exceptions=True)
    finally:
        pool.shutdown(wait=False)

    FAILED_CASES.clear()
    ok = []
    for i, p in enumerate(parts):
        if isinstance(p, BaseException):
            FAILED_CASES.append((int(dfv.iloc[i]["id"]), repr(p)))
        else:
            ok.append(p)
    if not ok:
        raise RuntimeError(
            f"All {len(dfv)} cases failed. First error: {FAILED_CASES[0][1]}")
    return pd.concat(ok, ignore_index=True)


# ----------------------------
# EXECUTE: mutually exclusive
# ----------------------------
if __name__ == "__main__":
    if MODE == "single":
        _ = run_single()

    elif MODE == "batch":
        INCLUDE_PER_VAR_RATIONALE = False

        print_resolved_config()
        dfv = load_validation_xlsx(BATCH_SHEET_PATH)
        _lim = int(os.environ.get("TEST_LIMIT", "0"))
        if _lim > 0:
            dfv = dfv.head(_lim)
            print(f"[NOTE] TEST_LIMIT={_lim}: running only the first {_lim} test targets.")
        if SAMPLE_N:
            dfv = stratified_sample(dfv, SAMPLE_N, SAMPLE_SEED)
        # fail before spending anything if the list and the GT disagree
        assert_batch_gt_consistent(dfv, GT_PATH)
        import time as _time
        _t0 = _time.time()
        out_df = asyncio.run(run_batch(dfv))
        RUN_STATS["elapsed_sec"] = round(_time.time() - _t0, 1)
        _n_cases = len(dfv)
        if _n_cases:
            RUN_STATS["sec_per_case_throughput"] = round(
                RUN_STATS["elapsed_sec"] / _n_cases, 4)
        if CASE_SECS:
            import statistics as _st
            RUN_STATS["sec_per_case_latency_measured"] = round(
                sum(CASE_SECS) / len(CASE_SECS), 2)
            # p50 is the REPORTED single-trajectory time: the mean is dragged up
            # by the handful of cases that hit a disconnect and sat through a
            # 30/60/90s retry backoff, which is a transport artefact, not the
            # time it takes to generate a trajectory.
            RUN_STATS["sec_per_case_latency_p50"] = round(_st.median(CASE_SECS), 2)
            if len(CASE_SECS) >= 20:
                RUN_STATS["sec_per_case_latency_p95"] = round(
                    _st.quantiles(CASE_SECS, n=20)[18], 2)
            RUN_STATS["sec_per_case_latency_max"] = round(max(CASE_SECS), 2)
        RUN_STATS.update(USAGE)
        RUN_STATS["est_cost_usd"] = round(usage_cost(), 4)
        RUN_STATS["est_cost_lost_max_usd"] = round(lost_cost_upper_bound(), 4)
        out_df = round_year_cols(out_df)          # emit at the GT's 2-decimal precision
        out_df.to_csv(OUT_BATCH_CSV, index=False)
        print(f"Saved batch CSV ({OUTPUT_DECIMALS} decimals):", OUT_BATCH_CSV)
        if SAVE_PROMPTS and PROMPT_STORE:
            with open(OUT_PROMPTS_JSONL, "w", encoding="utf-8") as _pf:
                for _i in sorted(PROMPT_STORE):
                    _pf.write(json.dumps({"id": _i, "prompt": PROMPT_STORE[_i]},
                                         ensure_ascii=False) + "\n")
            print(f"Saved prompts ({len(PROMPT_STORE)} cases):", OUT_PROMPTS_JSONL)
        _c = RUN_STATS["est_cost_usd"]
        print(f"\n[REPORT] {_n_cases} cases @ concurrency {MAX_CONCURRENCY}")
        print(f"  1. total cost            ${_c:,.2f}")
        print(f"  2. cost per case         ${_c / _n_cases:.5f}")
        print(f"  3. total wall clock      {RUN_STATS['elapsed_sec']:,.0f}s "
              f"({RUN_STATS['elapsed_sec'] / 60:.1f} min)")
        print(f"  4. wall clock per case   "
              f"{RUN_STATS.get('sec_per_case_throughput', float('nan')):.2f}s "
              f"(batch rate at concurrency {MAX_CONCURRENCY}; not a per-case cost)")
        print(f"  5. sec per trajectory    "
              f"{RUN_STATS.get('sec_per_case_latency_p50', float('nan')):.2f}s "
              f"(median, one case start to finish, concurrency-independent)")
        print(f"[DIAG] calls {USAGE['calls']} | tokens in {USAGE['tokens_in']:,} / "
              f"out {USAGE['tokens_out']:,} / thinking {USAGE['tokens_thinking']:,} | "
              f"latency mean {RUN_STATS.get('sec_per_case_latency_measured', float('nan')):.2f} "
              f"p95 {RUN_STATS.get('sec_per_case_latency_p95', float('nan')):.1f} "
              f"max {RUN_STATS.get('sec_per_case_latency_max', float('nan')):.1f} | "
              f"prices ${PRICE_IN}/M in, ${PRICE_OUT}/M out")
        if FAILED_CASES:
            print(f"\n[FAILED] {len(FAILED_CASES)} of {_n_cases} cases exhausted "
                  f"RETRY_MAX and are MISSING from the CSV above:")
            for _cid, _cerr in FAILED_CASES[:20]:
                print(f"    id={_cid}  {_cerr}")
            if len(FAILED_CASES) > 20:
                print(f"    ... and {len(FAILED_CASES) - 20} more")
        if USAGE["failed_calls"]:
            print(f"[RUN STATS] failed calls {USAGE['failed_calls']} "
                  f"(503/rate {USAGE['failed_503']} | disconnect {USAGE['failed_disconnect']} "
                  f"| other {USAGE['failed_other']}) | lost input est "
                  f"{USAGE['tokens_lost_est']:,} tok | un-tracked billing "
                  f"<= ${RUN_STATS['est_cost_lost_max_usd']:.4f}")
        if FAILED_CASES:
            raise SystemExit(
                f"\nABORT: {len(FAILED_CASES)} case(s) missing -> the batch is "
                f"incomplete, so run_evaluation() is SKIPPED (scoring it would "
                f"silently shrink the denominator and break comparability with "
                f"the other runs). The successful rows ARE saved above; rerun "
                f"the failed ids and merge, or rerun the whole batch.")

    else:
        raise ValueError("MODE must be 'single' or 'batch'.")


"""# validation (phase1)"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ============================================================
# PATHS
# ============================================================
# Evaluate the CSV from THIS run by default; override with IAM_LLM_PRED_PATH
# to re-evaluate an older prediction file.
PRED_PATH = os.environ.get("IAM_LLM_PRED_PATH", OUT_BATCH_CSV)


OUT_DIR   = OUTPUTS_DIR
# Eval report name. For THIS run it mirrors the pred CSV (RUN_TAG). When an
# external CSV is re-scored via IAM_LLM_PRED_PATH (older run, finetune output),
# RUN_TAG would describe the RAG config, not the file being scored - so derive
# the name from the pred file instead: <pred stem>_eval.xlsx next to it.
# IAM_LLM_EVAL_PATH overrides either.
if os.environ.get("IAM_LLM_EVAL_PATH"):
    OUT_XLSX = os.environ["IAM_LLM_EVAL_PATH"]
elif os.environ.get("IAM_LLM_PRED_PATH"):
    OUT_XLSX = os.path.splitext(PRED_PATH)[0] + "_eval.xlsx"
else:
    OUT_XLSX = os.path.join(OUT_DIR, f"{DATASET_TAG}_eval_{RUN_TAG}.xlsx")

# ============================================================
# HELPERS
# ============================================================
def norm_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def find_year_cols(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        cs = str(c).strip()
        if cs.isdigit():
            y = int(cs)
            if 1900 <= y <= 2200:
                cols.append(cs)
    cols = sorted(cols, key=lambda s: int(s))
    return cols

def smape(y_true, y_pred, eps=1e-12):
    """
    sMAPE. Points where both true and pred are 0 contribute near-zero (via eps).
    This keeps all points in the mean, pulling down overall sMAPE by the zero-error samples.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(denom, eps)) * 100.0)

def safe_corr(x, y, method="pearson"):
    """
    Correlation on already-filtered 1D arrays.  x = GROUND TRUTH, y = PREDICTION.

    Degenerate series follow scipy's own convention, so the numbers are
    comparable with correlations reported anywhere else:

    - n < 3                -> nan (undefined, dropped from the mean)
    - GT constant          -> nan.  The denominator is var(GT) = 0; there is no
      trend in the target to be predicted, so the case carries no information
      about model skill.
    - PREDICTION constant  -> nan.  The correlation is undefined from this side
      too.

    Only pearson/spearman are affected; every other metric is unchanged.
    Returns (corr, n)
    """
    x2 = np.asarray(x, dtype=float)
    y2 = np.asarray(y, dtype=float)
    n = x2.size

    # NB: constancy is tested with ptp (max == min), NOT std == 0.  np.std of
    # ten identical 0.03 values is 6.9e-18, not 0.0, so a std-based test misses
    # genuinely flat series; scipy then applies its own exact-equality check and
    # returns nan.  ptp == 0 is exactly scipy's criterion, so the two agree.
    if n < 3 or np.ptp(x2) == 0 or np.ptp(y2) == 0:
        return np.nan, n

    if method == "pearson":
        return pearsonr(x2, y2)[0], n
    elif method == "spearman":
        return spearmanr(x2, y2)[0], n
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")

def prepare_arrays_gt_only(y_true, y_pred):
    """
    Final evaluation mode:
    - keep only GT finite points
    - if pred is NaN at kept points, treat as 0
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(yt)
    yt2 = yt[mask]
    yp2 = yp[mask]
    yp2 = np.where(np.isfinite(yp2), yp2, 0.0)

    return yt2, yp2

def compute_metrics_flat(y_true, y_pred):
    """
    Final evaluation mode:
    - keep only GT finite points
    - pred NaN -> 0
    """
    yt, yp = prepare_arrays_gt_only(y_true, y_pred)

    n = yt.size
    if n == 0:
        return {
            "n_points": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "smape_%": np.nan,
            "pearson": np.nan,
            "spearman": np.nan,
        }

    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    sm = float(smape(yt, yp))
    pr, _ = safe_corr(yt, yp, "pearson")
    sr, _ = safe_corr(yt, yp, "spearman")

    return {
        "n_points": int(n),
        "mae": mae,
        "rmse": rmse,
        "smape_%": sm,
        "pearson": float(pr) if np.isfinite(pr) else np.nan,
        "spearman": float(sr) if np.isfinite(sr) else np.nan,
    }

def compute_metrics_timeseries_row(row, years):
    """
    Compute metrics for ONE (id, variable) trajectory across time.
    Final mode:
    - keep only GT finite points
    - pred NaN -> 0
    """
    yt = np.array([row[f"{y}_true"] for y in years], dtype=float)
    yp = np.array([row[f"{y}_pred"] for y in years], dtype=float)
    m = compute_metrics_flat(yt, yp)
    return pd.Series(m)

def summarize_ts_metrics(df_ts, group_cols):
    """
    Summarize trajectory-level metrics.
    Each row in df_ts is one (id, variable) trajectory.
    """
    rows = []
    for key, sub in df_ts.groupby(group_cols, dropna=False):
        rec = {}
        if isinstance(key, tuple):
            rec.update(dict(zip(group_cols, key)))
        else:
            rec[group_cols[0]] = key

        rec["n_series"] = int(len(sub))
        rec["avg_n_points"] = float(sub["n_points"].mean()) if len(sub) else np.nan
        rec["mae"] = float(sub["mae"].mean()) if len(sub) else np.nan
        rec["rmse"] = float(sub["rmse"].mean()) if len(sub) else np.nan
        rec["smape_%"] = float(sub["smape_%"].mean()) if len(sub) else np.nan
        # Each correlation carries the two counts it was averaged over: n_series
        # is every trajectory in the group (the theoretical denominator) and
        # n_valid_* is how many were actually defined.  They differ because a
        # constant GT or a constant prediction leaves the correlation undefined,
        # so pearson/spearman are means over a SMALLER set than mae/rmse/smape,
        # and the two blocks of metrics are only comparable once you can see by
        # how much.
        rec["pearson"] = float(sub["pearson"].mean(skipna=True)) if len(sub) else np.nan
        rec["n_valid_pearson"] = int(sub["pearson"].notna().sum())
        rec["pct_valid_pearson"] = (round(rec["n_valid_pearson"] / len(sub) * 100, 2)
                                    if len(sub) else np.nan)
        rec["spearman"] = float(sub["spearman"].mean(skipna=True)) if len(sub) else np.nan
        rec["n_valid_spearman"] = int(sub["spearman"].notna().sum())
        rec["pct_valid_spearman"] = (round(rec["n_valid_spearman"] / len(sub) * 100, 2)
                                     if len(sub) else np.nan)

        rows.append(rec)

    out = pd.DataFrame(rows)
    out = out.sort_values(group_cols).reset_index(drop=True)
    return out

def assert_no_dupe_keys(df, side_name, key_cols, show_n=20):
    dup = df.duplicated(subset=key_cols, keep=False)
    if dup.any():
        top = df.loc[dup, key_cols].value_counts().head(show_n)
        raise ValueError(
            f"[ERROR] {side_name}: duplicated keys found for {key_cols}. "
            f"Refusing to aggregate.\nTop duplicated keys:\n{top.to_string()}"
        )

def assert_exact_key_match(pred_df, gt_df, key_cols, show_n=20):
    pred_keys = set(map(tuple, pred_df[key_cols].to_numpy()))
    gt_keys   = set(map(tuple, gt_df[key_cols].to_numpy()))

    only_in_pred = sorted(pred_keys - gt_keys)
    only_in_gt   = sorted(gt_keys - pred_keys)

    if only_in_pred or only_in_gt:
        msg = ["[ERROR] PRED and GT keys do not match exactly."]
        msg.append(f"PRED unique keys: {len(pred_keys)}")
        msg.append(f"GT unique keys: {len(gt_keys)}")
        msg.append(f"Only in PRED: {len(only_in_pred)}")
        msg.append(f"Only in GT: {len(only_in_gt)}")

        if only_in_pred:
            msg.append("\nExamples only in PRED:")
            msg.extend([str(x) for x in only_in_pred[:show_n]])

        if only_in_gt:
            msg.append("\nExamples only in GT:")
            msg.extend([str(x) for x in only_in_gt[:show_n]])

        raise ValueError("\n".join(msg))

# ============================================================
# LOAD DATA
# ============================================================
def run_evaluation():
    print_resolved_config()
    print(f"  scoring: {Path(PRED_PATH).name}\n       vs: {Path(GT_PATH).name}\n")
    pred = pd.read_csv(PRED_PATH)
    gt   = pd.read_excel(GT_PATH)

    pred.columns = [str(c).strip() for c in pred.columns]
    gt.columns   = [str(c).strip() for c in gt.columns]

    # The pipeline predicts TRAJECTORIES ONLY, so the prediction CSV carries no
    # category column. The GT keeps category_c (C1..C8) and c_group as metadata,
    # used here purely to stratify the trajectory metrics (and later by the
    # standalone c-category classifier).
    required_base = ["id", "region", "model_family", "scenario", "variable", "unit"]
    CAT_META = ["category_c", "c_group"]          # GT-side only, optional
    for name, d in [("PRED", pred), ("GT", gt)]:
        miss = [c for c in required_base if c not in d.columns]
        if miss:
            raise KeyError(f"{name} is missing required columns: {miss}")

    for d in [pred, gt]:
        d["id"] = pd.to_numeric(d["id"], errors="coerce").astype("Int64")
        for c in required_base[1:]:
            d[c] = d[c].apply(norm_str)

    cat_meta = [c for c in CAT_META if c in gt.columns]
    for c in cat_meta:
        gt[c] = gt[c].astype(str).str.strip().str.lower()

    pred_years = find_year_cols(pred)
    gt_years   = find_year_cols(gt)
    years = [y for y in pred_years if y in gt_years]
    if len(years) == 0:
        raise ValueError(f"No overlapping year columns found.\nPRED years: {pred_years}\nGT years: {gt_years}")

    pred2 = pred[required_base + years].copy()
    gt2   = gt[required_base + cat_meta + years].copy()

    # Partial runs (TEST_LIMIT): evaluate only the ids that were predicted.
    _pred_ids = set(pred2["id"].astype(int).tolist())
    _gt_ids   = set(gt2["id"].astype(int).tolist())
    if _pred_ids < _gt_ids:
        print(f"[NOTE] Partial run: evaluating {len(_pred_ids)}/{len(_gt_ids)} test ids.")
        gt2 = gt2[gt2["id"].astype(int).isin(_pred_ids)].copy()

    for y in years:
        pred2[y] = pd.to_numeric(pred2[y], errors="coerce")
        gt2[y]   = pd.to_numeric(gt2[y], errors="coerce")

    # Score the predictions at the same precision the GT is stored at, whatever
    # produced the CSV (this run, an older run re-scored via IAM_LLM_PRED_PATH,
    # or the finetune inference output).  The GT is deliberately NOT touched.
    pred2 = round_year_cols(pred2, years)

    # ============================================================
    # STRICT DUPLICATE CHECK
    # ============================================================
    key_cols = ["id", "variable"]
    assert_no_dupe_keys(pred2, "PRED", key_cols)
    assert_no_dupe_keys(gt2, "GT", key_cols)

    # ============================================================
    # STRICT KEY MATCH CHECK
    # ============================================================
    assert_exact_key_match(pred2, gt2, key_cols)

    print(f"Matched rows (id, variable): {len(pred2)}")
    print(f"PRED unique keys: {len(pred2)} | GT unique keys: {len(gt2)}")

    # ============================================================
    # MERGE
    # ============================================================
    merged = pred2.merge(
        gt2,
        on=key_cols,
        how="outer",
        suffixes=("_pred", "_true"),
        indicator=True
    )

    if not (merged["_merge"] == "both").all():
        bad = merged.loc[merged["_merge"] != "both", key_cols + ["_merge"]].head(20)
        raise ValueError(
            "[ERROR] Unexpected non-matching rows found after strict key check.\n"
            f"{bad.to_string(index=False)}"
        )

    merged = merged.drop(columns=["_merge"])

    # ============================================================
    # TRUE TIME-SERIES METRICS
    # ============================================================
    ts_meta = [
        "id", "variable",
        "region_true", "model_family_true", "scenario_true",
        "region_pred", "model_family_pred", "scenario_pred",
    ] + cat_meta          # category_c / c_group come from the GT side only
    df_ts = merged[ts_meta].copy()

    ts_metrics = merged.apply(lambda row: compute_metrics_timeseries_row(row, years), axis=1)
    df_ts = pd.concat([df_ts, ts_metrics], axis=1)

    # overall time-series summary
    overall_timeseries = pd.DataFrame([{
        "scope": "ALL_timeseries_mean_over_(id,variable)_gt_only",
        "n_series": int(len(df_ts)),
        "avg_n_points": float(df_ts["n_points"].mean()) if len(df_ts) else np.nan,
        "mae": float(df_ts["mae"].mean()) if len(df_ts) else np.nan,
        "rmse": float(df_ts["rmse"].mean()) if len(df_ts) else np.nan,
        "smape_%": float(df_ts["smape_%"].mean()) if len(df_ts) else np.nan,
        "pearson": float(df_ts["pearson"].mean(skipna=True)) if len(df_ts) else np.nan,
        "n_valid_pearson": int(df_ts["pearson"].notna().sum()),
        "pct_valid_pearson": (round(df_ts["pearson"].notna().sum() / len(df_ts) * 100, 2)
                              if len(df_ts) else np.nan),
        "spearman": float(df_ts["spearman"].mean(skipna=True)) if len(df_ts) else np.nan,
        "n_valid_spearman": int(df_ts["spearman"].notna().sum()),
        "pct_valid_spearman": (round(df_ts["spearman"].notna().sum() / len(df_ts) * 100, 2)
                               if len(df_ts) else np.nan),
    }])

    # grouped summaries
    by_variable_ts    = summarize_ts_metrics(df_ts, ["variable"])
    by_modelfamily_ts = summarize_ts_metrics(df_ts, ["model_family_true"])
    by_region_ts      = summarize_ts_metrics(df_ts, ["region_true"])
    # stratified trajectory metrics by the GT category metadata (never predicted)
    by_cgroup_ts   = (summarize_ts_metrics(df_ts, ["c_group"])
                      if "c_group" in df_ts.columns else pd.DataFrame())
    by_category_ts = (summarize_ts_metrics(df_ts, ["category_c"])
                      if "category_c" in df_ts.columns else pd.DataFrame())

    by_var_family_ts  = summarize_ts_metrics(df_ts, ["model_family_true", "variable"])
    by_var_region_ts  = summarize_ts_metrics(df_ts, ["region_true", "variable"])

    # append run cost stats to the overall summary row
    _pos = overall_timeseries.columns.get_loc("pct_valid_spearman") + 1
    for _name, _val in [
        ("tokens_in",        RUN_STATS.get("tokens_in", USAGE["tokens_in"])),
        ("tokens_out",       RUN_STATS.get("tokens_out", USAGE["tokens_out"])),
        ("est_cost_usd",     RUN_STATS.get("est_cost_usd", round(usage_cost(), 4))),
    ]:
        overall_timeseries.insert(_pos, _name, _val)
        _pos += 1

    # ============================================================
    # WRITE EXCEL REPORT
    # ============================================================
    os.makedirs(OUT_DIR, exist_ok=True)

    def _timing_rows(n_cases):
        """Per-case timing for the run_config sheet.

          sec_per_case_throughput = wall clock / cases. It scales with the
            concurrency and therefore describes the deployment.

          sec_per_case_latency = measured mean wall time of one case, from the
            moment it acquires a concurrency slot to the moment it returns
            (queue wait excluded, retry sleeps included). It does not depend on
            the concurrency and describes the method. Collected in run_batch
            via CASE_SECS.

        When no measurement exists, the latency is derived as throughput x
        effective concurrency and reported as sec_per_case_latency_derived.
        """
        el = RUN_STATS.get("elapsed_sec")
        if not el or not n_cases:
            return []
        eff = min(MAX_CONCURRENCY, n_cases)
        thr = el / n_cases
        calls = RUN_STATS.get("calls", USAGE["calls"]) or 0
        lat = RUN_STATS.get("sec_per_case_latency_measured")
        p50 = RUN_STATS.get("sec_per_case_latency_p50")
        cost = RUN_STATS.get("est_cost_usd")

        # ---- the five REPORTED numbers (rpt_*) -------------------------------
        # Everything below them is diagnostics. rpt_sec_per_case_batch is only
        # meaningful next to max_concurrency, which run_config already carries.
        rows = [
            ("rpt_total_cost_usd",        cost),
            ("rpt_cost_per_case_usd",     round(cost / n_cases, 5) if cost else None),
            ("rpt_total_wallclock_sec",   round(el, 1)),
            ("rpt_sec_per_case_batch",    round(thr, 3)),
            ("rpt_sec_per_trajectory",    p50 if p50 is not None else lat),
        ]
        # ---- diagnostics ----------------------------------------------------
        rows.append(("effective_concurrency", eff))
        if lat is not None:
            rows.append(("diag_latency_mean", lat))
            rows.append(("diag_latency_p95",
                         RUN_STATS.get("sec_per_case_latency_p95")))
            rows.append(("diag_latency_max",
                         RUN_STATS.get("sec_per_case_latency_max")))
            # slot utilisation: 1.0 = the pool stayed full. Well under 1 means a
            # few slow cases (usually retries) held the wall clock open while
            # most workers idled - that is what makes rpt_sec_per_case_batch
            # worse than rpt_sec_per_trajectory / concurrency.
            rows.append(("diag_slot_utilisation",
                         round((n_cases * lat) / (el * eff), 3)))
        else:
            rows.append(("diag_latency_derived", round(thr * eff, 2)))
        if calls:
            rows.append(("diag_calls_per_case", round(calls / n_cases, 3)))
        return rows

    run_config = pd.DataFrame(
        [
            ("run_datetime",          _dt.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("provider",              PROVIDER),
            ("llm_model",             MODEL_NAME if PROVIDER == "openai" else GEMINI_MODEL_NAME),
            ("temperature",           TEMPERATURE),
            ("topk_scen",             TOPK_SCEN),
            ("topk_model",            TOPK_MODEL),
            ("topk_cross",            TOPK_CROSS),
            ("run_step3",             RUN_STEP3),
            ("fingerprint_path",      FINGERPRINT_PATH),
            ("fingerprint_sheet",     FP_SHEET_NAME),
            ("family_variable_scope", "core x22 | COFFEE/WITCH x11 supply | GCAM/TIAM x11 demand"),
            ("step2_family_pool",     "core->core-5 only | supply-only->own pair + core-5 | "
                                      "demand-only->own pair + core-5 (borrowed core neighbors "
                                      "shown on target-scope variables only)"),
            ("scenario_desc_used",    INCLUDE_SCENARIO_DESC),
            ("v_rerank_mode",         RERANK_MODE),
            ("rerank_pool_scen",      RERANK_POOL_SCEN if RERANK_MODE != "off" else None),
            ("rerank_pool_model",     RERANK_POOL_MODEL if RERANK_MODE == "llm" else None),
            ("rerank_blend",          RERANK_BLEND if RERANK_MODE == "hybrid" else None),
            ("sample_n",              SAMPLE_N),
            ("sample_seed",           SAMPLE_SEED if SAMPLE_N else None),
            ("embed_model",           EMBED_MODEL),
            ("test_list",             BATCH_SHEET_PATH),
            ("test_gt",               GT_PATH),
            ("train_gt",              EXCEL_PATH),
            ("pred_csv",              PRED_PATH),
            ("neighbor_pool",         "train_only" if not INCLUDE_EVAL_GT_IN_NEIGHBOR_POOL else "train+eval_gt"),
            ("n_pred_ids",            int(pred2["id"].nunique())),
            ("max_concurrency",       MAX_CONCURRENCY),
            ("retry_max",             RETRY_MAX),
        ] + [(k, v) for k, v in RUN_STATS.items()]
          + _timing_rows(int(pred2["id"].nunique())),
        columns=["key", "value"],
    )

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        run_config.to_excel(w, index=False, sheet_name="run_config")
        overall_timeseries.to_excel(w, index=False, sheet_name="overall_timeseries")

        by_variable_ts.to_excel(w, index=False, sheet_name="by_variable_ts")
        by_modelfamily_ts.to_excel(w, index=False, sheet_name="by_model_family_ts")
        by_region_ts.to_excel(w, index=False, sheet_name="by_region_ts")
        if not by_cgroup_ts.empty:
            by_cgroup_ts.to_excel(w, index=False, sheet_name="by_c_group_ts")
        if not by_category_ts.empty:
            by_category_ts.to_excel(w, index=False, sheet_name="by_category_c_ts")

        by_var_family_ts.to_excel(w, index=False, sheet_name="by_family_x_variable_ts")
        by_var_region_ts.to_excel(w, index=False, sheet_name="by_region_x_variable_ts")

        df_ts.to_excel(w, index=False, sheet_name="trajectory_metrics")

        merged_out_cols = [
            "id", "variable",
            "region_true", "model_family_true", "scenario_true", "unit_true",
            "region_pred", "model_family_pred", "scenario_pred", "unit_pred",
        ] + cat_meta
        for y in years:
            merged_out_cols += [f"{y}_true", f"{y}_pred"]
        merged[merged_out_cols].to_excel(w, index=False, sheet_name="merged_wide")

    print(f"\nDone. Report saved to:\n{OUT_XLSX}")

    print("\nOverall timeseries metrics (mean over each (id, variable) trajectory):")
    print(overall_timeseries.to_string(index=False))


if __name__ == "__main__" and MODE == "batch":
    run_evaluation()
