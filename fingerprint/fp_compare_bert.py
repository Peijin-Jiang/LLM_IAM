"""
fp_compare_bert.py
──────────────────
Reproducible BERT similarity between fingerprint files.

Given two or more fingerprint xlsx files (the "fingerprint" sheet with columns
Model family | Energy System Preference), computes for EVERY pair of
files, per model family:
  - sentence cosine similarity  (mean-pooled MiniLM embeddings)
  - BERTScore P/R/F1 via the OFFICIAL bert-score package (roberta-large),
    i.e. the standard configuration of Zhang et al. (ICLR 2020)
for "Energy System Preference", plus an
exact-match flag. If both files carry a "consensus" sheet (structured
pipeline), the exact-agreement rate of the consensus tables is also computed.

Models:
  - cosine   : sentence-transformers/all-MiniLM-L6-v2 (FP_BERT_MODEL)
  - BERTScore: roberta-large via bert-score (FP_BERTSCORE_MODEL)
BERTScore is computed with the official package, so the values are directly
comparable with scenario_desc_repeat_eval.py.

Output -> outputs/fp_bert_compare/fp_bert_compare_<stamp>.xlsx
  sheet "summary"     : per-pair means over the 9 families
  sheet "per_family"  : every pair x field x family row
  sheet "consensus"   : per-pair consensus-table agreement (when available)
  sheet "run_config"

Usage (from this folder):
    python3 fp_compare_bert.py FILE1.xlsx FILE2.xlsx [FILE3.xlsx ...]
    python3 fp_compare_bert.py --labels textA,textB FILE1.xlsx FILE2.xlsx

Env:
    FP_BERT_MODEL       cosine model (default "sentence-transformers/all-MiniLM-L6-v2")
    FP_BERTSCORE_MODEL  BERTScore model (default "roberta-large")

Requires:  pip install torch transformers bert-score
           (first run downloads both models)
"""

import argparse
import os
from datetime import datetime
from itertools import combinations
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
BERT_MODEL = os.environ.get("FP_BERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BERTSCORE_MODEL = os.environ.get("FP_BERTSCORE_MODEL", "roberta-large")
OUT_DIR = _ROOT / "outputs" / "fp_bert_compare"

PREF_FIELD = "Energy System Preference"
COMPONENT_FIELDS = [PREF_FIELD]


# One aggregation spec, reused for the per-field rows and the pooled row so the
# two can never drift apart. n_items = number of scored (family, field) cells:
# = n_families for a per-field row, = 2 x n_families for the pooled row.
_SUMMARY_AGG = dict(
    n_items=("bertscore_F1", "size"),
    n_families=("family", "nunique"),
    mean_cosine=("cosine", "mean"),
    mean_P=("bertscore_P", "mean"),
    mean_R=("bertscore_R", "mean"),
    mean_F1=("bertscore_F1", "mean"),
    min_F1=("bertscore_F1", "min"),
    n_identical=("identical", "sum"),
)


_tok = _model = None


def _load_model():
    global _tok, _model
    if _model is None:
        import torch
        from transformers import AutoTokenizer, AutoModel
        _tok = AutoTokenizer.from_pretrained(BERT_MODEL)
        _model = AutoModel.from_pretrained(BERT_MODEL)
        _model.eval()
    return _tok, _model


def encode(texts):
    """Sentence embeddings [N,H], mean-pooled and L2-normalized (for cosine)."""
    import torch
    tok, model = _load_model()
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    return torch.nn.functional.normalize((out * mask).sum(1) / mask.sum(1), dim=-1)


# ============================================================
# comparison
# ============================================================
def load_fingerprint(path):
    df = pd.read_excel(path, sheet_name="fingerprint").set_index("Model family")
    return df


def load_consensus(path):
    """Consensus table as {(family, variable): (level, comparison)}.

    'level' is absent from the comparison-only fingerprints (they carry
    Energy_System_Preference_vs_other_models + n_comp_votes only), so it is read
    with .get and falls back to "". Two such files are then compared on the
    comparison alone."""
    try:
        c = pd.read_excel(path, sheet_name="consensus")
    except Exception:
        return None
    if "Energy_System_Preference_vs_other_models" not in c.columns:
        return None
    return {(r["Model family"], r["variable"]):
            (str(r.get("level", "")), str(r.get("Energy_System_Preference_vs_other_models", "")))
            for _, r in c.iterrows()}


def compare_pair(label, dfa, dfb):
    from bert_score import score as bertscore_score
    def _txt(df, f, field):
        v = df.loc[f, field] if field in df.columns else ""
        return "" if pd.isna(v) else str(v).strip()

    fams = [f for f in dfa.index if f in dfb.index]

    texts = {}
    for field in COMPONENT_FIELDS:
        ta = [_txt(dfa, f, field) for f in fams]
        tb = [_txt(dfb, f, field) for f in fams]
        # empty strings crash bert-score; the fingerprint files write "Not Available"
        # instead of an empty cell
        bad = [f for f, a, b in zip(fams, ta, tb) if not a or not b]
        if bad:
            raise SystemExit(
                f"empty '{field}' for {bad} - this fingerprint file predates "
                "the 'Not Available' convention; regenerate it (or fill the "
                "empty cells with 'Not Available') and rerun.")
        texts[field] = (ta, tb)
    rows = []
    for field, (ta, tb) in texts.items():
        ea, eb = encode(ta), encode(tb)
        P, R, F1 = bertscore_score(ta, tb, lang="en",
                                   model_type=BERTSCORE_MODEL, verbose=False)
        for i, f in enumerate(fams):
            rows.append({
                "pair": label, "field": field, "family": f,
                "cosine": round(float(ea[i] @ eb[i]), 4),
                "bertscore_P": round(float(P[i]), 4),
                "bertscore_R": round(float(R[i]), 4),
                "bertscore_F1": round(float(F1[i]), 4),
                "identical": ta[i] == tb[i],
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="fingerprint xlsx files (2 or more)")
    ap.add_argument("--labels", default="",
                    help="comma-separated short labels, same order as files")
    args = ap.parse_args()
    if len(args.files) < 2:
        raise SystemExit("Give at least two fingerprint xlsx files.")

    paths = [Path(f) for f in args.files]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"File not found: {p}")
    labels = ([s.strip() for s in args.labels.split(",")] if args.labels
              else [p.stem.replace("fingerprint_from_papers_", "") for p in paths])
    if len(labels) != len(paths):
        raise SystemExit("--labels count must match the number of files")

    dfs = {lab: load_fingerprint(p) for lab, p in zip(labels, paths)}
    cons = {lab: load_consensus(p) for lab, p in zip(labels, paths)}

    per_rows, cons_rows = [], []
    for (la, pa), (lb, pb) in combinations(zip(labels, paths), 2):
        label = f"{la} vs {lb}"
        print(f"comparing: {label}")
        per_rows += compare_pair(label, dfs[la], dfs[lb])
        if cons[la] is not None and cons[lb] is not None:
            keys = set(cons[la]) | set(cons[lb])
            exact = sum(1 for k in keys if cons[la].get(k) == cons[lb].get(k))
            cons_rows.append({"pair": label, "entries_a": len(cons[la]),
                              "entries_b": len(cons[lb]), "union": len(keys),
                              "identical": exact,
                              "agreement": round(exact / len(keys), 4) if keys else None})

    detail = pd.DataFrame(per_rows)

    summary = (detail.groupby(["pair", "field"], sort=False)
               .agg(**_SUMMARY_AGG).round(4).reset_index())

    print("\n" + summary.to_string(index=False))
    if cons_rows:
        print("\nconsensus-table agreement:")
        print(pd.DataFrame(cons_rows).to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    out_xlsx = OUT_DIR / f"fp_bert_compare_{stamp}.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        summary.to_excel(w, index=False, sheet_name="summary")
        detail.to_excel(w, index=False, sheet_name="per_family")
        if cons_rows:
            pd.DataFrame(cons_rows).to_excel(w, index=False, sheet_name="consensus")
        pd.DataFrame(
            [("cosine_model", BERT_MODEL),
             ("bertscore_model", BERTSCORE_MODEL),
             ("bertscore_impl", "official bert-score package (Zhang et al. 2020)")]
            + [(f"file_{lab}", str(p)) for lab, p in zip(labels, paths)]
            + [("run_datetime", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))],
            columns=["key", "value"]).to_excel(w, index=False, sheet_name="run_config")
    print(f"\nSaved: {out_xlsx}")


if __name__ == "__main__":
    main()
