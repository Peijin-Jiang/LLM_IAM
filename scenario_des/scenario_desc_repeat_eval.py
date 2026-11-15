"""
scenario_desc_repeat_eval.py
────────────────────────────
Compare EXISTING scenario-description outputs with BERTScore F1 - no
regeneration. Reads the "descriptions" sheet of each xlsx and scores every
pair of files (robustness / pure LLM noise).
Every unique scenario is counted exactly once (on the common set of each pair).

Output -> outputs/scenario_desc/scenario_desc_repeat_eval_<stamp>.xlsx
  sheet "summary"      : comparison | n_scenarios | mean_F1 | min_F1 | mean_P | mean_R
  sheet "per_scenario" : comparison | scenario | P | R | F1
  sheet "files"        : the xlsx files that were compared
  sheet "run_config"

Usage (from this folder):
    python3 scenario_desc_repeat_eval.py
        -> auto-picks the NEWEST 2 scenario_desc_gen_*.xlsx in outputs/scenario_desc
    python3 scenario_desc_repeat_eval.py FILE1.xlsx FILE2.xlsx [FILE3.xlsx ...]
        -> compares exactly these files (2 or more)

Env overrides:
    SD_N_FILES     how many newest files to auto-pick when none are given (default 2)
    SD_BERT_MODEL  BERTScore model (default "roberta-large")

Requires:  pip install bert-score  (first run downloads the BERT model)
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
OUT_DIR = _ROOT / "outputs" / "scenario_desc"

N_FILES = int(os.environ.get("SD_N_FILES", "2"))
BERT_MODEL = os.environ.get("SD_BERT_MODEL", "roberta-large")


def norm(x):
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def pick_files(argv):
    """Explicit paths from the command line, or the newest N_FILES
    scenario_desc_gen_*.xlsx in OUT_DIR."""
    if argv:
        files = [Path(a) for a in argv]
    else:
        cand = [p for p in OUT_DIR.glob("scenario_desc_gen_*.xlsx")
                if not p.name.startswith("~$")]
        cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        files = sorted(cand[:N_FILES], key=lambda p: p.stat().st_mtime)  # oldest first
    for f in files:
        if not f.exists():
            raise SystemExit(f"file not found: {f}")
    if len(files) < 2:
        raise SystemExit(f"need at least 2 files to compare, got {len(files)} "
                         f"(searched {OUT_DIR})")
    return files


def short_label(path):
    name = Path(path).stem
    return name.replace("scenario_desc_gen_", "")


def load_desc(path):
    """scenario -> non-empty description from the 'descriptions' sheet."""
    df = pd.read_excel(path, sheet_name="descriptions")
    out = {}
    for _, r in df.iterrows():
        s = norm(r.get("scenario"))
        d = str(r.get("description") or "").strip()
        if s and d and d.lower() != "nan":
            out[s] = d
    return out


def main():
    files = pick_files(sys.argv[1:])
    print("comparing files:")
    for f in files:
        print("  -", f)

    runs = {short_label(f): load_desc(f) for f in files}
    for lbl, d in runs.items():
        print(f"  {lbl}: {len(d)} descriptions")

    # ---- BERTScore comparisons ----
    from bert_score import score as bertscore_score

    def compare(name, d1, d2):
        common = sorted(s for s in d1 if s in d2)
        if not common:
            print(f"  (no common scenarios for {name})")
            return []
        P, R, F1 = bertscore_score([d1[s] for s in common], [d2[s] for s in common],
                                   lang="en", model_type=BERT_MODEL, verbose=False)
        return [{"comparison": name, "scenario": s,
                 "P": round(p, 4), "R": round(r, 4), "F1": round(f, 4)}
                for s, p, r, f in zip(common, P.tolist(), R.tolist(), F1.tolist())]

    labels = list(runs)
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            print(f"BERTScore: {a} vs {b} ...")
            rows += compare(f"{a} vs {b}", runs[a], runs[b])

    detail = pd.DataFrame(rows)
    summary = (detail.groupby("comparison")
               .agg(n_scenarios=("F1", "size"), mean_F1=("F1", "mean"),
                    min_F1=("F1", "min"), mean_P=("P", "mean"), mean_R=("R", "mean"))
               .round(4).reset_index())
    print("\n" + summary.to_string(index=False))

    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    out_xlsx = OUT_DIR / f"scenario_desc_repeat_eval_{stamp}.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        summary.to_excel(w, index=False, sheet_name="summary")
        detail.to_excel(w, index=False, sheet_name="per_scenario")
        pd.DataFrame({"file": [str(f) for f in files]}).to_excel(
            w, index=False, sheet_name="files")
        pd.DataFrame([
            ("mode", "compare existing xlsx files (no regeneration)"),
            ("bert_model", BERT_MODEL),
            ("n_files", len(files)),
            ("run_datetime", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ], columns=["key", "value"]).to_excel(w, index=False, sheet_name="run_config")
    print(f"\nSaved: {out_xlsx}")


if __name__ == "__main__":
    main()
