"""
scenario_desc_apply.py
──────────────────────
Replace the scenario descriptions in EVERY xlsx under make_datasets/output/
with the newly generated ones.

For each workbook and each sheet that has a "scenario" column together with a
"scenario_description" column, the description is replaced by the new one
wherever a new description exists for that scenario; scenarios without a new
description KEEP their old text (reported at the end). Originals are backed up
to make_datasets/output/backup_desc_<stamp>/ before anything is overwritten.

Usage (from this folder):
    python3 scenario_desc_apply.py                      # uses the newest gen file
    SD_NEW_DESC_XLSX=outputs/scenario_desc/....xlsx python3 scenario_desc_apply.py

Env overrides:
    SD_NEW_DESC_XLSX  the generation output to apply (sheet "descriptions",
                      columns scenario | description). Default: the newest
                      scenario_desc_gen_*.xlsx under outputs/scenario_desc/.
    SD_APPLY_DIR      folder whose xlsx files are updated
                      (default make_datasets/output)
"""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

APPLY_DIR = Path(os.environ.get("SD_APPLY_DIR", _ROOT / "make_datasets" / "output"))
NEW_XLSX = os.environ.get("SD_NEW_DESC_XLSX", "").strip()


def norm(x):
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def main():
    # ---- locate the generation output ----
    if NEW_XLSX:
        new_path = Path(NEW_XLSX)
        if not new_path.is_absolute():
            new_path = _ROOT / new_path
    else:
        cands = sorted((_ROOT / "outputs" / "scenario_desc").glob("scenario_desc_gen_*.xlsx"))
        if not cands:
            raise SystemExit("No scenario_desc_gen_*.xlsx found under outputs/scenario_desc/ "
                             "- run scenario_desc_generate.py first or set SD_NEW_DESC_XLSX.")
        new_path = cands[-1]
    if not new_path.exists():
        raise SystemExit(f"Generation output not found: {new_path}")

    nd = pd.read_excel(new_path, sheet_name="descriptions")
    desc_col = "description" if "description" in nd.columns else "description_new"
    new_map = {}
    for _, row in nd.iterrows():
        sc, d = norm(row["scenario"]), str(row.get(desc_col, "")).strip()
        if sc and d and d.lower() != "nan":
            new_map[sc] = d
    print(f"new descriptions: {len(new_map)} scenarios  (from {new_path.name})")

    # ---- backup then update every workbook in APPLY_DIR ----
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    backup_dir = APPLY_DIR / f"backup_desc_{stamp}"
    targets = [p for p in sorted(APPLY_DIR.glob("*.xlsx"))
               if not p.name.startswith(("~$", "backup_"))]
    if not targets:
        raise SystemExit(f"No xlsx files found in {APPLY_DIR}")

    total_replaced, kept_old = 0, set()
    for path in targets:
        sheets = pd.read_excel(path, sheet_name=None)
        touched = False
        report = []
        for sh, df in sheets.items():
            if "scenario" not in df.columns or "scenario_description" not in df.columns:
                continue
            scen = df["scenario"].map(norm)
            new_vals = scen.map(new_map)
            n_new = new_vals.notna().sum()
            if n_new == 0:
                continue
            kept = df.loc[new_vals.isna(), "scenario"].map(norm)
            kept_old.update(k for k in kept if k)
            df["scenario_description"] = new_vals.where(new_vals.notna(),
                                                        df["scenario_description"])
            sheets[sh] = df
            touched = True
            total_replaced += int(n_new)
            report.append(f"{sh}: {n_new}/{len(df)} rows updated")
        if not touched:
            print(f"  {path.name}: no scenario_description sheet - skipped")
            continue
        backup_dir.mkdir(exist_ok=True)
        shutil.copy2(path, backup_dir / path.name)
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            for sh, df in sheets.items():
                df.to_excel(w, index=False, sheet_name=sh)
        print(f"  {path.name}: " + "; ".join(report))

    print(f"\nreplaced {total_replaced} rows across {len(targets)} file(s)")
    if kept_old:
        print(f"scenarios keeping their OLD description (no new one): {len(kept_old)}")
        for s in sorted(kept_old)[:10]:
            print("   -", s)
    if backup_dir.exists():
        print(f"originals backed up to: {backup_dir}")
    print("\nNOTE: rag_core embedding caches are keyed by full text - new descriptions "
          "re-embed automatically on the next run.")


if __name__ == "__main__":
    main()
