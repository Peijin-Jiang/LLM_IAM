"""
build_r10_sd_mixed.py
─────────────────────
Build the analyzed dataset from the AR6 Scenarios Database, with variable
requirements that depend on the IAM family.

Pipeline:
  1. Download the 25 variables (14 supply + 11 demand), 10 R10 regions
     (Rest of the World excluded), years 2010-2100 (10-y steps), meta C1-C8.
  2. Completion rules (applied first, per combo = model-scenario-region):
       - 2010 missing but 2020 reported          -> 2010 = 2020
       - fossil total + ONE child reported       -> other child
                                                    = total - child (clip 0)
  3. Evaluate the 22 required variables (11 supply children/standalone +
     11 demand), then apply PER-FAMILY rules:
       - GCAM (GCAM 5.2 / GCAM 5.3 / GCAM-PR 5.3) and TIAM (TIAM-ECN 1.1):
         first CHECK that >=90% of their combos have incomplete supply
         (expected: GCAM ~93%, TIAM 100%). If confirmed -> keep combos with
         a complete 11-variable DEMAND set only (output = 11 demand rows).
       - COFFEE, WITCH: first CHECK that ~100% of their combos have
         incomplete demand. If confirmed -> keep combos with a complete
         11-variable SUPPLY set only (output = 11 supply rows).
       - AIM/CGE, GEM-E3, IMAGE, MESSAGEix-GLOBIOM, REMIND: report their
         supply/demand missing rates (expected well below 90%), keep only
         combos with ALL 22 variables (output = 22 rows).
       - any other model -> dropped (listed in the stats for transparency).
     If a CHECK fails, the family falls back to the all-22 rule and a
     WARNING is printed.
  4. Outputs (make_datasets/output/):
       AR6_R10_sd_mixed_<N>.xlsx      main dataset (scenario_description left
                                      empty; supply-only, demand-only and
                                      complete cases in the same sheet)
       build_stats_mixed.xlsx         statistics of the build

Usage (local machine, needs internet):
    python3 make_datasets/build_r10_sd_mixed.py
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyam

_HERE = Path(__file__).resolve().parent
OUT_DIR = _HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2010, 2101, 10))
YCOLS = [str(y) for y in YEARS]

SUPPLY_14 = [
    "Primary Energy|Coal", "Primary Energy|Coal|w/ CCS", "Primary Energy|Coal|w/o CCS",
    "Primary Energy|Gas", "Primary Energy|Gas|w/ CCS", "Primary Energy|Gas|w/o CCS",
    "Primary Energy|Oil", "Primary Energy|Oil|w/ CCS", "Primary Energy|Oil|w/o CCS",
    "Primary Energy|Solar", "Primary Energy|Wind", "Primary Energy|Hydro",
    "Primary Energy|Nuclear", "Primary Energy|Biomass",
]
DEMAND_11 = [
    "Final Energy|Liquids", "Final Energy|Gases",
    "Final Energy|Industry", "Final Energy|Transportation",
    "Final Energy|Residential and Commercial", "Final Energy|Hydrogen",
    "Final Energy|Heat", "Final Energy|Solids",
    "Final Energy|Industry|Electricity", "Final Energy|Transportation|Electricity",
    "Final Energy|Residential and Commercial|Electricity",
]
PARENTS = {
    "Primary Energy|Coal": ("Primary Energy|Coal|w/ CCS", "Primary Energy|Coal|w/o CCS"),
    "Primary Energy|Gas": ("Primary Energy|Gas|w/ CCS", "Primary Energy|Gas|w/o CCS"),
    "Primary Energy|Oil": ("Primary Energy|Oil|w/ CCS", "Primary Energy|Oil|w/o CCS"),
}
OTHER_SUPPLY = ["Primary Energy|Solar", "Primary Energy|Wind", "Primary Energy|Hydro",
                "Primary Energy|Nuclear", "Primary Energy|Biomass"]
SUPPLY_11 = [c for pair in PARENTS.values() for c in pair] + OTHER_SUPPLY

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

FAMILY_MAP = {
    "AIM/CGE 2.2": "AIM/CGE",
    "COFFEE 1.1": "COFFEE",
    "GEM-E3_V2021": "GEM-E3",
    "IMAGE 3.0": "IMAGE",
    "IMAGE 3.2": "IMAGE",
    "MESSAGEix-GLOBIOM_1.1": "MESSAGEix-GLOBIOM",
    "MESSAGEix-GLOBIOM_1.2": "MESSAGEix-GLOBIOM",
    "MESSAGEix-GLOBIOM_GEI 1.0": "MESSAGEix-GLOBIOM",
    "REMIND 2.1": "REMIND",
    "REMIND-Buildings 2.0": "REMIND",
    "REMIND-MAgPIE 1.7-3.0": "REMIND",
    "REMIND-MAgPIE 2.0-4.1": "REMIND",
    "REMIND-MAgPIE 2.1-4.2": "REMIND",
    "REMIND-MAgPIE 2.1-4.3": "REMIND",
    "REMIND-Transport 2.1": "REMIND",
    "WITCH 4.6": "WITCH",
    "WITCH 5.0": "WITCH",
    "GCAM 5.2": "GCAM",
    "GCAM 5.3": "GCAM",
    "GCAM-PR 5.3": "GCAM",
    "TIAM-ECN 1.1": "TIAM",
}

CORE_FAMILIES = ["AIM/CGE", "GEM-E3", "IMAGE", "MESSAGEix-GLOBIOM", "REMIND"]
DEMAND_ONLY_CANDIDATES = ["GCAM", "TIAM"]      # expect supply missing >=90%
SUPPLY_ONLY_CANDIDATES = ["COFFEE", "WITCH"]   # expect demand missing ~100%
CHECK_THRESHOLD = 0.90

KEY = ["model", "scenario", "region"]


def clean(s):
    s = s.astype("string").str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    # zero-width chars, BOM, NBSP, soft hyphen (explicit escapes on purpose)
    return s.str.replace("[\u200b\u200c\u200d\ufeff\u00a0\u00ad]", "", regex=True)


# ════════════════════════════════════════════════════════════════════════════
# 1. DOWNLOAD
# ════════════════════════════════════════════════════════════════════════════
print("downloading 25 variables x 10 R10 regions, meta C1-C8 ...")
df = pyam.read_iiasa("ar6-public", variable=SUPPLY_14 + DEMAND_11,
                     region=R10_REGIONS, year=YEARS, meta=True)

meta = df.meta.reset_index()
cat_col = next(c for c in meta.columns if c.lower() == "category")
meta["category_C"] = meta[cat_col].astype(str).str.strip()
meta = meta[meta["category_C"].isin([f"C{i}" for i in range(1, 9)])]

REF_COL = "Literature Reference (if applicable)"
lit = meta[["model", "scenario", "category_C"]].copy()
# clean() the keys here too: `d` below is cleaned, so an uncleaned `lit` would
# silently fail to line up with the final scenario names in section 7.
lit["model"] = clean(lit["model"])
lit["scenario"] = clean(lit["scenario"])
lit["literature_reference"] = (meta[REF_COL].astype(str).str.strip()
                               if REF_COL in meta.columns else "")

d = df.data.copy()
for c in ["model", "scenario", "region", "variable", "unit"]:
    d[c] = clean(d[c])
d = d.merge(meta[["model", "scenario", "category_C"]].drop_duplicates(),
            on=["model", "scenario"], how="inner")
d["model_family"] = d["model"].map(FAMILY_MAP).fillna("non-target")

# sanity: any GCAM/TIAM model versions we did not map?
unmapped = sorted(m for m in d.loc[d["model_family"] == "non-target", "model"].unique()
                  if "GCAM" in m.upper() or "TIAM" in m.upper())
if unmapped:
    print(f"WARNING: unmapped GCAM/TIAM-like models treated as non-target: {unmapped}")

n0 = d[KEY].drop_duplicates().shape[0]
print(f"original combos (C1-C8, any of the 25 variables): {n0}")

wide = d.pivot_table(index=KEY + ["model_family", "variable", "unit"],
                     columns="year", values="value", aggfunc="first").reset_index()
wide.columns = [str(c) for c in wide.columns]
for y in YCOLS:
    if y not in wide.columns:
        wide[y] = np.nan


# ════════════════════════════════════════════════════════════════════════════
# 2. PER-COMBO EVALUATION (completion rules + 22-variable availability)
# ════════════════════════════════════════════════════════════════════════════
def series_of(block, var, fills):
    """Complete 2010-2100 series or None; 2010 may be copied from 2020
    (recorded in fills as a '2010<-2020' event for that variable)."""
    r = block[block.variable == var]
    if r.empty:
        return None
    s = r.iloc[0][YCOLS].astype(float)
    missing = [y for y in YCOLS if pd.isna(s[y])]
    if not missing:
        return s
    if missing == ["2010"] and pd.notna(s["2020"]):
        s = s.copy()
        s["2010"] = s["2020"]
        fills.append(var)
        return s
    return None


records = []   # one dict per combo
for (m, sc, rg), block in wide.groupby(KEY):
    fam = block["model_family"].iloc[0]
    fills_2010 = []          # variables where 2010 was copied from 2020
    computed_children = []   # fossil children computed as total - sibling
    series = {}

    # supply with completion
    for parent, (c1, c2) in PARENTS.items():
        p = series_of(block, parent, fills_2010)
        s1 = series_of(block, c1, fills_2010)
        s2 = series_of(block, c2, fills_2010)
        if p is not None and s1 is not None and s2 is None:
            s2 = (p - s1).clip(lower=0)
            computed_children.append(c2)
        elif p is not None and s2 is not None and s1 is None:
            s1 = (p - s2).clip(lower=0)
            computed_children.append(c1)
        if s1 is not None:
            series[c1] = s1
        if s2 is not None:
            series[c2] = s2
    for v in OTHER_SUPPLY:
        s = series_of(block, v, fills_2010)
        if s is not None:
            series[v] = s

    # demand (no completion beyond the 2010 rule)
    for v in DEMAND_11:
        s = series_of(block, v, fills_2010)
        if s is not None:
            series[v] = s

    miss_supply = [v for v in SUPPLY_11 if v not in series]
    miss_demand = [v for v in DEMAND_11 if v not in series]

    records.append({
        "model": m, "scenario": sc, "region": rg, "model_family": fam,
        "supply_ok": not miss_supply, "demand_ok": not miss_demand,
        "miss_supply": miss_supply, "miss_demand": miss_demand,
        "computed_children": computed_children, "fills_2010": fills_2010,
        "series": series,
        "block": block,
    })

res = pd.DataFrame([{k: r[k] for k in
                     ["model", "scenario", "region", "model_family",
                      "supply_ok", "demand_ok"]} for r in records])

# ════════════════════════════════════════════════════════════════════════════
# 3. FAMILY CHECKS -> MODE PER FAMILY
# ════════════════════════════════════════════════════════════════════════════
fam_stats = []
modes = {}
for fam, g in res.groupby("model_family"):
    n = len(g)
    p_sup_miss = 1 - g["supply_ok"].mean()
    p_dem_miss = 1 - g["demand_ok"].mean()

    if fam in DEMAND_ONLY_CANDIDATES:
        if p_sup_miss >= CHECK_THRESHOLD:
            mode = "demand_only"
        else:
            mode = "both22"
            print(f"WARNING: {fam} supply-missing rate is only "
                  f"{p_sup_miss:.1%} (<{CHECK_THRESHOLD:.0%}) -> falling back "
                  f"to the all-22 rule")
    elif fam in SUPPLY_ONLY_CANDIDATES:
        if p_dem_miss >= CHECK_THRESHOLD:
            mode = "supply_only"
        else:
            mode = "both22"
            print(f"WARNING: {fam} demand-missing rate is only "
                  f"{p_dem_miss:.1%} (<{CHECK_THRESHOLD:.0%}) -> falling back "
                  f"to the all-22 rule")
    elif fam in CORE_FAMILIES:
        mode = "both22"
        if max(p_sup_miss, p_dem_miss) > CHECK_THRESHOLD:
            print(f"WARNING: core family {fam} has a missing rate above "
                  f"{CHECK_THRESHOLD:.0%} (supply {p_sup_miss:.1%}, "
                  f"demand {p_dem_miss:.1%}) — check the data!")
    else:
        mode = "excluded"

    modes[fam] = mode
    fam_stats.append({
        "model_family": fam, "n_combos_downloaded": n,
        "pct_supply_incomplete": round(100 * p_sup_miss, 1),
        "pct_demand_incomplete": round(100 * p_dem_miss, 1),
        "mode_applied": mode,
    })

fam_stats = pd.DataFrame(fam_stats).sort_values("model_family")
print("\n===== FAMILY CHECKS (missing rates on downloaded combos) =====")
print(fam_stats.to_string(index=False))

# which variables are missing, per family (downloaded combos, mapped families)
fam_n = res.groupby("model_family").size().to_dict()
miss_cnt = Counter()
for r in records:
    for v in r["miss_supply"] + r["miss_demand"]:
        miss_cnt[(r["model_family"], v)] += 1
miss_by_family = pd.DataFrame(
    [{"model_family": f, "variable": v, "n_combos_missing": n,
      "pct_of_family_combos": round(100 * n / fam_n[f], 1)}
     for (f, v), n in sorted(miss_cnt.items())
     if f != "non-target"])
print("\n===== MISSING VARIABLES BY FAMILY (n combos missing each variable) =====")
print(miss_by_family.to_string(index=False) if len(miss_by_family) else "  none")

# ════════════════════════════════════════════════════════════════════════════
# 4. KEEP COMBOS PER MODE + BUILD OUTPUT ROWS
# ════════════════════════════════════════════════════════════════════════════
FILL_CHILD = Counter()      # (family, variable) -> n series computed
FILL_CHILD_COMBOS = Counter()   # family -> n kept combos with >=1 computed child
FILL_2010 = Counter()       # (family, variable) -> n series 2010<-2020
FILL_2010_COMBOS = Counter()    # family -> n kept combos with >=1 2010 fill
NEG_CLIPS = []              # source values < 0 (AR6 netting artifacts) set to 0

kept_rows, kept_info = [], []
for r in records:
    fam = r["model_family"]
    mode = modes.get(fam, "excluded")
    if mode == "excluded":
        continue
    if mode == "both22" and not (r["supply_ok"] and r["demand_ok"]):
        continue
    if mode == "demand_only" and not r["demand_ok"]:
        continue
    if mode == "supply_only" and not r["supply_ok"]:
        continue

    out_vars = (SUPPLY_11 + DEMAND_11 if mode == "both22"
                else DEMAND_11 if mode == "demand_only" else SUPPLY_11)

    # fill statistics (kept combos only, and only for variables in the output)
    ch = [v for v in r["computed_children"] if v in out_vars]
    f10 = [v for v in r["fills_2010"] if v in out_vars]
    for v in ch:
        FILL_CHILD[(fam, v)] += 1
    for v in f10:
        FILL_2010[(fam, v)] += 1
    if ch:
        FILL_CHILD_COMBOS[fam] += 1
    if f10:
        FILL_2010_COMBOS[fam] += 1

    block = r["block"]
    for v in out_vars:
        s = r["series"][v]
        unit_rows = block[block.variable == v]
        if not unit_rows.empty:
            unit = unit_rows["unit"].iloc[0]
        else:  # computed child: take the parent's unit
            parent = v.rsplit("|", 1)[0]
            p_rows = block[block.variable == parent]
            unit = p_rows["unit"].iloc[0] if not p_rows.empty else "EJ/yr"
        # negative source values (netting artifacts, e.g. Coal w/o CCS =
        # total - CCS share) are clipped to 0 and recorded for the stats sheet
        vals = {}
        for y in YCOLS:
            val = round(float(s[y]), 2)
            if val < 0:
                NEG_CLIPS.append({
                    "model": r["model"], "model_family": fam,
                    "scenario": r["scenario"], "region": r["region"],
                    "variable": v, "year": y, "original_value": val,
                })
                val = 0.0
            vals[y] = val
        kept_rows.append({
            "model": r["model"], "model_family": fam,
            "scenario": r["scenario"], "region": r["region"],
            "variable": v, "unit": unit,
            **vals,
        })
    kept_info.append({"model": r["model"], "scenario": r["scenario"],
                      "region": r["region"], "model_family": fam, "mode": mode,
                      "supply_ok": r["supply_ok"], "demand_ok": r["demand_ok"]})

out = pd.DataFrame(kept_rows)
info = pd.DataFrame(kept_info)
n_final = len(info)

# ════════════════════════════════════════════════════════════════════════════
# 5. FINAL COUNT STATISTICS
# ════════════════════════════════════════════════════════════════════════════
# classification of kept combos: both / demand_only / supply_only
info["kept_as"] = info["mode"].map({"both22": "both_supply_and_demand",
                                    "demand_only": "demand_only",
                                    "supply_only": "supply_only"})
cnt = (info.groupby(["kept_as", "model_family"]).size()
       .reset_index(name="n_combos").sort_values(["kept_as", "model_family"]))
tot = info.groupby("kept_as").size().reset_index(name="n_combos")
tot.insert(1, "model_family", "TOTAL")
final_counts = pd.concat([cnt, tot], ignore_index=True)

# info: demand-only combos that incidentally also had complete supply
extra = (info[(info["mode"] == "demand_only") & info["supply_ok"]]
         .groupby("model_family").size())

print(f"\n===== FINAL KEEP: {n_final} combos "
      f"({info['scenario'].nunique()} unique scenarios) =====")
print(final_counts.to_string(index=False))
if len(extra):
    print("\n(FYI: demand-only combos that would ALSO pass the full supply "
          "rule, not used):")
    print(extra.to_string())

child_df = pd.DataFrame([{"model_family": f, "variable": v, "n_series_computed": n}
                         for (f, v), n in sorted(FILL_CHILD.items())])
child_combo_df = pd.DataFrame([{"model_family": f, "n_combos_with_computed_child": n}
                               for f, n in sorted(FILL_CHILD_COMBOS.items())])
f2010_df = pd.DataFrame([{"model_family": f, "variable": v, "n_series_2010_filled": n}
                         for (f, v), n in sorted(FILL_2010.items())])
f2010_combo_df = pd.DataFrame([{"model_family": f, "n_combos_with_2010_fill": n}
                               for f, n in sorted(FILL_2010_COMBOS.items())])

print("\n===== CHILD COMPLETION (kept combos): combos per family =====")
print(child_combo_df.to_string(index=False) if len(child_combo_df) else "  none")
print("\n===== CHILD COMPLETION (kept combos): series per family x variable =====")
print(child_df.to_string(index=False) if len(child_df) else "  none")

NEG_COLS = ["model", "model_family", "scenario", "region",
            "variable", "year", "original_value"]
neg_df = pd.DataFrame(NEG_CLIPS, columns=NEG_COLS)
neg_summary = (neg_df.groupby(["model_family", "variable"]).size()
               .reset_index(name="n_values_clipped") if len(neg_df) else
               pd.DataFrame([{"model_family": "-", "variable": "-",
                              "n_values_clipped": 0}]))
print(f"\n===== NEGATIVE SOURCE VALUES CLIPPED TO 0 (output data): {len(neg_df)} =====")
print(neg_summary.to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════
# 6. MAIN DATASET
# ════════════════════════════════════════════════════════════════════════════
cat_map = d[["model", "scenario", "category_C"]].drop_duplicates()
out = out.merge(cat_map, on=["model", "scenario"], how="left")
out["scenario_description"] = ""          # left empty on purpose

scen_ids = {s: i + 1 for i, s in enumerate(sorted(out["scenario"].unique()))}
out["scenario_id"] = out["scenario"].map(scen_ids)
ms_ids = {ms: i + 1 for i, ms in enumerate(
    sorted((out["model"] + "||" + out["scenario"]).unique()))}
out["model_scenario_id"] = (out["model"] + "||" + out["scenario"]).map(ms_ids)
out["region_group"] = "R10"

COLS = ["model", "model_family", "scenario", "scenario_description", "scenario_id",
        "model_scenario_id", "region", "region_group", "variable", "unit",
        "category_C"] + YCOLS
out = out[COLS].sort_values(["region", "scenario", "model_family", "variable"])

main_path = OUT_DIR / f"AR6_R10_sd_mixed_{n_final}.xlsx"
out.to_excel(main_path, index=False)
print(f"\nsaved main dataset: {main_path}")
print(f"  rows: {len(out)} | combos: {n_final} | "
      f"scenarios: {out['scenario'].nunique()} | "
      f"families: {sorted(out['model_family'].unique())}")

# ════════════════════════════════════════════════════════════════════════════
# 7. LITERATURE-REFERENCE WORKBOOK (by scenario NAME, across all models)
# ════════════════════════════════════════════════════════════════════════════
# AR6 meta tags the SAME scenario name with different references depending on
# the model, and the completeness filter above keeps only some of those models.
# Collecting literature from the retained (model, scenario) rows would therefore
# drop the references of the rows that were filtered out.
#
# Descriptions are written per scenario NAME, so literature is collected per
# scenario NAME too, from every model carrying that name in the C1-C8 meta.
#
# NOTE: scenario_description is written empty here and is filled by
# scenario_desc_apply.py.
final_scenarios = set(out["scenario"].unique())
ref = lit[lit["scenario"].isin(final_scenarios)].copy()
ref["literature_reference"] = (ref["literature_reference"].astype(str).str.strip()
                               .replace({"nan": "", "None": ""}))
sheet1 = (ref.groupby("scenario")["literature_reference"]
          .apply(lambda s: "; ".join(sorted(set(x for x in s if x))))
          .reset_index())
sheet1["scenario_description"] = ""       # filled in below / by scenario_desc_apply.py
sheet1 = sheet1[["scenario", "scenario_description", "literature_reference"]] \
    .sort_values("scenario").reset_index(drop=True)

refs = sorted(set(r for s in sheet1["literature_reference"] for r in s.split("; ") if r))
sheet2 = pd.DataFrame({"literature_reference": refs})

lit_path = OUT_DIR / "ar6_c1_c8_r10_scenarios_with_literature_reference_mixed.xlsx"

if lit_path.exists():
    prev = pd.read_excel(lit_path, sheet_name="scenario_reference")
    prev_desc = dict(zip(prev["scenario"],
                         prev["scenario_description"].fillna("").astype(str)))
    prev_ref = dict(zip(prev["scenario"],
                        prev["literature_reference"].fillna("").astype(str)))
    sheet1["scenario_description"] = sheet1["scenario"].map(prev_desc).fillna("")
    n_desc = int((sheet1["scenario_description"].str.strip() != "").sum())

    gone = sorted(set(prev_ref) - set(sheet1["scenario"]))
    added = sorted(set(sheet1["scenario"]) - set(prev_ref))
    moved = [sc for sc, r in zip(sheet1["scenario"], sheet1["literature_reference"])
             if sc in prev_ref and prev_ref[sc].strip() != r.strip()]
    print(f"\nliterature workbook vs the copy on disk: "
          f"{len(moved)} scenario(s) changed reference, "
          f"{len(added)} added, {len(gone)} removed; "
          f"{n_desc}/{len(sheet1)} descriptions carried over")
    for sc in moved[:20]:
        print(f"   ~ {sc}")
    for sc in added[:10]:
        print(f"   + {sc}")
    for sc in gone[:10]:
        print(f"   - {sc}")
with pd.ExcelWriter(lit_path, engine="openpyxl") as w:
    sheet1.to_excel(w, sheet_name="scenario_reference", index=False)
    sheet2.to_excel(w, sheet_name="literature", index=False)
print(f"saved literature workbook: {lit_path} "
      f"({len(sheet1)} scenarios, {len(sheet2)} unique references)")

# ════════════════════════════════════════════════════════════════════════════
# 8. STATS WORKBOOK
# ════════════════════════════════════════════════════════════════════════════
funnel = pd.DataFrame([
    ("original combos (C1-C8, any of the 25 variables)", n0),
    ("kept: both supply+demand (22 vars)",
     int((info["kept_as"] == "both_supply_and_demand").sum())),
    ("kept: demand only (11 vars)", int((info["kept_as"] == "demand_only").sum())),
    ("kept: supply only (11 vars)", int((info["kept_as"] == "supply_only").sum())),
    ("kept TOTAL", n_final),
], columns=["step", "n_combos"])

readme = pd.DataFrame([
    ["IMPORTANT: two different scopes are used in this workbook", ""],
    ["", ""],
    ["MISSING statistics", "family_missing_rates, missing_vars_by_family"],
    ["  scope", "ALL downloaded combos (C1-C8, R10), BEFORE any filtering"],
    ["  note", "'missing' is judged AFTER trying both completion rules "
               "(fossil child = total - sibling; 2010 = 2020). A variable "
               "counts as missing only if it could not be rescued."],
    ["  used for", "the family checks (GCAM/TIAM supply-missing >= 90% -> "
                   "demand-only; COFFEE/WITCH demand-missing ~100% -> "
                   "supply-only)"],
    ["", ""],
    ["FILL / COMPLETION statistics", "child_fill_combos, child_fill_series, "
                                     "fill_2010_combos, fill_2010_series"],
    ["  scope", "ONLY combos KEPT in the final dataset, and ONLY variables "
                "actually written to the output (e.g. a GCAM supply child "
                "computed during evaluation is NOT counted, because GCAM is "
                "demand-only and its supply rows are not in the dataset)"],
    ["  answers", "how much of the FINAL dataset is completed/filled data"],
    ["", ""],
    ["consequence", "a combo dropped from the dataset appears in the missing "
                    "statistics but never in the fill statistics"],
    ["", ""],
    ["NEGATIVE-VALUE statistics", "negatives_clipped_summary, "
                                  "negatives_clipped_detail"],
    ["  rule", "any output value < 0 (AR6 source netting artifacts, e.g. "
               "Coal w/o CCS = total - CCS share) is set to 0; the original "
               "values are recorded in the detail sheet"],
    ["  scope", "final output rows only (kept combos, output variables)"],
], columns=["item", "explanation"])

stats_path = OUT_DIR / "build_stats_mixed.xlsx"
with pd.ExcelWriter(stats_path, engine="openpyxl") as w:
    readme.to_excel(w, sheet_name="README", index=False)
    funnel.to_excel(w, sheet_name="funnel", index=False)
    fam_stats.to_excel(w, sheet_name="family_missing_rates", index=False)
    (miss_by_family if len(miss_by_family) else
     pd.DataFrame([{"model_family": "-", "variable": "-",
                    "n_combos_missing": 0, "pct_of_family_combos": 0}])
     ).to_excel(w, sheet_name="missing_vars_by_family", index=False)
    final_counts.to_excel(w, sheet_name="final_counts", index=False)
    (child_combo_df if len(child_combo_df) else
     pd.DataFrame([{"model_family": "-", "n_combos_with_computed_child": 0}])
     ).to_excel(w, sheet_name="child_fill_combos", index=False)
    (child_df if len(child_df) else
     pd.DataFrame([{"model_family": "-", "variable": "-", "n_series_computed": 0}])
     ).to_excel(w, sheet_name="child_fill_series", index=False)
    (f2010_combo_df if len(f2010_combo_df) else
     pd.DataFrame([{"model_family": "-", "n_combos_with_2010_fill": 0}])
     ).to_excel(w, sheet_name="fill_2010_combos", index=False)
    (f2010_df if len(f2010_df) else
     pd.DataFrame([{"model_family": "-", "variable": "-", "n_series_2010_filled": 0}])
     ).to_excel(w, sheet_name="fill_2010_series", index=False)
    neg_summary.to_excel(w, sheet_name="negatives_clipped_summary", index=False)
    (neg_df if len(neg_df) else
     pd.DataFrame([{c: "-" for c in NEG_COLS}])
     ).to_excel(w, sheet_name="negatives_clipped_detail", index=False)
    info.drop(columns="mode").to_excel(w, sheet_name="kept_combos", index=False)
print(f"saved stats workbook: {stats_path}")
