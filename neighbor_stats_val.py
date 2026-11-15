"""
neighbor_stats_val.py
─────────────────────
How many retrieval neighbours does each VAL case
actually have available in train70?

Counted the way rag_core retrieval scopes them - within the SAME REGION:
  scen_nbrs  = train cases, same region + same scenario, DIFFERENT model family
               (the TOPK_SCEN pool; report mean and share <= 3)
  model_nbrs = train cases, same region + same model family, DIFFERENT scenario
               (the TOPK_MODEL pool; report mean and share <= 15)

Usage (from the project root):  python3 neighbor_stats_val.py
Output: outputs/neighbor_stats_val.xlsx  (summary + per_case)
"""

from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
SPLIT = _HERE / "make_datasets" / "output" / "split"
OUT = _HERE / "outputs" / "neighbor_stats_val.xlsx"


def cases(path):
    df = pd.read_excel(path)
    df.columns = [str(c) for c in df.columns]
    return (df[["id", "region", "model_family", "scenario"]]
            .drop_duplicates("id").astype(str))


def main():
    tr = cases(SPLIT / "AR6_R10sd_train70_ground_truth.xlsx")
    va = cases(SPLIT / "AR6_R10sd_val10_gt.xlsx")
    print(f"train {len(tr)} cases | val {len(va)} cases")

    scen_pool = tr.groupby(["region", "scenario"])["model_family"].apply(set)
    fam_pool = tr.groupby(["region", "model_family"])["scenario"].apply(set)

    rows = []
    for _, c in va.iterrows():
        fams = scen_pool.get((c.region, c.scenario), set())
        scens = fam_pool.get((c.region, c.model_family), set())
        rows.append({
            "id": c.id, "region": c.region, "model_family": c.model_family,
            "scenario": c.scenario,
            "scen_nbrs": len(fams - {c.model_family}),
            "model_nbrs": len(scens - {c.scenario}),
        })
    pc = pd.DataFrame(rows)

    summary = pd.DataFrame([
        {"pool": "same scenario, other model (TOPK_SCEN)",
         "mean": round(pc.scen_nbrs.mean(), 2),
         "median": pc.scen_nbrs.median(),
         "min": pc.scen_nbrs.min(), "max": pc.scen_nbrs.max(),
         "share_le_3": round((pc.scen_nbrs <= 3).mean(), 4),
         "share_eq_0": round((pc.scen_nbrs == 0).mean(), 4)},
        {"pool": "same model, other scenario (TOPK_MODEL)",
         "mean": round(pc.model_nbrs.mean(), 2),
         "median": pc.model_nbrs.median(),
         "min": pc.model_nbrs.min(), "max": pc.model_nbrs.max(),
         "share_le_15": round((pc.model_nbrs <= 15).mean(), 4),
         "share_eq_0": round((pc.model_nbrs == 0).mean(), 4)},
    ])
    print(summary.to_string(index=False))

    with pd.ExcelWriter(OUT) as xw:
        summary.to_excel(xw, sheet_name="summary", index=False)
        pc.to_excel(xw, sheet_name="per_case", index=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
