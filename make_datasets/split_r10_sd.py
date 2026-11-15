"""
split_r10_sd.py
───────────────
Split the MIXED R10 supply+demand dataset (AR6_R10_sd_mixed_*.xlsx) into
train 7/10, val 1/10 and test 2/10 in ONE pass.

Split unit : (model_family, scenario) GROUP - every region of one scenario run
             by one family moves together, so no split ever holds a different
             region of a case another split also holds.  Regions are not a
             stratum: almost every group carries all ten of them, which keeps
             the regional mix even on its own (<=0.1pp apart).
Strata     : model_family || policy-dimension signature, allocated
             proportionally with largest remainders, seed 42.

The signature is the set of design dimensions the scenario states, read from
the coverage labels (S = alternative socioeconomic pathway, M = mitigation
constraint, P = new climate policy, T = new technology assumption; "-" when
the description states none).  Stratifying on the joint controls both margins at
once - a margin is a sum of joint cells - which neither margin alone can do.

The cost of the joint is thin cells.  Roughly a third of the strata hold fewer
than ten groups, and largest-remainder allocation cannot give those a val or a
test share; they land whole in one split.  That is a property of having one to
three groups, not of the allocation, and the run prints which strata it hit.

Not enforced: a scenario run by SEVERAL families is still split across sets,
because the unit is (family, scenario) rather than scenario.  Most test groups
therefore share their description text with a training group under a different
family.  That is what makes RAG neighbours available, and what makes the text
a weak identifier for anything trained on it; the neighbour report counts it.

Category columns (the same in all outputs):
  category_c : fine AR6 label C1..C8
  c_group    : coarse 3-way group derived from category_c (metadata only)
Neither is a prediction target of the trajectory models. The climate category
classifier (classifier/) is trained on these columns.

Outputs (make_datasets/output/split/):
  AR6_R10sd_train70_list.xlsx / AR6_R10sd_train70_ground_truth.xlsx
  AR6_R10sd_val10_list.xlsx   / AR6_R10sd_val10_gt.xlsx
  AR6_R10sd_test_list.xlsx    / AR6_R10sd_test_gt.xlsx
  AR6_R10sd_split_summary.xlsx
  AR6_R10sd_neighbor_report.xlsx

Usage (from the project root):
    python3 make_datasets/split_r10_sd.py
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_COVERAGE = _ROOT / "policy_coverage_analysis"
OUT_DIR = _HERE / "output" / "split"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _archive_existing_split():
    """Move any existing split aside before this run overwrites it.

    The case ids are reassigned on every split, and predictions are keyed on
    them. Keeping every split means that a prediction file can be scored
    against the ground truth it belongs to.

    The archive is a sibling directory named by the timestamp of the newest file
    being replaced, so its name identifies the split, not the day it was moved.
    """
    existing = sorted(OUT_DIR.glob("*.xlsx"))
    if not existing:
        return None
    born = max(f.stat().st_mtime for f in existing)
    stem = OUT_DIR.parent / f"_archive_split_{datetime.fromtimestamp(born):%Y%m%d}"
    dest, n = stem, 1
    while dest.exists() and any(dest.iterdir()):
        n += 1
        dest = stem.with_name(f"{stem.name}_v{n}")
    dest.mkdir(parents=True, exist_ok=True)
    for f in existing:
        shutil.move(str(f), str(dest / f.name))
    print(f"archived previous split ({len(existing)} files) -> {dest}")
    return dest


_archive_existing_split()

SEED = 42
VAL_RATIO, TEST_RATIO = 0.10, 0.20
N_SUB = 5                       # test is also written as N_SUB disjoint folds
BASE = "AR6_R10sd"
YCOLS = [str(y) for y in range(2010, 2101, 10)]

LABELS = _COVERAGE / "coverage" / "output" / "scenario_labels.csv"
# One letter per design dimension, in the order they appear in a signature.
LETTER = {"Alternative socioeconomic pathway": "S", "Mitigation constraint": "M",
          "New climate policy": "P", "New technology assumption": "T"}
NONE_SIGNATURE = "-"

src = max((_HERE / "output").glob("AR6_R10_sd_mixed_*.xlsx"),
          key=lambda p: p.stat().st_mtime)
print("source:", src.name)
df = pd.read_excel(src)
df.columns = [str(c) for c in df.columns]

# safety: the build clips every negative source value to 0, so the mixed
# dataset (and therefore every split written below) must be non-negative
_neg = df[YCOLS].to_numpy(float)
_neg_n = int((_neg < 0).sum())
if _neg_n:
    raise ValueError(f"{_neg_n} negative values found in {src.name} - "
                     "re-run build_r10_sd_mixed.py (it clips negatives to 0 "
                     "and logs them in build_stats_mixed.xlsx)")


def c_group(cat):
    c = str(cat).strip().lower()
    if c in {"c1", "c2", "c3", "c4"}: return "c1-c4"
    if c in {"c5", "c6"}: return "c5-c6"
    if c in {"c7", "c8"}: return "c7-c8"
    return ""


def signatures(scenarios):
    """scenario -> the dimensions its description states, e.g. "MP", "SMT", "-".

    The axis definitions are imported rather than restated: which axis belongs
    to which dimension is decided in coverage/axes.py, and a second copy here
    would silently go stale the next time an axis moves.
    """
    if not LABELS.exists():
        raise SystemExit(
            f"coverage labels not found: {LABELS}\n"
            "Run policy_coverage_analysis/coverage/classify.py first - the split "
            "strata are model_family x policy-dimension signature.")
    sys.path.insert(0, str(_COVERAGE))
    from coverage.axes import AXES

    labels = pd.read_csv(LABELS).set_index("scenario")
    missing = sorted(set(scenarios) - set(labels.index))
    if missing:
        raise SystemExit(
            f"{len(missing)} scenarios of the dataset are not in {LABELS.name}, "
            f"e.g. {missing[:3]} - re-run classify.py against the current "
            "description workbook.")

    by_dimension: dict[str, list[str]] = {}
    for axis in AXES:
        by_dimension.setdefault(axis.dimension, []).append(axis.key)
    ordered = [(LETTER[d], keys) for d, keys in by_dimension.items()]
    ordered.sort(key=lambda pair: "SMPT".index(pair[0]))

    out = {}
    for scenario in scenarios:
        row = labels.loc[scenario]
        out[scenario] = "".join(letter for letter, keys in ordered
                                if bool(row[keys].any())) or NONE_SIGNATURE
    return out


def allocate(counts, target, rng=None):
    """Spread ``target`` items over the strata, proportionally, largest
    remainders first.

    Ties are frequent - half the strata hold one to four groups, so their
    fractional parts are identical - and a stable sort would hand every tie to
    whichever stratum sorts first by name, which means by model family.  The
    seeded rng breaks them instead.
    """
    counts = counts.astype(int)
    if target == 0 or counts.sum() == 0:
        return counts * 0
    raw = counts * (target / counts.sum())
    base = np.minimum(np.floor(raw).astype(int), counts)
    rem = int(target - base.sum())
    jitter = rng.random(len(counts)) if rng is not None else np.zeros(len(counts))
    order = (pd.DataFrame({"fraction": raw - np.floor(raw), "tiebreak": jitter},
                          index=counts.index)
             .sort_values(["fraction", "tiebreak"], ascending=False).index)
    for s in order:
        if rem <= 0: break
        if base[s] < counts[s]: base[s] += 1; rem -= 1
    assert base.sum() == target
    return base


# ── case list ────────────────────────────────────────────────────────────────
cases = (df[["region", "scenario", "model_family"]].drop_duplicates()
         .sort_values(["region", "scenario", "model_family"]).reset_index(drop=True))
cases.insert(0, "id", np.arange(1, len(cases) + 1))

# ── groups: the unit that is actually split ──────────────────────────────────
sig = signatures(sorted(cases.scenario.unique()))
cases["signature"] = cases.scenario.map(sig)
groups = (cases.groupby(["model_family", "scenario"], as_index=False)
          .agg(n_cases=("id", "size"), signature=("signature", "first")))
groups["strata"] = groups.model_family.astype(str) + "||" + groups.signature
print(f"total cases: {len(cases)}  |  groups (family x scenario): {len(groups)}  |  "
      f"regions per group: {groups.n_cases.min()}-{groups.n_cases.max()}")

rng = np.random.default_rng(SEED)
counts = groups.strata.value_counts().sort_index()
n_test = int(round(len(groups) * TEST_RATIO))
n_val = int(round(len(groups) * VAL_RATIO))

# Held out first, validation and test decided second.  Allocating the held-out
# pool first and splitting it afterwards puts the two sets on the same footing:
# a small stratum that gives up one group has that group land in the validation
# set about a third of the time.
held_out = allocate(counts, n_test + n_val, rng)
val_alloc = allocate(held_out, n_val, rng)
test_alloc = held_out - val_alloc

assigned = {}
for strata in counts.index:
    idx = groups.index[groups.strata == strata].to_numpy()
    rng.shuffle(idx)
    t, v = int(test_alloc.get(strata, 0)), int(val_alloc.get(strata, 0))
    assigned |= {i: "test" for i in idx[:t]} | {i: "val" for i in idx[t:t + v]}
groups["split"] = [assigned.get(i, "train") for i in groups.index]

# strata too small to give up a group: reported, because they are the one place
# where the 7:1:2 is not even approximately kept
starved = [s for s in counts.index
           if int(test_alloc.get(s, 0)) == 0 or int(val_alloc.get(s, 0)) == 0]
if starved:
    print(f"{len(starved)}/{len(counts)} strata hold too few groups to fill both a "
          f"val and a test share ({int(counts[starved].sum())} groups, "
          f"{int(groups.loc[groups.strata.isin(starved), 'n_cases'].sum())} cases); "
          f"each lands whole in one split:")
    for s in starved:
        print(f"    {s:<40} {int(counts[s]):>2} group(s)")

cases = cases.merge(groups[["model_family", "scenario", "strata", "split"]],
                    on=["model_family", "scenario"], how="left")
spanning = cases.groupby(["model_family", "scenario"]).split.nunique()
assert (spanning == 1).all(), "a (family, scenario) group was split across sets"
print(cases.split.value_counts().to_string())

# ── GT builders ──────────────────────────────────────────────────────────────
# Category columns are UNIFIED across every split written below:
#   category_c : the fine AR6 label C1..C8  <- authoritative
#   c_group    : the coarse 3-way group derived from it, metadata only
# No model predicts either one; a separate c-category classifier consumes them.
keys = ["region", "scenario", "model_family"]
df = df.rename(columns={"category_C": "category_c"})
df["c_group"] = df["category_c"].apply(c_group)
df = df.merge(cases[keys + ["id", "split"]], on=keys, how="left")

def gt_eval(sub):
    """Evaluation format: id + key cols + category_c + c_group + years."""
    cols = ["id", "region", "model_family", "scenario", "category_c", "c_group",
            "variable", "unit"] + YCOLS
    return sub[cols].sort_values(["id", "variable"])

LIST_COLS = ["id", "region", "scenario", "model_family"]

def save(name, list_df, gt_df):
    list_df[LIST_COLS].to_excel(OUT_DIR / f"{BASE}_{name}_list.xlsx", index=False)
    gt_df.to_excel(OUT_DIR / f"{BASE}_{name}_gt.xlsx", index=False)

# train70: full-format ground truth (all original columns), with the case `id`
# and both category columns.
tr_cases = cases[cases.split == "train"]
tr_gt = df[df.split == "train"].drop(columns=["split"])
# column order: id | original mixed-dataset columns (category_c in place) |
#               c_group right after category_c | year columns
_head = [c for c in tr_gt.columns if c not in ("id", "c_group") + tuple(YCOLS)]
_head.insert(_head.index("category_c") + 1, "c_group")
tr_gt = tr_gt[["id"] + _head + YCOLS]
tr_cases[LIST_COLS].to_excel(OUT_DIR / f"{BASE}_train70_list.xlsx", index=False)
tr_gt.to_excel(OUT_DIR / f"{BASE}_train70_ground_truth.xlsx", index=False)

# val10 + test: eval-format gt
va_cases = cases[cases.split == "val"]
save("val10", va_cases, gt_eval(df[df.split == "val"]))
te_cases = cases[cases.split == "test"]
save("test", te_cases, gt_eval(df[df.split == "test"]))

# ── test folds ───────────────────────────────────────────────────────────────
# N_SUB disjoint subsets whose union is the whole test set.  Groups stay intact
# here too: a fold that held three regions of a case while another fold held the
# other seven would make the folds correlated, and comparing them - which is
# what folds are for - would understate the spread.  Groups are dealt round
# robin within each stratum, largest first, so the folds match on family and on
# signature and end up close in size despite groups holding 3 to 10 regions.
te_groups = (groups[groups.split == "test"]
             .sort_values(["strata", "n_cases", "scenario"],
                          ascending=[True, False, True]).reset_index(drop=True))
fold_cases = np.zeros(N_SUB, dtype=int)
fold_of = {}
for strata, block in te_groups.groupby("strata", sort=True):
    for row in block.itertuples(index=False):
        # the emptiest fold takes the next group; ties go to the lowest index
        pick = int(np.argmin(fold_cases))
        fold_of[(row.model_family, row.scenario)] = pick + 1
        fold_cases[pick] += row.n_cases
te_cases = te_cases.assign(sub=[fold_of[(r.model_family, r.scenario)]
                                for r in te_cases.itertuples(index=False)])
te_gt = gt_eval(df[df.split == "test"])
sub_of_id = te_cases.set_index("id")["sub"]
for k in range(1, N_SUB + 1):
    part = te_cases[te_cases["sub"] == k]
    save(f"sub{k:02d}_test", part, te_gt[te_gt.id.map(sub_of_id) == k])
sizes = te_cases["sub"].value_counts().sort_index()
print(f"test folds (cases): {sizes.to_dict()}")
assert sizes.sum() == len(te_cases) and len(sizes) == N_SUB

# ── summary + neighbor report ───────────────────────────────────────────────
rows = []
for name, cs in [("train70", tr_cases), ("val10", va_cases), ("test", te_cases)]:
    rows.append({"split": name, "summary_type": "TOTAL", "group": "TOTAL",
                 "n_cases": len(cs), "n_groups": len(cs.groupby(
                     ["model_family", "scenario"]))})
    for col in ["model_family", "region", "signature"]:
        for value, sub in cs.groupby(col):
            rows.append({"split": name, "summary_type": col, "group": str(value),
                         "n_cases": len(sub),
                         "n_groups": len(sub.groupby(["model_family", "scenario"]))})
    for dimension, letter in LETTER.items():
        sub = cs[cs.signature.str.contains(letter)]
        rows.append({"split": name, "summary_type": "dimension", "group": dimension,
                     "n_cases": len(sub),
                     "n_groups": len(sub.groupby(["model_family", "scenario"]))})
summary = pd.DataFrame(rows)
summary["share_pct"] = (summary.n_cases / summary.groupby(
    ["summary_type", "group"]).n_cases.transform("sum") * 100).round(1)
summary.to_excel(OUT_DIR / f"{BASE}_split_summary.xlsx", index=False)

# what the stratification was for: every row of these two should read 70/10/20
for kind in ("model_family", "dimension"):
    table = (summary[summary.summary_type == kind]
             .pivot(index="group", columns="split", values="share_pct")
             .reindex(columns=["train70", "val10", "test"]))
    print(f"\n{kind} share of its own cases (%):")
    print(table.to_string())

rep = []
tset = tr_cases[keys]
for row in te_cases.itertuples(index=False):
    same_scen = tset[(tset.region == row.region) & (tset.scenario == row.scenario)
                     & (tset.model_family != row.model_family)]
    same_fam = tset[(tset.region == row.region) & (tset.model_family == row.model_family)
                    & (tset.scenario != row.scenario)]
    rep.append({"id": row.id, "region": row.region, "scenario": row.scenario,
                "model_family": row.model_family,
                "n_same_scenario_neighbors": len(same_scen),
                "n_same_family_neighbors": len(same_fam)})
rep = pd.DataFrame(rep)
rep.to_excel(OUT_DIR / f"{BASE}_neighbor_report.xlsx", index=False)
print(f"\ntest cases without same-scenario neighbor: "
      f"{(rep.n_same_scenario_neighbors == 0).sum()}/{len(rep)}")
print(f"test cases without same-family neighbor: "
      f"{(rep.n_same_family_neighbors == 0).sum()}/{len(rep)}")
shared = te_cases.scenario.isin(set(tr_cases.scenario))
print(f"test cases whose scenario text also appears in train under another "
      f"family: {int(shared.sum())}/{len(te_cases)} "
      f"({shared.mean() * 100:.0f}%)")
print("saved ->", OUT_DIR)
