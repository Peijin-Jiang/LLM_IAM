#!/usr/bin/env python3
"""Policy-coverage sparsity, counted in distinct scenarios.

Coverage is reported only for the two IAM-family cross-distributions used in
the analysis:

* dimension x family : scenarios in the dimension that the family ran.
* axis x family      : scenarios on the axis that the family ran.

The three technology axes roll up into one dimension, so a scenario stating
multiple technology axes is counted once in the dimension table and once on
each stated axis in the axis table.

Region is deliberately not analysed: within a model family, region coverage is
close to all-or-nothing, so it says something about model reporting rather than
about scenario design.

Counts do not sum to the scenario total: the classes are multi-label, and a
scenario may be run by several families.

Coverage is split at the median of the counts, separately within each cross
table and over its non-empty cells only.  Cells at or below the median are
``Lower coverage`` and cells above it are ``Higher coverage``.  ``0`` is
reported as its own class, not as an extreme of the same scale: an empty cell
means something stronger than a low one, and it is kept out of the threshold
for the same reason.

Scenarios stating none of the classes are carried as a row of their own,
``None of the four stated``, and counted like any other.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config, io
from coverage.axes import AXES, DIMENSIONS, DIMENSION_OF, OTHER

HERE = Path(__file__).resolve().parent
DEFAULT_LABELS = HERE / "output/scenario_labels.csv"
DEFAULT_QUANTILE = 1 / 2
LOWER_COVERAGE = "Lower coverage"
HIGHER_COVERAGE = "Higher coverage"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--cell-quantile", "--class-x-family-quantile", type=float,
        default=DEFAULT_QUANTILE,
        help="Quantile separating lower- and higher-coverage non-empty cells "
             "(default: 0.5).",
    )
    return parser.parse_args(argv)


def load_membership(path: Path, scenarios: set[str]) -> pd.DataFrame:
    """Return the long ``(scenario, policy_class)`` table, one row per true axis."""
    labels = pd.read_csv(path)
    keys = [axis.key for axis in AXES]
    missing = [key for key in keys if key not in labels.columns]
    if missing:
        raise KeyError(f"{path.name} is missing axis columns: {missing}")
    if set(labels["scenario"]) != scenarios:
        raise ValueError(
            f"{path.name} covers {len(set(labels['scenario']))} scenarios but the "
            f"workbooks have {len(scenarios)}; re-run classify.py"
        )
    # The CSV is keyed; everything downstream is named.  This is the only
    # place the two meet, so a sheet never has to translate anything.
    name_of = {axis.key: axis.label for axis in AXES}
    return (
        labels.melt(id_vars="scenario", value_vars=keys,
                    var_name="policy_class", value_name="present")
        .query("present")
        .drop(columns="present")
        .assign(policy_class=lambda frame: frame.policy_class.map(name_of))
        .reset_index(drop=True)
    )


def add_status(frame: pd.DataFrame, quantile: float) -> pd.DataFrame:
    """Flag rows at or below the within-table quantile; empty stays distinct.

    The quantile is taken over the non-empty rows only.  An empty row is already
    reported as a gap, so letting it also pull the threshold down would count it
    twice -- in the cross-tables a third of the cells are empty.
    """
    counts = frame.loc[frame["n_scenarios"] > 0, "n_scenarios"]
    # Rounded because the counts are integers and the quantile interpolates:
    # a third of 33 gaps is 10.999999999999998, which lands the cut at
    # 11.99999999999999 -- printed as 12, but excluding every row that holds
    # exactly 12.
    threshold = round(float(counts.quantile(quantile)), 6) if len(counts) else 0.0
    frame = frame.copy()
    frame["sparse_threshold"] = threshold
    frame["status"] = np.where(
        frame["n_scenarios"] == 0, "gap",
        np.where(frame["n_scenarios"] <= threshold,
                 LOWER_COVERAGE, HIGHER_COVERAGE),
    )
    return frame


def count_cross_table(membership: pd.DataFrame, classes: list[str],
                      scenario_family: pd.DataFrame,
                      quantile: float) -> pd.DataFrame:
    """Return one complete class-by-family grid with sparse/gap status."""
    runs = membership.merge(scenario_family, on="scenario")
    families = sorted(scenario_family["model_family"].unique())
    grid = pd.MultiIndex.from_product([classes, families],
                                      names=["policy_class", "model_family"])
    cells = (runs.groupby(["policy_class", "model_family"]).scenario.nunique()
             .reindex(grid, fill_value=0).rename("n_scenarios").reset_index())
    return add_status(cells, quantile)


def build_tables(labels: Path = DEFAULT_LABELS,
                 quantile: float = DEFAULT_QUANTILE,
                 ) -> dict[str, pd.DataFrame]:
    """The two cross-distributions, keyed by output-sheet name."""
    descriptions = io.load_descriptions(config.DESCRIPTIONS)
    coverage = io.load_coverage(config.TRAJECTORIES)
    io.join_scenarios(descriptions, coverage)

    membership = load_membership(labels, set(descriptions["scenario"]))
    unlabelled = sorted(set(descriptions["scenario"]) - set(membership["scenario"]))
    if unlabelled:
        membership = pd.concat([membership, pd.DataFrame(
            {"scenario": unlabelled, "policy_class": OTHER})], ignore_index=True)
        print(f"{len(unlabelled)} scenarios stated none of the "
              f"{len(AXES)} axes -> {OTHER!r}")

    # The same membership, rolled up: an axis becomes the dimension it belongs
    # to, and a scenario stating both technology axes is counted once.
    by_dimension = (membership.assign(
        policy_class=membership.policy_class.map(lambda c: DIMENSION_OF.get(c, c)))
        .drop_duplicates().reset_index(drop=True))

    axes = [axis.label for axis in AXES] + [OTHER]
    dimensions = list(DIMENSIONS) + [OTHER]
    scenario_family = coverage[["scenario", "model_family"]].drop_duplicates()
    axis_cross = count_cross_table(membership, axes, scenario_family, quantile)
    dim_cross = count_cross_table(by_dimension, dimensions, scenario_family, quantile)

    return {"Dimension x family": dim_cross, "Axis x family": axis_cross}


def describe(tables: dict[str, pd.DataFrame], quantile: float) -> None:
    """One line per table: size, threshold, and how much it flagged."""
    for name, table in tables.items():
        counts = table["status"].value_counts()
        print(f"{name:>18}: {len(table):>3} cells | q{quantile:.2f} threshold "
              f"{table['sparse_threshold'].iloc[0]:g} | gap {counts.get('gap', 0)} | "
              f"lower coverage {counts.get(LOWER_COVERAGE, 0)} | "
              f"higher coverage {counts.get(HIGHER_COVERAGE, 0)}")


def main(argv: list[str] | None = None) -> int:
    """Print the tables.  Use report.py to get them as a workbook."""
    args = parse_args(argv)
    describe(build_tables(args.labels, args.cell_quantile), args.cell_quantile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
