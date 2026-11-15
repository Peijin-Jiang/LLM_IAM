#!/usr/bin/env python3
"""Compute the coverage tables and render them into one workbook.

This is the only entry point that writes the TABLES: sparse.py builds them in
memory and prints a summary but saves nothing.  The workbook contains the two
cross-distributions used by the analysis: dimension x IAM family and axis x IAM
family.  (classify.py writes the labels this reads, and consistency.py its own
workbook; neither goes through here.)

The tables arrive carrying the names they are shown under -- sparse.py resolves
the CSV's keys once, when it loads them -- so nothing here translates anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import xlsxwriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coverage.sparse import (DEFAULT_LABELS, DEFAULT_QUANTILE, HIGHER_COVERAGE,
                             LOWER_COVERAGE, build_tables, describe)

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "report/policy_coverage.xlsx"

NOTE = ("Every count is a number of distinct scenarios. Counts do not sum to the "
        "scenario total: the classes are multi-label and a scenario may be run by "
        "several model families. The two technology axes share one dimension, so a "
        "scenario stating both is counted twice among the axes and once among the "
        "dimensions. Region is not analysed; within a family, region coverage is "
        "close to all-or-nothing and reflects model reporting rather than scenario "
        "design.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cell-quantile", "--class-x-family-quantile", type=float,
        default=DEFAULT_QUANTILE,
        help="Quantile separating lower- and higher-coverage non-empty cells "
             "(default: 0.5).",
    )
    return parser.parse_args(argv)


def styles_for(book) -> dict:
    return {
        "title": book.add_format({"bold": True, "font_size": 13}),
        "note": book.add_format({"font_color": "#5A5A5A", "italic": True, "text_wrap": True}),
        "header": book.add_format({"bold": True, "bg_color": "#EFEFEF", "border": 1,
                                   "text_wrap": True, "valign": "top"}),
        "name": book.add_format({"bold": True, "text_wrap": True, "valign": "top",
                                 "border": 1}),
        "cell": book.add_format({"border": 1, "align": "center", "valign": "top"}),
        "text": book.add_format({"border": 1, "valign": "top"}),
        "gap": book.add_format({"bg_color": "#F4C7C3", "bold": True, "align": "center"}),
        LOWER_COVERAGE: book.add_format({"bg_color": "#FCE8B2", "align": "center"}),
    }


def write_matrix(book, st, sheet_name, title, cells: pd.DataFrame) -> None:
    matrix = cells.pivot(index="policy_class", columns="model_family", values="n_scenarios")
    status = cells.pivot(index="policy_class", columns="model_family", values="status")
    threshold = float(cells["sparse_threshold"].iloc[0])
    sheet = book.add_worksheet(sheet_name)
    sheet.write(0, 0, title, st["title"])
    sheet.merge_range(1, 0, 1, len(matrix.columns),
                      f"Scenarios of the class that the family ran. Lower coverage = at "
                      f"or below the median of this table's non-empty cells, at "
                      f"{threshold:g}; 0 is a gap and is left out of the threshold.",
                      st["note"])
    sheet.write_row(3, 1, list(matrix.columns), st["header"])
    for offset, (key, values) in enumerate(matrix.iterrows()):
        sheet.write(4 + offset, 0, key, st["name"])
        for column, value in enumerate(values.tolist()):
            sheet.write(4 + offset, 1 + column, value,
                        st.get(status.loc[key].iloc[column], st["cell"]))
    last_col = len(matrix.columns)
    sheet.set_column(0, 0, 62)
    sheet.set_column(1, last_col, 15)
    sheet.freeze_panes(4, 1)


def write_summary(book, st, rows) -> None:
    sheet = book.add_worksheet("Summary")
    sheet.write(0, 0, "Policy coverage across model families", st["title"])
    sheet.merge_range(1, 0, 1, 1, NOTE, st["note"])
    sheet.write_row(4, 0, ["Sheet", "Result"], st["header"])
    for offset, (name, table) in enumerate(rows):
        counts = table["status"].value_counts()
        sheet.write(5 + offset, 0, name, st["name"])
        sheet.write(5 + offset, 1,
                    f"{len(table)} rows, lower-coverage threshold "
                    f"{table['sparse_threshold'].iloc[0]:g}, "
                    f"{counts.get('gap', 0)} gaps, "
                    f"{counts.get(LOWER_COVERAGE, 0)} lower-coverage cells, "
                    f"{counts.get(HIGHER_COVERAGE, 0)} higher-coverage cells",
                    st["text"])
    sheet.set_column(0, 0, 26)
    sheet.set_column(1, 1, 70)
    sheet.set_row(1, 62)


def write_gaps(book, st, rows) -> None:
    flagged: list[dict] = []
    for name, table in rows:
        for row in table[table["status"] != HIGHER_COVERAGE].itertuples(index=False):
            key = getattr(row, "policy_class", None)
            family = getattr(row, "model_family", None)
            flagged.append({
                "sheet": name,
                "policy_class": key or "",
                "model_family": family or "",
                "n_scenarios": row.n_scenarios,
                "status": row.status,
            })
    sheet = book.add_worksheet("Gaps")
    sheet.write(0, 0, "Every gap and lower-coverage cell", st["title"])
    sheet.merge_range(1, 0, 1, 4,
                      "gap = no scenario at all; Lower coverage = at or below the "
                      "median of the table's non-empty cells.", st["note"])
    headers = ["Sheet", "Class", "Model family", "Scenarios", "Status"]
    sheet.write_row(3, 0, headers, st["header"])
    for offset, row in enumerate(flagged):
        sheet.write_row(4 + offset, 0, list(row.values()), st["text"])
    sheet.set_column(0, 0, 22)
    sheet.set_column(1, 1, 62)
    sheet.set_column(2, 4, 20)
    sheet.freeze_panes(4, 0)
    sheet.autofilter(3, 0, 3 + len(flagged), len(headers) - 1)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tables = build_tables(args.labels, args.cell_quantile)
    describe(tables, args.cell_quantile)
    dim_cross, axis_cross = tables.values()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    book = xlsxwriter.Workbook(args.output)
    st = styles_for(book)

    rows = list(tables.items())
    write_summary(book, st, rows)

    write_matrix(book, st, "Dimension x family", "Dimension x model family", dim_cross)
    write_matrix(book, st, "Axis x family", "Axis x model family", axis_cross)
    write_gaps(book, st, rows)

    book.close()
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
