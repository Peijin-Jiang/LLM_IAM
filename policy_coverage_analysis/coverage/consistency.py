#!/usr/bin/env python3
"""Compare two independent classification runs.

The cache makes a repeat run free, and free means identical, so a genuine
second opinion needs a second cache file:

    python3 coverage/classify.py --model M
    python3 coverage/classify.py --model M \\
        --cache  coverage/cache/scenario_labels_run2.json \\
        --output coverage/output/scenario_labels_run2.csv \\
        --workbook coverage/output/scenario_labels_run2.xlsx
    python3 coverage/consistency.py

Per axis this reports the raw agreement rate and Cohen's kappa.  Kappa matters
here: an axis that is true for 85% of scenarios agrees 75% of the time by
chance alone, so a high raw rate on a lopsided axis says very little.  Where
both runs answer true, it also reports how often they cite the same span --
two runs can agree on the verdict while reading different sentences.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xlsxwriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config, io
from coverage.axes import AXES

HERE = Path(__file__).resolve().parent
DEFAULT_A = HERE / "output/scenario_labels.csv"
DEFAULT_B = HERE / "output/scenario_labels_run2.csv"
DEFAULT_OUTPUT = HERE / "report/classification_consistency.xlsx"
KEYS = [axis.key for axis in AXES]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, default=DEFAULT_A)
    parser.add_argument("--b", type=Path, default=DEFAULT_B)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, help="also write the disagreements as csv")
    return parser.parse_args(argv)


def load_run(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    missing = [k for k in KEYS if k not in labels.columns]
    if missing:
        raise KeyError(f"{path.name} is missing axis columns: {missing}")
    for key in KEYS:
        labels[key] = labels[key].astype(bool)
        labels[f"{key}__evidence"] = labels[f"{key}__evidence"].fillna("")
    return labels.set_index("scenario").sort_index()


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    observed = float((a == b).mean())
    expected = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return np.nan if np.isclose(expected, 1.0) else (observed - expected) / (1 - expected)


def axis_summary(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for axis in AXES:
        key = axis.key
        left, right = a[key], b[key]
        both_true = left & right
        same_span = (
            (a.loc[both_true, f"{key}__evidence"] == b.loc[both_true, f"{key}__evidence"])
            .mean() if both_true.any() else np.nan
        )
        rows.append({
            "axis": axis.label,
            "true_in_run_a": int(left.sum()),
            "true_in_run_b": int(right.sum()),
            "agreement": float((left == right).mean()),
            "cohen_kappa": cohen_kappa(left, right),
            "n_disagreements": int((left != right).sum()),
            "only_a_true": int((left & ~right).sum()),
            "only_b_true": int((~left & right).sum()),
            "same_evidence_when_both_true": same_span,
        })
    return pd.DataFrame(rows)


def disagreements(a: pd.DataFrame, b: pd.DataFrame, text: dict[str, str]) -> pd.DataFrame:
    rows = []
    for axis in AXES:
        key = axis.key
        differing = a.index[a[key] != b[key]]
        for scenario in differing:
            rows.append({
                "scenario": scenario,
                "axis": axis.label,
                "run_a": bool(a.at[scenario, key]),
                "run_b": bool(b.at[scenario, key]),
                "evidence_a": a.at[scenario, f"{key}__evidence"],
                "evidence_b": b.at[scenario, f"{key}__evidence"],
                "description": text[scenario],
            })
    return pd.DataFrame(rows, columns=["scenario", "axis", "run_a", "run_b",
                                       "evidence_a", "evidence_b", "description"])


def write_workbook(path: Path, summary: pd.DataFrame, differing: pd.DataFrame,
                   exact: float, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    book = xlsxwriter.Workbook(path)
    title = book.add_format({"bold": True, "font_size": 13})
    note = book.add_format({"font_color": "#5A5A5A", "italic": True, "text_wrap": True})
    header = book.add_format({"bold": True, "bg_color": "#EFEFEF", "border": 1,
                              "text_wrap": True, "valign": "top"})
    cell = book.add_format({"border": 1, "valign": "top"})
    pct = book.add_format({"border": 1, "align": "center", "num_format": "0.0%"})
    num = book.add_format({"border": 1, "align": "center", "num_format": "0.000"})
    wrap = book.add_format({"border": 1, "valign": "top", "text_wrap": True})

    sheet = book.add_worksheet("Summary")
    sheet.write(0, 0, "Agreement between two independent classification runs", title)
    sheet.merge_range(1, 0, 1, 8,
                      f"{n} scenarios. Both runs answer every axis for every scenario "
                      f"identically in {exact:.1%} of cases. Kappa corrects agreement for "
                      f"the rate expected by chance; on a lopsided axis a high raw "
                      f"agreement can still mean an unreliable judgement.", note)
    columns = list(summary.columns)
    sheet.write_row(3, 0, columns, header)
    for offset, row in enumerate(summary.itertuples(index=False)):
        sheet.write(4 + offset, 0, row.axis, cell)
        sheet.write(4 + offset, 1, row.true_in_run_a, cell)
        sheet.write(4 + offset, 2, row.true_in_run_b, cell)
        sheet.write(4 + offset, 3, row.agreement, pct)
        sheet.write(4 + offset, 4, row.cohen_kappa, num)
        sheet.write(4 + offset, 5, row.n_disagreements, cell)
        sheet.write(4 + offset, 6, row.only_a_true, cell)
        sheet.write(4 + offset, 7, row.only_b_true, cell)
        sheet.write(4 + offset, 8, row.same_evidence_when_both_true, pct)
    sheet.set_column(0, 0, 62)
    sheet.set_column(1, 8, 15)
    sheet.set_row(1, 46)

    sheet = book.add_worksheet("Disagreements")
    sheet.write(0, 0, "Every scenario and axis where the two runs differ", title)
    heads = ["Scenario", "Axis", "Run A", "Run B", "Evidence A", "Evidence B", "Description"]
    sheet.write_row(2, 0, heads, header)
    for offset, row in enumerate(differing.itertuples(index=False)):
        sheet.write_row(3 + offset, 0, list(row), wrap)
    sheet.set_column(0, 0, 28)
    sheet.set_column(1, 1, 50)
    sheet.set_column(2, 3, 9)
    sheet.set_column(4, 5, 46)
    sheet.set_column(6, 6, 80)
    sheet.freeze_panes(3, 0)
    sheet.autofilter(2, 0, 2 + len(differing), len(heads) - 1)
    book.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    a, b = load_run(args.a), load_run(args.b)
    if not a.index.equals(b.index):
        raise ValueError("The two runs cover different scenarios")

    text = io.load_descriptions(config.DESCRIPTIONS).set_index("scenario")["description"]
    summary = axis_summary(a, b)
    differing = disagreements(a, b, text.to_dict())
    exact = float((a[KEYS] == b[KEYS]).all(axis=1).mean())

    write_workbook(args.output, summary, differing, exact, len(a))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        differing.to_csv(args.csv, index=False)

    print(f"{len(a)} scenarios | every axis identical in {exact:.1%}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{len(differing)} scenario-axis disagreements")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
