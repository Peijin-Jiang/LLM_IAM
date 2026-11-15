"""Load and validate the two input workbooks.

The ``descriptions`` sheet is the only source of scenario text; the AR6
workbook is the only source of model and region coverage.  Every scenario in
one must be present in the other -- a mismatch is a bug in the inputs, not
something to work around.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

ZERO_WIDTH = re.compile(r"[​‌‍﻿­]")


def clean(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", ZERO_WIDTH.sub("", text)).strip()


def read_api_key(env_file: Path, name: str) -> str:
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith(f"{name}=") and not line.startswith("#"):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise KeyError(f"{name} is not set in {env_file}")


def load_descriptions(path: Path) -> pd.DataFrame:
    """Return one row per scenario with columns ``scenario`` and ``description``."""
    desc = pd.read_excel(path, sheet_name="descriptions", usecols=["scenario", "description"])
    desc = desc.map(clean)
    if desc["scenario"].duplicated().any():
        raise ValueError(f"Duplicate scenarios in {path.name}")
    if (desc == "").any().any():
        raise ValueError(f"Blank scenario or description in {path.name}")
    return desc.sort_values("scenario").reset_index(drop=True)


def load_coverage(path: Path) -> pd.DataFrame:
    """Return the distinct ``(scenario, model, model_family, region)`` rows."""
    columns = ["scenario", "model", "model_family", "region"]
    data = pd.read_excel(path, usecols=columns)[columns]
    if data.isna().any().any():
        raise ValueError(f"Missing values in {path.name}: {data.isna().sum().to_dict()}")
    data = data.map(clean)
    if (data == "").any().any():
        raise ValueError(f"Blank key values in {path.name}")
    return data.drop_duplicates().sort_values(columns).reset_index(drop=True)


def load_groups(scenarios: set[str], scenref: Path, litfile: Path) -> list[list[str]]:
    """Group scenario names by the set of papers that document them.

    Scenarios whose literature ids are identical form one group, exactly as the
    description generator groups them, so a group is a study and its members
    differ only in the choices that study varied.  A scenario with no mapping
    is its own group rather than being dropped: it still needs labelling, it
    just has no siblings to be compared against.
    """
    ref = pd.read_excel(scenref, sheet_name="scenario_reference")
    scenario_refs: dict[str, set[str]] = {}
    for _, row in ref.iterrows():
        name = clean(row.get("scenario"))
        if not name:
            continue
        for piece in str(row.get("literature_reference", "")).split(";"):
            piece = clean(piece)
            if piece and piece.lower() != "nan":
                scenario_refs.setdefault(name, set()).add(piece)

    sheets = pd.read_excel(litfile, sheet_name=None)
    ids_frame = (sheets["literature_reference"] if "literature_reference" in sheets
                 else sheets[list(sheets)[1]])
    ref_ids: dict[str, set[str]] = {}
    for _, row in ids_frame.iterrows():
        refs = [clean(p) for p in str(row.get("literature_reference", "")).split(";")]
        refs = [r for r in refs if r]
        ids = [i.strip() for i in re.split(r"[;,]", str(row.get("literature_id", ""))) if i.strip()]
        if not refs or not ids:
            continue
        pairs = zip(refs, ids) if len(refs) == len(ids) else ((r, i) for r in refs for i in ids)
        for reference, identifier in pairs:
            ref_ids.setdefault(reference, set()).add(identifier)

    squashed = {re.sub(r"\s+", "", r).lower(): r for r in ref_ids}

    def resolve(reference: str) -> str | None:
        if reference in ref_ids:
            return reference
        return squashed.get(re.sub(r"\s+", "", reference).lower())

    grouped: dict[tuple[str, ...], list[str]] = {}
    for name in sorted(scenarios):
        ids: set[str] = set()
        for reference in scenario_refs.get(name, set()):
            resolved = resolve(reference)
            if resolved:
                ids |= ref_ids[resolved]
        grouped.setdefault(tuple(sorted(ids)) or (f"__ungrouped__{name}",), []).append(name)
    return [members for _, members in sorted(grouped.items())]


def join_scenarios(descriptions: pd.DataFrame, coverage: pd.DataFrame) -> None:
    """Fail loudly when the two workbooks disagree about the scenario set."""
    only_text = sorted(set(descriptions["scenario"]) - set(coverage["scenario"]))
    only_data = sorted(set(coverage["scenario"]) - set(descriptions["scenario"]))
    if only_text or only_data:
        raise ValueError(
            f"Scenario sets differ: {len(only_text)} description-only "
            f"(e.g. {only_text[:3]}), {len(only_data)} data-only (e.g. {only_data[:3]})"
        )
