"""Shared paths and constants for the policy-coverage and similarity analyses."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # project root

DESCRIPTIONS = (
    ROOT
    / "outputs/scenario_desc"
    / "scenario_desc_gen_gpt-5.6_20260826_0429.xlsx"
)
TRAJECTORIES = ROOT / "make_datasets/output/AR6_R10_sd_mixed_6761.xlsx"

# Scenario -> literature grouping.  Scenarios documented by the same paper(s)
# are labelled together, so the model can see which settings every member of a
# study shares (its baseline) and which distinguish one member from another
# (its treatment).  Three of the five axes ask for a departure from the default,
# and a departure is only visible against its siblings.
SCENARIO_REFERENCES = (
    ROOT / "make_datasets/output"
    / "ar6_c1_c8_r10_scenarios_with_literature_reference_mixed.xlsx"
)
LITERATURE_IDS = ROOT / "scenario_des/literature_reference.xlsx"

# Scenarios per request.  One group is 182 scenarios, so this is a ceiling
# rather than a target: most groups go in a single call.
BATCH = 200
ENV_FILE = ROOT / ".env"

YEARS = [str(year) for year in range(2010, 2101, 10)]
