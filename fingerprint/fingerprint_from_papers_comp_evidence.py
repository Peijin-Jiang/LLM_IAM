"""
fingerprint_from_papers_comp_evidence.py
────────────────────────────────────────
Extract one model fingerprint per IAM family from diagnostic and multi-model
comparison studies (PDFs in iam_study/).

For each IAM family and each of its required variables, the LLM judges whether
the studies report the variable as higher, similar or lower in that family
than in other IAMs. A judgement is returned only when the studies compare the
family explicitly with other IAMs, and it is returned together with the
sentence it rests on. A family-variable pair without such a comparison gets no
judgement, and a family without any judgement gets the fingerprint
"Not Available".

Pipeline:
  1. Extract the text of all PDFs (cached).
  2. N_ROUNDS rounds (default 10): one LLM call per round, with all studies in
     a shuffled order and a structured table as output.
  3. Consensus (code only): a judgement is kept once FP_MIN_VOTES rounds
     returned it (default 1), as the mode of the votes. A tie raises an error.
     n_comp_votes records how many rounds returned the judgement.
  4. One text call writes the consensus table as one paragraph per IAM family.

Total LLM calls = N_ROUNDS + 1.

Usage (from this folder):
    python3 fingerprint_from_papers_comp_evidence.py

Env overrides:
    FP_GEN_MODEL             LLM to use (default "gpt-5.6")
    FP_REASONING             reasoning effort (default "none" = thinking off; "" = API default)
    FP_N_ROUNDS              number of shuffled rounds (default 10)
    FP_SEED                  shuffle seed (default 42)
    FP_MIN_VOTES             rounds that must vote for a slot to survive
                             (default 1 = gate off; floor is 1, so 0 and 1 are
                             the same). Raise it to trade recall for confidence.
"""

import json
import collections
import os
import random
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
PAPER_DIR = _HERE / "iam_study"
OUT_DIR = _ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

GEN_MODEL = os.environ.get("FP_GEN_MODEL", "gpt-5.6")
TEMPERATURE = 0
# reasoning effort; "none" = thinking off. Set FP_REASONING="" to omit the parameter.
REASONING = os.environ.get("FP_REASONING", "none").strip().lower()
N_ROUNDS = int(os.environ.get("FP_N_ROUNDS", "10"))
SEED = int(os.environ.get("FP_SEED", "42"))
MIN_VOTES = int(os.environ.get("FP_MIN_VOTES", "1"))

# Families to fingerprint (the 9 present in the AR6 R10 sd MIXED dataset).
# Variable scope differs per family:
#   - 5 families have BOTH sides in the dataset  -> all 22 variables
#   - COFFEE, WITCH have SUPPLY data only        -> the 11 supply variables
#   - GCAM, TIAM have DEMAND data only           -> the 11 demand variables
CORE_FAMILIES = ["AIM/CGE", "GEM-E3", "IMAGE", "MESSAGEix-GLOBIOM", "REMIND"]
SUPPLY_ONLY_FAMILIES = ["COFFEE", "WITCH"]
DEMAND_ONLY_FAMILIES = ["GCAM", "TIAM"]
TARGET_FAMILIES = CORE_FAMILIES + SUPPLY_ONLY_FAMILIES + DEMAND_ONLY_FAMILIES

SUPPLY_VARS = [
    "Primary Energy|Coal|w/ CCS", "Primary Energy|Coal|w/o CCS",
    "Primary Energy|Gas|w/ CCS", "Primary Energy|Gas|w/o CCS",
    "Primary Energy|Oil|w/ CCS", "Primary Energy|Oil|w/o CCS",
    "Primary Energy|Solar", "Primary Energy|Wind", "Primary Energy|Hydro",
    "Primary Energy|Nuclear", "Primary Energy|Biomass",
]
DEMAND_VARS = [   # Final Energy|Electricity excluded from the dataset
    "Final Energy|Liquids", "Final Energy|Gases",
    "Final Energy|Industry", "Final Energy|Transportation",
    "Final Energy|Residential and Commercial", "Final Energy|Hydrogen",
    "Final Energy|Heat", "Final Energy|Solids",
    "Final Energy|Industry|Electricity", "Final Energy|Transportation|Electricity",
    "Final Energy|Residential and Commercial|Electricity",
]
ALL_VARS = SUPPLY_VARS + DEMAND_VARS


def vars_for(fam):
    """Every variable this family reports in the dataset (see the scope comment)."""
    if fam in SUPPLY_ONLY_FAMILIES:
        return SUPPLY_VARS
    if fam in DEMAND_ONLY_FAMILIES:
        return DEMAND_VARS
    return ALL_VARS


PREF_KEY = "Energy_System_Preference_vs_other_models"
QUOTE_KEY, PAPER_KEY = "evidence_quote", "evidence_paper"

COMP_VOCAB = ["higher", "similar", "lower", "no-evidence"]
# the three judgments that count as a vote ("no-evidence" is not one). Ordered
# low -> high only so pick_mode reports a deterministic winner list on a tie.
COMP_ORDER = ["lower", "similar", "higher"]


def load_api_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    for env_file in (_ROOT / ".env", _ROOT.parent / ".env"):
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("OPENAI_API_KEY") and "=" in line:
                    v = line.partition("=")[2].strip().strip('"').strip("'")
                    if v:
                        return v
    raise ValueError("No OpenAI API key found (environment variable or a .env file in the project root).")


def extract_json(text):
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j <= i:
        raise ValueError(f"No JSON object found in model output:\n{s[:500]}")
    return json.loads(re.sub(r",\s*([}\]])", r"\1", s[i:j + 1]))


def chat(client, content, system="You are a careful IAM literature analyst. Return JSON only."):
    """content: str (text-only) or a list of input_text items."""
    if isinstance(content, str):
        content = [{"type": "input_text", "text": content}]
    kwargs = dict(
        model=GEN_MODEL,
        temperature=TEMPERATURE,
        input=[{"role": "system",
                "content": [{"type": "input_text", "text": system}]},
               {"role": "user", "content": content}],
    )
    if REASONING:
        kwargs["reasoning"] = {"effort": REASONING}
    try:
        resp = client.responses.create(**kwargs)
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("temperature", "reasoning", "effort",
                                  "unsupported parameter", "unknown parameter")):
            # parameter rejected -> STOP the whole run so you can adjust
            # FP_GEN_MODEL / FP_REASONING (or TEMPERATURE in the script)
            raise SystemExit(
                f"MODEL PARAMETER REJECTED by the API (model={GEN_MODEL}, "
                f"temperature={TEMPERATURE}, reasoning_effort={REASONING or '(none sent)'}):\n"
                f"  {e}\nAdjust FP_GEN_MODEL / FP_REASONING and rerun.")
        raise
    return resp.output_text


def extract_pdf_text(pdf_path):
    """PyMuPDF only (fixed extractor - no fallback, so every run uses identical text)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise SystemExit("PyMuPDF is required:  pip install pymupdf")
    doc = fitz.open(pdf_path)
    pages = [f"[page {i+1}]\n" + page.get_text("text") for i, page in enumerate(doc)]
    doc.close()
    return "\n".join(pages)


# ============================================================
# ROUND: structured-slots prompt (ONE field per variable)
# ============================================================
ROUND_PROMPT = """You will read the {n_papers} research papers provided below as PLAIN TEXT extracted from PDF files (figures are not included; tables appear as extracted text) involving integrated assessment models (IAMs).

TARGET IAM FAMILIES AND THEIR VARIABLE SCOPE:
{families}

SUPPLY-SIDE (PRIMARY ENERGY) VARIABLES:
{supply_vars}

DEMAND-SIDE (FINAL ENERGY) VARIABLES:
{demand_vars}

REQUIRED OUTPUT CHECKLIST - your JSON must contain EXACTLY these family x variable
entries, no additions and no omissions:
{checklist}

TASK - after reading ALL papers, fill ONE FIXED TABLE per target family describing how
that family COMPARES WITH OTHER IAMs under mitigation, following the REQUIRED OUTPUT
CHECKLIST above EXACTLY: a family's table must contain every variable on its checklist
line and NO variable that is not on it. For each required variable report:

"Energy_System_Preference_vs_other_models": how the family's deployment/reliance/level of that variable
compares with other IAMs in general - one of "higher" / "similar" / "lower" /
"no-evidence".

Whenever this is NOT "no-evidence", the same entry MUST also carry:
  "evidence_quote": the supporting content, taken FROM THE PAPERS. Copy the
     sentence that states or supports the comparison. When the evidence is a table,
     a figure caption, or a column header rather than prose, reproduce the
     relevant part as faithfully as the extracted text allows (e.g. the row
     label and the values being compared).
  "evidence_paper": the filename it came from.

Every word must trace to the provided text. If you cannot point at something in the papers,
the answer is "no-evidence", not a plausible-sounding sentence.

This is a RELATIVE judgment about how the model tends to compare with other IAMs.
Evidence should come from a MULTI-MODEL analysis (e.g., tables, figures, or statements
in papers comparing multiple IAMs).

"similar" is itself a positive claim that needs explicit comparative evidence - NEVER use it
as a default when no comparison is available.

Use explicit comparative evidence whenever possible. Statements that position the target
model relative to multiple other IAMs are the strongest evidence, for example, "this model
uses more biomass than the other models," "this model is among the highest in CCS
deployment," or "this model lies at the lower end of the multi-model range for solar."

Qualitative directional descriptions such as "high," "low," "relatively high," or "relatively
low" MAY also be used when they appear in the context of a multi-model analysis and the paper
clearly presents them as a characteristic of the target model or its scenario set. For example,
a statement that "this model shows relatively high electrification across these scenarios"
may support a "higher" judgment when those scenarios are part of a multi-model comparison.
Likewise, if a paper reports an index calculated from a scenario ensemble and shows that TIAM
has a high value for an index associated with a specific variable, that may also support a
"higher" judgment for that variable, provided the index is clearly interpretable as representing
greater deployment or a higher level of that variable.

Do NOT use purely absolute numerical levels by themselves. For example, "this model uses
80 EJ of biomass" does not establish whether its biomass use is higher or lower than that
of other IAMs.

Do NOT infer a general comparison from a comparison with only ONE other model. For example,
"X uses more CCS than Y" establishes only that X > Y; it does not show that X has higher
CCS deployment relative to IAMs in general.

"Carbon capture" and "CCS" on their own do not name a fuel. A statement that a model relies
more on carbon capture, captures more CO2, or scores high on a carbon-capture index is NOT
evidence for "Primary Energy|Coal|w/ CCS", "Primary Energy|Gas|w/ CCS" or "Primary
Energy|Oil|w/ CCS". Judge those three only where the paper names the fuel being captured
(e.g. "gas-CCS", "coal with CCS"); otherwise "no-evidence".

RULES:
- Ground every judgment ONLY in the provided papers. No outside knowledge, no
  extrapolation.
- EVERY variable in the family's scope MUST appear in its table with the
  "Energy_System_Preference_vs_other_models" field - never omit one; say "no-evidence" instead. Never invent
  evidence. NEVER report variables outside the family's scope.
- Only the exact vocabulary values given above are allowed for the judgment field.
  "evidence_quote" and "evidence_paper" are the one exception: they are free text,
  copied from the papers.
- FAMILY NAMING
  Different version numbers of the same model count as the same family
  (e.g., IMAGE 3.2, WITCH 5.0, REMIND 2.1, COFFEE 1.1,
  MESSAGEix-GLOBIOM 1.1, TIAM-ECN 1.2).
  Use the following name mappings:
  - MESSAGEix-GLOBIOM
    ACCEPT: MESSAGEix-GLOBIOM, MESSAGE-GLOBIOM, MESSAGE
    DO NOT USE: MESSAGE-MACRO
    Use "MESSAGEix" only when the paper refers to MESSAGEix-GLOBIOM.
  - AIM/CGE
    ACCEPT: AIM/CGE, AIM/Hub, AIM-Hub, and AIM when referring to these models.
    DO NOT USE: AIM-Enduse, AIM/Enduse, AIM/Transport.
  - IMAGE
    ACCEPT: IMAGE, IMAGE/TIMER
    DO NOT USE: TIMER when it is clearly used as a standalone model.
  - TIAM
    ACCEPT: TIAM-ECN, TIAM when it refers to TIAM-ECN.
    DO NOT USE: TIAM-UCL, TIAM-Grantham, TIAM-World, TIAM WORLD,
    ETSAP-TIAM.
  - WITCH
    ACCEPT: WITCH and versioned WITCH names.
  - REMIND
    ACCEPT: REMIND and versioned REMIND names.
  - GCAM
    ACCEPT: GCAM and versioned GCAM names.
  - COFFEE
    ACCEPT: COFFEE and versioned COFFEE names.
  - GEM-E3
    ACCEPT: GEM-E3 and versioned GEM-E3 names.

OUTPUT JSON (exactly this structure; variable names must match the lists verbatim):
{{
  "fingerprints": {{
    "<family name>": {{
      "variables": {{
        "<variable name>": {{"Energy_System_Preference_vs_other_models": "higher|similar|lower|no-evidence",
                             "evidence_quote": "<copied from the papers, omit when no-evidence>",
                             "evidence_paper": "<filename, omit when no-evidence>"}}
      }}
    }}
  }}
}}

THE PAPERS (each preceded by a marker with its filename, as plain extracted text):
"""

# ============================================================
# FINAL: render the voted consensus table into prose
# ============================================================
FINAL_PROMPT = """You are given, for each IAM family, a CONSENSUS TABLE of how that family compares
with OTHER IAMs, variable by variable, in mitigation scenarios. Every entry is a purely
RELATIVE judgment - higher, lower, or similar - and carries NO absolute level. Variables
absent from a family's table have no cross-model evidence: never mention them and never
remark that evidence is missing.

THE CONSENSUS TABLES:
{tables}

TASK - for EACH family write "energy_system_preference": fluent technical prose in TWO
parts that expresses EXACTLY the table content - every table entry must be covered, in
the given order, and NOTHING that is not in the table may be added.
- Supply-side part (1-2 sentences) opening with "In primary energy, <family name>
  shows ...". If a family has no supply-side entries, write nothing for that part and
  begin directly with the final-energy part.
- Demand-side part (1-2 sentences) opening with "In final energy, <family name>
  shows ...". If a family has no demand-side entries, write nothing for that part.

WRITING RULES:
- EVERY statement is RELATIVE. Never use an absolute level word (high, moderate, low,
  strong, limited, significant, substantial, negligible) to describe how much of a
  variable the family deploys. Use ONLY these fixed patterns:
  higher  -> "<variable> at a relatively high level compared with other models"
  lower   -> "<variable> at a relatively low level compared with other models"
  similar -> "<variable> at a level similar to other models"
- Group variables that share the SAME comparison into ONE clause, e.g.
  "solar and wind at relatively high levels compared with other models". Pluralize
  "level" to "levels" when a clause covers more than one variable. NEVER merge
  variables whose comparisons differ.
- Never use other connecting phrasings (no "that is", "which is", "and is", "being").
  Never name another IAM model. No absolute numbers.
- If a family is marked "(no evidenced entries)", output an empty string for its
  energy_system_preference. Do not apologise, explain, or hedge - just the
  empty string.
- Neutral, comparative, technical tone (similar to model documentation). State the
  model's behavior directly - never mention tables, evidence, consensus, or the
  synthesis process.

OUTPUT JSON:
{{
  "fingerprints": {{
    "<family name>": {{
      "energy_system_preference": "<the two-part prose>"
    }}
  }}
}}
"""


def _canon_key(s):
    return re.sub(r"\s+", "", str(s)).lower()


VAR_CANON = {_canon_key(v): v for v in ALL_VARS}
FAM_CANON = {_canon_key(f): f for f in TARGET_FAMILIES}
# Aliases for the family KEY a round writes in its JSON - not a rule for reading
# the papers (that is the prompt's FAMILY NAMING block). Without an entry here an
# off-name key makes canonicalize_rounds drop that family's whole table for that
# round, so every ACCEPT name the prompt allows should appear below.
# Names the prompt marks DO NOT USE (AIM-Enduse/-Transport, MESSAGE-MACRO, the
# non-ECN TIAMs) and land-module couplings the prompt does not discuss
# (REMIND-MAgPIE, WITCH-GLOBIOM, COFFEE-TEA) are deliberately absent: a round that
# emits one is reported as unknown rather than silently absorbed.
for _alias, _fam in [
    ("AIM", "AIM/CGE"), ("AIM/Hub", "AIM/CGE"), ("AIM-Hub", "AIM/CGE"),
    ("GEM-E3_V2021", "GEM-E3"),
    ("IMAGE/TIMER", "IMAGE"),
    ("MESSAGE", "MESSAGEix-GLOBIOM"), ("MESSAGEix", "MESSAGEix-GLOBIOM"),
    ("MESSAGE-GLOBIOM", "MESSAGEix-GLOBIOM"),
    ("GCAM-PR", "GCAM"),
    ("TIAM-ECN", "TIAM"),
]:
    FAM_CANON[_canon_key(_alias)] = _fam


def canonicalize_rounds(round_results):
    """Normalize family/variable keys (whitespace/case tolerant) so a cosmetically
    misspelled key becomes a vote instead of silently counting as no-evidence.
    Prints a warning for anything unrecognized or missing."""
    for rr in round_results:
        fps, unknown_fams, unknown_vars, bad_vals = {}, [], 0, 0
        for fam_key, fp in rr["fingerprints"].items():
            fam = FAM_CANON.get(_canon_key(fam_key))
            if fam is None:
                unknown_fams.append(str(fam_key))
                continue
            vars_out = {}
            for vk, slot in (fp.get("variables", {}) or {}).items():
                v = VAR_CANON.get(_canon_key(vk))
                if v is None:
                    unknown_vars += 1
                    continue
                if v not in vars_for(fam):      # outside the family's scope
                    unknown_vars += 1
                    continue
                # a round may return a bare string instead of the slot object.
                # The vote still counts; the evidence sheet then shows an empty
                # evidence cell for that slot.
                if not isinstance(slot, dict):
                    slot = {PREF_KEY: slot}
                cp = str(slot.get("Energy_System_Preference_vs_other_models", "")).strip().lower()
                if cp not in COMP_VOCAB:
                    bad_vals += 1
                vars_out[v] = slot
            fp = dict(fp)
            fp["variables"] = vars_out
            fps[fam] = fp
        missing = [f for f in TARGET_FAMILIES if f not in fps]
        if unknown_fams or unknown_vars or bad_vals or missing:
            print(f"  [round {rr['round']} warning] unknown families: {unknown_fams} | "
                  f"unknown variables: {unknown_vars} | off-vocab values: {bad_vals} | "
                  f"missing families: {missing}")
        rr["fingerprints"] = fps
    return round_results


# ============================================================
# EVIDENCE: flatten the per-round evidence into a long table
# ============================================================
def evidence_rows(round_results):
    """LONG format: one row per judgment that is not 'no-evidence'.

    No verbatim check. Evidence in these papers often lives in a table, a figure
    caption, or a column header, and PDF text extraction mangles all three - a
    substring test against the extracted text would fail on evidence that is
    perfectly real, and the only way for the model to satisfy it would be to
    answer "no-evidence" instead. Coverage is the scarce resource here, so the
    prompt asks for faithful copying and the reader judges the result."""
    rows = []
    for rr in round_results:
        for fam in TARGET_FAMILIES:
            fp = rr["fingerprints"].get(fam, {})
            for var, slot in (fp.get("variables", {}) or {}).items():
                j = str(slot.get(PREF_KEY, "")).strip().lower()
                if j not in COMP_ORDER:            # no-evidence / off-vocab
                    continue
                rows.append({"round": rr["round"], "Model family": fam,
                             "field": var, "judgment": j,
                             "evidence": str(slot.get(QUOTE_KEY, "") or ""),
                             "evidence_paper": str(slot.get(PAPER_KEY, "") or "")})
    return rows


def pick_mode(votes):
    """Most frequent judgment among the votes cast for one slot.

    At temperature 0 the only stochasticity in the design is the order of the
    papers, which changes WHETHER a comparison is found and not the direction
    in which it is read, so the votes for one slot are expected to agree.

    A TIE therefore means that an assumption of this design has broken, so it
    raises an error instead of being resolved silently. build_consensus runs
    after all N_ROUNDS API calls and before anything is written.
    """
    counts = collections.Counter(votes)
    top = max(counts.values())
    winners = [v for v in COMP_ORDER if counts.get(v, 0) == top]
    if len(winners) > 1:
        raise ValueError(
            f"tied vote, no rule to resolve it: {dict(counts)}. The rounds "
            f"disagreed, which has never happened before; decide the tie-break "
            f"deliberately instead of letting the code guess.")
    return winners[0]


def build_consensus(round_results):
    """Pure-code consensus over the structured round tables.
    Returns {family: {"vars": [(variable, comp_str, n_comp_votes)]}}"""
    out = {}
    for fam in TARGET_FAMILIES:
        entries = []
        for var in vars_for(fam):
            cp_votes = []
            for rr in round_results:
                slot = rr["fingerprints"].get(fam, {}).get("variables", {}).get(var, {})
                cp = str(slot.get("Energy_System_Preference_vs_other_models", "no-evidence")).strip().lower()
                if cp in COMP_ORDER:
                    cp_votes.append(cp)
            if len(cp_votes) < max(1, MIN_VOTES):
                continue
            comp = pick_mode(cp_votes)
            entries.append((var, comp, len(cp_votes)))
        out[fam] = {"vars": entries}
    return out


def consensus_tables_text(consensus):
    """Render the voted tables as the FINAL prompt input (fixed order)."""
    blocks = []
    for fam in TARGET_FAMILIES:
        c = consensus[fam]
        sup = [e for e in c["vars"] if e[0] in SUPPLY_VARS]
        dem = [e for e in c["vars"] if e[0] in DEMAND_VARS]
        lines = [f"{fam}:"]
        if not sup and not dem:
            lines.append("  (no evidenced entries)")
        for side, items in (("supply side", sup), ("demand side", dem)):
            if not items:
                continue
            lines.append(f"  {side}:")
            for var, comp, _ in items:
                comp_txt = ("similar to other models" if comp == "similar"
                            else f"{comp} than other models")
                lines.append(f"    - {var}: {comp_txt}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def main():
    from openai import OpenAI
    client = OpenAI(api_key=load_api_key())

    pdfs = sorted(PAPER_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {PAPER_DIR}")
    print(f"model: {GEN_MODEL} | papers: {len(pdfs)} | rounds: {N_ROUNDS} "
          f"| finals: 1 | mode: comparison-only + evidence quotes")

    # ---- extract text from all PDFs once (cached) ----
    txt_cache = _ROOT / "_cache" / "pdf_text"
    txt_cache.mkdir(parents=True, exist_ok=True)
    paper_text = {}
    for pdf in pdfs:
        cache_f = txt_cache / (pdf.stem + ".txt")
        if cache_f.exists() and cache_f.stat().st_mtime >= pdf.stat().st_mtime:
            paper_text[pdf.name] = cache_f.read_text(encoding="utf-8")
        else:
            paper_text[pdf.name] = extract_pdf_text(pdf)
            cache_f.write_text(paper_text[pdf.name], encoding="utf-8")
        print(f"  text {pdf.name[:60]:<62} {len(paper_text[pdf.name]):>8} chars")

    fam_str = (
        "- Evaluated on BOTH sides (all 22 variables): "
        + ", ".join(CORE_FAMILIES) + "\n"
        + "- Evaluated on the SUPPLY side ONLY (the 11 supply variables): "
        + ", ".join(SUPPLY_ONLY_FAMILIES) + "\n"
        + "- Evaluated on the DEMAND side ONLY (the 11 demand variables): "
        + ", ".join(DEMAND_ONLY_FAMILIES)
    )
    # The checklist names the exact entry count per family so a round cannot
    # silently drop or invent one.
    def _side(fam):
        if fam in SUPPLY_ONLY_FAMILIES:
            return "the SUPPLY-SIDE variables ONLY - do NOT report any Final Energy variable"
        if fam in DEMAND_ONLY_FAMILIES:
            return "the DEMAND-SIDE variables ONLY - do NOT report any Primary Energy variable"
        return "SUPPLY-SIDE + DEMAND-SIDE variables"

    check_lines, n_entries = [], 0
    for fam in TARGET_FAMILIES:
        asked = vars_for(fam)
        n_entries += len(asked)
        check_lines.append(f"- {fam}: {len(asked)} entries = {_side(fam)}")
    check_lines.append(f"Total: {len(TARGET_FAMILIES)} family tables, "
                       f"{n_entries} variable entries.")
    checklist = "\n".join(check_lines)
    supply_str = "\n".join(f"- {v}" for v in SUPPLY_VARS)
    demand_str = "\n".join(f"- {v}" for v in DEMAND_VARS)

    # ---- N shuffled rounds, structured output ----
    rng = random.Random(SEED)
    round_results = []
    for r in range(1, N_ROUNDS + 1):
        order = list(paper_text.keys())
        rng.shuffle(order)
        content = [{"type": "input_text", "text": ROUND_PROMPT.format(
            n_papers=len(order), families=fam_str,
            supply_vars=supply_str, demand_vars=demand_str, checklist=checklist)}]
        for pos, name in enumerate(order, 1):
            content.append({"type": "input_text",
                            "text": (f"===== PAPER {pos}/{len(order)}: {name} =====\n\n"
                                     + paper_text[name])})
        print(f"\nRound {r}/{N_ROUNDS} | order: {[o[:25] for o in order]}")
        out = extract_json(chat(client, content))
        round_results.append({"round": r, "order": order,
                              "fingerprints": out["fingerprints"]})
        print(f"  got tables for {len(out['fingerprints'])} families")

    # ---- CONSENSUS: pure-code per-slot voting ----
    round_results = canonicalize_rounds(round_results)
    consensus = build_consensus(round_results)
    n_slots = sum(len(vars_for(f)) for f in TARGET_FAMILIES)
    n_kept = sum(len(consensus[f]["vars"]) for f in TARGET_FAMILIES)
    for fam in TARGET_FAMILIES:
        print(f"  consensus {fam:<18} {len(consensus[fam]['vars']):>2} compared vars")
    print(f"  -> {n_kept} of {n_slots} in-scope slots carry a comparison "
          f"({n_kept / n_slots * 100:.0f}%)")
    silent = [f for f in TARGET_FAMILIES if not consensus[f]["vars"]]
    if silent:
        print(f"  NOTE: no cross-model evidence at all for {silent} - these families "
              f"get an empty preference (written as 'Not Available'). Expected: this "
              f"variant reports only what the papers actually compare.")

    # ---- FINAL: one rendering call ----
    tables_txt = consensus_tables_text(consensus)
    final_prompt = FINAL_PROMPT.format(tables=tables_txt)
    print("\nFinal rendering call ...")
    final = extract_json(chat(client, final_prompt))["fingerprints"]
    esp = {fam: str(final.get(fam, {}).get("energy_system_preference", "")).strip()
           for fam in TARGET_FAMILIES}

    # ---- SAVE ----
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    final_df = pd.DataFrame([{
        "Model family": fam,
        # empty preference (no comparative evidence at all) -> explicit text, so
        # downstream never sees a blank cell (fp_compare_bert rejects empties)
        "Energy System Preference": esp.get(fam, "") or "Not Available",
    } for fam in TARGET_FAMILIES])

    ev_rows = evidence_rows(round_results)
    n_blank = sum(1 for r in ev_rows if not r["evidence"].strip())
    print(f"\nevidence: {len(ev_rows)} evidenced judgments across all rounds"
          + (f"; {n_blank} came back WITHOUT any evidence text" if n_blank else ""))

    rounds_rows = []
    for rr in round_results:
        for fam in TARGET_FAMILIES:
            fp = rr["fingerprints"].get(fam, {})
            rounds_rows.append({
                "round": rr["round"], "Model family": fam,
                "variables_json": json.dumps(fp.get("variables", {}), ensure_ascii=False),
                "paper_order": " | ".join(rr["order"]),
            })

    consensus_rows = [{
        "Model family": fam, "variable": var,
        "Energy_System_Preference_vs_other_models": comp or "", "n_comp_votes": nv,
    } for fam in TARGET_FAMILIES
      for var, comp, nv in consensus[fam]["vars"]]

    fp_dir = OUT_DIR / "fingerprint_from_papers"
    fp_dir.mkdir(exist_ok=True)
    out_xlsx = fp_dir / f"fingerprint_from_papers_comp_evidence_{GEN_MODEL}_{stamp}_struct.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        final_df.to_excel(w, index=False, sheet_name="fingerprint")
        pd.DataFrame(rounds_rows).to_excel(w, index=False, sheet_name="rounds")
        pd.DataFrame(consensus_rows).to_excel(w, index=False, sheet_name="consensus")
        pd.DataFrame(ev_rows, columns=["round", "Model family", "field", "judgment",
                                       "evidence", "evidence_paper"]
                     ).to_excel(w, index=False, sheet_name="evidence")
        pd.DataFrame([
            ("model", GEN_MODEL), ("temperature", TEMPERATURE),
            ("reasoning_effort", REASONING or "(api default)"),
            ("families", "9: 5 both-sides + COFFEE/WITCH supply-only + GCAM/TIAM demand-only"),
            ("n_rounds", len(round_results)), ("shuffle_seed", SEED),
            ("mode", "comparison-only structured slots (no level field) + evidence quotes"),
            ("fields_per_variable", "Energy_System_Preference_vs_other_models only"),
            ("slots_with_comparison", f"{n_kept} of {n_slots}"),
            ("families_without_evidence", ", ".join(silent) or "(none)"),
            ("evidence_rule",
             "every non-'no-evidence' judgment carries supporting text copied from "
             "the papers + the filename; not machine-verified (tables/captions "
             "would fail a verbatim check)"),
            ("evidence_rows", str(len(ev_rows))),
            ("min_votes", str(MIN_VOTES)),
            ("consensus_rule",
             f"Energy_System_Preference_vs_other_models kept if >={max(1, MIN_VOTES)} "
             f"round(s) of {N_ROUNDS}; mode of the votes"),
            ("final_selection", "single rendering call from voted consensus table"),
            ("input_mode", "plain text extracted locally from PDFs (no vision)"),
            ("papers", "; ".join(p.name for p in pdfs)),
            ("run_datetime", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ], columns=["key", "value"]).to_excel(w, index=False, sheet_name="run_config")

    print(f"\nSaved: {out_xlsx}")
    for fam in TARGET_FAMILIES:
        print(f"\n{fam}:\n  {(esp.get(fam, '') or '(no comparative evidence)')[:150]}...")


if __name__ == "__main__":
    main()
