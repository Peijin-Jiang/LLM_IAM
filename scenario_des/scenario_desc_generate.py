"""
scenario_desc_generate.py
─────────────────────────
Generate one description per scenario from the source documents that the AR6
metadata lists for it.

The scenarios are read from make_datasets/output/AR6_R10_sd_mixed_*.xlsx (the
newest file; override with SD_TARGET_XLSX). Scenarios with the same source
documents form a group and are processed together. Every document of a group
is read in full. The descriptions cover scenario design and do not name an IAM.

════════════════════════════════════════════════════════════════════════════
PIPELINE
════════════════════════════════════════════════════════════════════════════
  A1. profile pass        1 call per group -> the TOKENIZATION of every name
                          in it, plus a TOKEN GLOSSARY for the tokens the
                          papers define. Splitting and defining both happen
                          here and nowhere else
  A2. glossary retries    up to SD_SPEC_ROUNDS more calls per group, each
                          aimed at the tokens still open - the tokenization
                          minus the glossary minus the model names. Same
                          system prompt, same output shape, same evidence
                          rule as A1: it IS A1, asked a shorter question.
                          Stops as soon as a round adds nothing
  B.  generation pass     1 call per group -> per scenario: a 1-5 sentence
                          description, written ONCE, from a glossary nothing
                          will add to. It resolves no tokens of its own; a
                          token the glossary misses is carried verbatim in a
                          fixed fallback form. It still reads the literature -
                          its Step 2 has no other source

Mapping chain:
  scenario -> literature_reference : ar6_c1_c8_r10_scenarios_with_literature_reference.xlsx
                                     (sheet "scenario_reference", ";"-separated)
  literature_reference -> id       : literature_reference.xlsx (2nd sheet;
                                     multi-ref cells paired positionally)
  id -> PDF                        : literature/<id>.pdf

Output -> outputs/scenario_desc/scenario_desc_gen_<model>_<stamp>.xlsx
Four sheets, in the order you read them:
  "tokens"       every token settled by the PROFILE PASS or by a SPEC ROUND,
                 which scenarios it occurs in, and which of the two settled it.
                 Those two passes are the ONLY places a meaning is worked out,
                 so between this sheet and "unresolved" the workbook accounts
                 for every token: settled with a meaning here, or still open
                 there. The generation pass in between neither derives meanings
                 of its own nor decodes them from the shape of a name - it uses
                 this glossary or falls back and declares - so there is no
                 third, invisible category of "resolved somehow, recorded
                 nowhere".
                 The one way a token can still go missing from both is a
                 generation pass that leaves it verbatim and forgets to declare
                 it. Nothing detects that; the declaration is the only record.
                 A token appearing in neither sheet is worth a look at the
                 description text for exactly that reason.
  "unresolved"   tokens of a name whose meaning that name's description does
                 not carry, after the spec rounds. Derived, not declared
  "other_info"   the Step 2 half of each description on its own, verbatim as
                 it appears in the final text
  "descriptions" scenario | description - the final text, and the only sheet
                 scenario_desc_apply.py reads

Usage (from this folder):
    python3 scenario_desc_generate.py

Env overrides:
    SD_MODEL           LLM (default "gpt-5.6")
    SD_REASONING       default reasoning effort for every stage
                       (default "none" = thinking off; "" = omit the parameter)
    SD_REASONING_PROFILE  reasoning effort for the profile pass only
    SD_REASONING_SPEC     reasoning effort for the spec-resolution pass only
                       Both default to SD_REASONING.
    SD_TARGET_XLSX     dataset providing the scenario list
    SD_SCENREF_XLSX    scenario -> literature_reference mapping
    SD_LIT_FILE        literature_reference.xlsx
    SD_PDF_DIR         folder with <literature_id>.pdf
    SD_SCEN_BATCH      scenarios per LLM call within a group (default 200, i.e.
                       one call per group for all groups but the largest).
                       Within one call a name component shared by several
                       scenarios is worded in the same way, so one call per
                       group is preferred.
    SD_RETRY_ROUNDS    extra rounds re-requesting scenarios the model did not
                       return (default 2)
    SD_PROFILE_PASS    1 (default) = run the profile pass (A1). 0 SKIPS BOTH
                       A1 AND A2 and leaves the generation pass with no
                       tokenization and no glossary, which is not a useful
                       run: every token comes out as a fallback. It is kept
                       only as a way to isolate A1's cost
    SD_SPEC_PASS       1 (default) = run the glossary retries (A2); 0 = stop
                       after A1 and write descriptions from whatever it found
    SD_SPEC_ROUNDS     max retry rounds (default 3; stops early as soon as a
                       round adds no token meaning). Each round asks only
                       about tokens still open, so later rounds are cheap;
                       what they cost is one more full-text call per group
    SD_LIMIT           only process the first N groups (smoke test; 0 = all)
"""

import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

MODEL = os.environ.get("SD_MODEL", "gpt-5.6")
TEMPERATURE = 0
# reasoning effort for the Responses API; "none" = thinking off.
# set SD_REASONING="" to omit the parameter (use the API default).
REASONING = os.environ.get("SD_REASONING", "none").strip().lower()
REASONING_PROFILE = os.environ.get("SD_REASONING_PROFILE", REASONING).strip().lower()
REASONING_SPEC = os.environ.get("SD_REASONING_SPEC", REASONING).strip().lower()
def _default_target():
    cands = sorted((_ROOT / "make_datasets" / "output").glob("AR6_R10_sd_mixed_*.xlsx"),
                   key=lambda q: q.stat().st_mtime)
    if not cands:
        raise SystemExit("no AR6_R10_sd_mixed_*.xlsx found under "
                         f"{_ROOT / 'make_datasets' / 'output'} - run "
                         "build_r10_sd_mixed.py first or set SD_TARGET_XLSX")
    return cands[-1]

TARGET_XLSX = Path(os.environ.get("SD_TARGET_XLSX", "") or _default_target())
SCENREF_XLSX = Path(os.environ.get(
    "SD_SCENREF_XLSX", _ROOT / "make_datasets" / "output" /
    "ar6_c1_c8_r10_scenarios_with_literature_reference_mixed.xlsx"))
LIT_FILE = Path(os.environ.get("SD_LIT_FILE", _HERE / "literature_reference.xlsx"))
PDF_DIR = Path(os.environ.get("SD_PDF_DIR", _HERE / "literature"))
SCEN_BATCH = int(os.environ.get("SD_SCEN_BATCH", "200"))
RETRY_ROUNDS = int(os.environ.get("SD_RETRY_ROUNDS", "2"))


def _flag(name, default="1"):
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no")


SPEC_PASS = _flag("SD_SPEC_PASS")
# Ask the description call to return verbatim support for "other_info" too.
OI_EVIDENCE = _flag("SD_OI_EVIDENCE")
PROFILE_PASS = _flag("SD_PROFILE_PASS")
SPEC_ROUNDS = int(os.environ.get("SD_SPEC_ROUNDS", "3"))
LIMIT = int(os.environ.get("SD_LIMIT", "0"))

OUT_DIR = _ROOT / "outputs" / "scenario_desc"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def norm(x):
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def match_key(x):
    """Whitespace-insensitive, case-insensitive key for scenario-name matching.
    Handles names that contain stray spaces (e.g. 'DeepElec_SSP2_ HighRE_...')."""
    return re.sub(r"\s+", "", str(x or "")).lower()


def canon_token_from(tokens, token):
    """The token as the PROFILE PASS spelled it, or None if it is not one of
    this scenario's tokens.

    The model hands tokens back in whatever casing it used, and every later
    step matches a token case-sensitively - so "cov" declared against a
    tokenization containing "COV" would match nothing, silently drop out of
    the spec pass, and never show up as unresolved either. Normalising here is
    what keeps that from happening. A string that is not in the list at all is
    rejected: it is a token the model invented."""
    t = str(token or "").strip()
    if not t:
        return None
    for nt in tokens:
        if nt.lower() == t.lower():
            return nt
    return None


# NO TOKENIZER HERE, DELIBERATELY.
# Splitting a scenario name is a semantic decision - "-" separates in
# BEG-Budg600-EG and joins in CD-LINKS - so the PROFILE PASS makes it ONCE,
# with the surrounding names and the papers in front of it, and reports it.
# Everything downstream reads that answer through tokens_of().


def extract_json(text):
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j <= i:
        raise ValueError(f"No JSON object found in model output:\n{s[:500]}")
    return json.loads(re.sub(r",\s*([}\]])", r"\1", s[i:j + 1]))


# ============================================================
# SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = """
You are an expert in IAM scenario interpretation.

TASK
Describe each scenario named in the request, using ONLY the supplied
literature text. Work through Steps 1-2 below; report the result in the shape
given under OUTPUT.

WHAT YOU ARE GIVEN
Each scenario in the request comes with its TOKENS, already split, and the
request carries a TOKEN GLOSSARY giving what those tokens mean. Both were
produced once for the whole family by an earlier step that had the papers in
front of it, and both are authoritative here.
- Use the split exactly as given. Do NOT re-split a name, and do not treat any
  other substring of it as a token: a token is what the list says it is,
  spelled the way the list spells it, and the list is complete.
- The LEADING FAMILY TOKEN is the first token in that list.
- The token list is working material, not output. It never appears in the
  description; individual tokens appear only where the rules below put them.

## Step 1: The scenario from its tokens
The result of this step is the "scenario_from_tokens" field.

Give each token the meaning assigned by the TOKEN GLOSSARY. Where the glossary
defines a token, its entry is authoritative: do not re-derive or second-guess
it. Use its SUBSTANCE exactly as given, and its WORDING too, except where
WRITING CONVENTIONS prescribe the name of the same term. Each entry applies
ONLY to a scenario whose name actually contains that token.

For any token the glossary does not cover, do not infer a meaning - not from
the literature, and not from the shape of the scenario name. Follow WHEN A
TOKEN CANNOT BE RESOLVED below.

GLOSSARY WORDING vs WRITING CONVENTIONS. Where WRITING CONVENTIONS prescribe
wording for a standard term (SSP narratives, temperature targets, radiative
forcing, carbon budgets, policy narratives), that wording takes precedence for
NAMING the same term only. Preserve any additional study-specific substance
the glossary supplies. For example, write

    Middle of the Road (SSP2), in which non-CO₂ greenhouse-gas emissions from
    land use are hardly reduced

rather than dropping the study-specific half or replacing the prescribed
standard name with the glossary's own phrasing.

WHEN A TOKEN CANNOT BE RESOLVED
The request marks these for you: they are the tokens listed after "No meaning
established:". You do not report them back - which tokens are open is already
known - you only have to write them properly.
- Keep the token verbatim and carry it in one of these two fixed forms:
      inner token   -> "with a TOKEN specification"
      family token  -> "a TOKEN scenario"
  Fit the form into the sentence so the grammar stays natural.
- NEVER comment in the description on a token being unknown. "not further
  defined", "not defined", "not specified", "meaning is unclear", "unknown"
  and any similar remark are forbidden anywhere in the text you write.
- EXCEPTION: a token the request marks as a model name or version is left
  verbatim and is never remarked on either, but it is not an open token and
  gets no fallback form - write the name as it stands.

## Step 2: Information the name does not encode
The result of this step is the "other_info" field.
- SCENARIO DESIGN, NOT SCENARIO RESULTS. What you write is how the scenario
  is DEFINED and SET UP: its scenario definition, scenario design, scenario
  specification, scenario protocol, experimental setup, study design,
  assumptions, imposed constraints, policy rules, exogenous parameters and
  input pathways, technology and resource availability, technology
  assumptions, socio-economic drivers and boundary conditions. It is NOT what
  the model computed under those settings.
- Relevant information includes, but is not limited to: temperature targets,
  tax, pricing adjustment, cost, policy timing, emissions constraints,
  socioeconomics, carbon budgets, policy frameworks, policy targets, policy
  changes, carbon prices, energy demand, discount rates and technology
  assumptions.
- Where the literature presents a quantity as a COMPUTED RESULT of running
  this scenario, do not write it.
- Gather what the literature says about THIS scenario that its NAME DOES NOT
  ENCODE. The dividing line is the name, not the topic: a carbon budget
  written into the name is Step 1; a carbon budget the paper imposes on this
  scenario without naming it is Step 2.
- A BACKGROUND ASSUMPTION THE STUDY SETS FOR ALL ITS SCENARIOS applies to this
  scenario too, unless this scenario departs from it. 
- SCOPE CHECK before using a statement: never take one that belongs to a
  DIFFERENT scenario, and never take one from an unrelated study.
- This step may legitimately find nothing. If the paper says nothing of this
  kind about this scenario, contribute nothing.

RULES THAT APPLY TO EVERYTHING YOU WRITE
- Do not hallucinate. Every claim must be supported by the supplied text.
- Never report a quantity that the literature presents as a computed result
  of the scenario. Write only what the scenario imposes.
- Never name or imply an IAM model or model version.
- No meta-language: no reference to the literature or the evidence, no
  "discussed", "based on", "according to".
- No remark about a token being missing, unclear or undefined (see Step 1).
- Do not put tokenization, token lists or any other intermediate working into
  the description text.

WRITING CONVENTIONS
These govern HOW you write, never WHAT you may claim: they fix the wording of
something the literature already established, and can never supply a meaning
by themselves. They apply to both output fields - "other_info" is your own
prose and is bound by them exactly as "scenario_from_tokens" is.
- Units and numbers: always CO₂ / GtCO₂ / tCO₂ with the subscript ₂, never
  "CO2". No thousands separators: 1300 GtCO₂, not 1,300 GtCO₂.
- No quotation marks anywhere in either field.
- Temperature targets: "limiting global warming to X °C", preserving the
  stated probability/percentile, peak vs end-of-century timing, any
  allowed overshoot, and other relevant information.
- Radiative forcing: "radiative forcing of X W/m²".
- Carbon budgets: where the glossary gives the accounting - cumulative or
  peak, the start year, the end point, overshoot, the gases and sectors
  covered - write it out: "cumulative fossil-fuel and industrial CO₂ emissions
  from 2011 to 2100 of 1100 GtCO₂". Fall back to the bare "a carbon budget of
  X GtCO₂" only where the glossary carries the value and not the accounting.
  Where it confirms overshoot: "a full carbon budget of X GtCO₂ that permits
  the budget to be temporarily overspent".
- SSP narratives: Sustainability (SSP1), Middle of the Road (SSP2), Regional
  Rivalry (SSP3), Inequality (SSP4), Fossil-fueled Development (SSP5).
- Policy narratives: Implemented National Policies for NPi/NPI, Nationally
  Determined Contributions for NDC, Intended Nationally Determined
  Contribution for INDC, Shared Policy Assumptions for SPA, COVID-19 for COV,
  Business as Usual for BAU.
- Years: name the role explicitly - policy start year, target year, until
  year, end year. Write "implemented national policies until 2030", not a
  bare "2030".

OUTPUT
- "scenario_from_tokens": 1-2 sentences, the first starting
  "<SCENARIO> represents ...". It identifies the scenario from its name tokens
  - family token, policy or target, budget or forcing number, and every other
  confidently interpretable token - and must stand on its own as a complete
  identification. Prefer one sentence; use a second rather than compressing a
  token's meaning or chaining clauses past readability.
- "other_info": 1 to 3 sentences carrying the Step 2 material. It is
  concatenated after "scenario_from_tokens" and must read on from it as
  continuous prose, so repeat nothing already said there. Length follows the
  evidence, not a quota; return an empty string if Step 2 found nothing, since
  padding is worse than brevity.
- "other_info_evidence": the verbatim support for "other_info". A list of
  short quotes copied character-for-character out of the supplied literature,
  one per distinct claim made in "other_info", in the same order as the claims
  appear. Copy, never paraphrase. Never quote a passage that belongs to a
  DIFFERENT scenario or an unrelated study - the SCOPE CHECK of Step 2 applies
  to the quote exactly as it applies to the claim. Return an empty list when
  "other_info" is an empty string. If you cannot copy a supporting quote for a
  claim, DROP THE CLAIM from "other_info" rather than inventing support: this
  field exists to make Step 2 checkable, so an unsupported sentence is a
  sentence that should not have been written.
  This field is NOT part of the description. The WRITING CONVENTIONS and the
  no-meta-language rule govern "scenario_from_tokens" and "other_info" only,
  and never apply here: quotes are reproduced exactly as the source spells
  them, including "CO2", quotation marks, thousands separators and any
  reference to the study itself.
There are no token fields. You are not asked which tokens you could or could
not explain: the glossary and the token list are both given to you, so which
ones are open is already known, and reporting it back could only disagree.

Return JSON:
{
  "results": [
    {"scenario": "...", "scenario_from_tokens": "...", "other_info": "...",
     "other_info_evidence": ["...", "..."]}
  ]
}

"""


def call_model(client, prompt, system=None, reasoning=None):
    """NO truncation - the full prompt is sent.
    temperature=0 and reasoning={"effort": ...} are sent as-is.
    `system` defaults to SYSTEM_PROMPT; `reasoning` defaults to REASONING, so
    each pipeline stage can pick its own effort level.
    NO fallback/downgrade: if the model rejects a parameter, the error is
    raised so you can adjust SD_MODEL / SD_REASONING / TEMPERATURE yourself."""
    system = SYSTEM_PROMPT if system is None else system
    effort = REASONING if reasoning is None else reasoning
    kwargs = dict(
        model=MODEL,
        temperature=TEMPERATURE,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ],
    )
    if effort:
        kwargs["reasoning"] = {"effort": effort}
    try:
        resp = client.responses.create(**kwargs)
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("temperature", "reasoning", "effort",
                                  "unsupported parameter", "unknown parameter")):
            # parameter rejected -> STOP the whole run (SystemExit is not
            # swallowed by the batch retry loops) so you can adjust
            # SD_MODEL / SD_REASONING / TEMPERATURE
            raise SystemExit(
                f"MODEL PARAMETER REJECTED by the API (model={MODEL}, "
                f"temperature={TEMPERATURE}, reasoning_effort={effort or '(none sent)'}):\n"
                f"  {e}\n"
                f"Adjust SD_MODEL / SD_REASONING (or TEMPERATURE in the "
                f"script) and rerun.")
        raise
    return resp.output_text


# ============================================================
# Post-processing (deterministic; no validation)
# ============================================================
def polish(desc):
    """Formatting fixes: subscript CO₂ and no thousands separators."""
    d = norm(desc)
    d = re.sub(r"CO2(?!\d)", "CO₂", d)          # CO2, GtCO2, tCO2, MtCO2 -> ...CO₂
    while True:
        d2 = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", d)   # 1,300 -> 1300
        if d2 == d:
            break
        d = d2
    return d


def canonical_start(scenario, desc):
    """If the description starts with '<name> represents' (name matched
    loosely), restore the EXACT original scenario name (the model may have
    dropped stray spaces). Otherwise return the description unchanged."""
    idx = desc.find(" represents")
    if idx > 0 and match_key(desc[:idx]) == match_key(scenario):
        return scenario + desc[idx:]
    return desc


# WHAT THIS STEP IS FOR: tokenization and token meanings are worked out once,
# up front, for the whole group: ONE call per group whose only job is to build
# the token glossary. profile_to_prompt_block() then hands that glossary to
# the generation call, where the rule "the glossary is authoritative" makes
# the model use it instead of re-deriving. Two consequences:
#   - the glossary gets a full call per group instead of a part of one shared
#     response;
#   - the consistency rule ("phrase shared tokens identically in every member")
#     becomes mechanical, because every scenario is handed the same glossary
#     string.
# An entry's `meaning` is the token's full explanation, including the design
# facts that the token switches on (accounting rule, policy definition,
# constraint).
#
# Every entry carries the verbatim quote it was drawn from. Nothing in the
# code checks it: a quote can be re-located in the PDF and still not support
# the claim attached to it, so an automatic check would only produce false
# confidence. The quote is written to the 'tokens' sheet, column
# evidence_quote, for you to read.
# ============================================================
PROFILE_SYSTEM_PROMPT = """
You are an expert in integrated assessment model (IAM) scenario protocols.

You are given the FULL text of the paper(s) that document one family of IAM
scenarios, plus the exact scenario names taken from the AR6 scenario database.

Your job is NOT to describe individual scenarios. Your job is Steps 1 and 2:
SPLIT those names into tokens, and work out what each token means, so that the
step which writes the descriptions has to do neither.

Both are yours alone. That step is handed your tokenization and your glossary
and is not allowed to redo the split or to derive a meaning of its own, so a
name you split wrongly stays split wrongly, and a token you omit is carried
through unexplained.

HARD RULES
- Use ONLY the supplied literature text. Do not use outside knowledge.
- YOU ARE THE ONLY PASS THAT READS THE LITERATURE FOR TOKEN MEANINGS. What you
  omit here is not picked up later: the step that writes the descriptions is
  given this glossary and is not allowed to derive meanings of its own, so a
  token missing from your output is carried through unexplained. Report every
  token the paper defines, including ones whose definition is spread across a
  passage rather than stated in a single sentence.
- EVERY entry must carry `evidence`: a VERBATIM quote copied
  character-for-character from the supplied text, usually 10-200 characters.
  An entry you cannot quote for must be omitted entirely.
- If the paper does not state something, return null / an empty list. Never
  fill a gap with a plausible default.
- NEVER name an integrated assessment model or a model version, anywhere in
  any field. Downstream descriptions are forbidden to mention model names, so
  a fact phrased around one is unusable. State the substance without the name:
  write "bioenergy deployment is capped at 100 EJ/yr" and not "REMIND 2.1 caps
  bioenergy deployment at 100 EJ/yr". A fact whose only content is a model's
  identity should be omitted altogether. This applies to the token glossary
  too: if a token IS a model name or version, do not put it there - list it in
  `model_name_tokens` instead, so that the steps after you know it was
  recognised and left alone rather than missed.

## Step 1: Tokenization
- Split each scenario name into tokens.
- Usually "_" and "-" are separators, but they sometimes join the parts of a
  single compound token instead. Decide from the surrounding names and from
  what the literature calls things, not from the punctuation.
- Hyphen separating - these are several tokens:
      BEG-Budg600-EG                -> BEG | Budg600 | EG
      R2p1_SSP1-PkBudg900           -> R2p1 | SSP1 | PkBudg900
      Transport_Budg1100_Conv-LowD  -> Transport | Budg1100 | Conv | LowD
      TechCost-SSP2-B1100-windH     -> TechCost | SSP2 | B1100 | windH
      SSP1-baseline                 -> SSP1 | baseline
- Hyphen joining - each of these is ONE token, and splitting it destroys it:
      CD-LINKS_INDC2080i_1000          -> CD-LINKS | INDC2080i | 1000
      Diff_1300Gt_no-transfer_def      -> Diff | 1300Gt | no-transfer | def
      Diff_1300Gt_uniform-pricing_def  -> Diff | 1300Gt | uniform-pricing | def
      Diff_No-policy_baseline          -> Diff | No-policy | baseline
      NGFS2_Net-Zero 2050              -> NGFS2 | Net-Zero 2050
  Split the wrong way, these become fragments - "no", "transfer", "uniform",
  "policy", "Net" - that mean nothing on their own and can never be resolved.
- Report the split for EVERY name you were given, in `tokenization`, using the
  exact spelling from the name.

## Step 2: Token meaning

THE TOKEN GLOSSARY IS THE MAIN OUTPUT
Scenarios in a family differ from each other ONLY through their name tokens.
So anything that differs between two members is, by construction, carried by a
token, and it belongs in that token's entry.

  THE TEST: before writing anything into an entry, ask whether it is true of
  every scenario carrying that token and false of the ones that do not. If a
  statement is about a different token, it does not belong in this entry.

  A worked example. A family contains DeepElec_SSP2_HighRE_Budg1100 and
  DeepElec_SSP2_def_Budg1100. The paper caps bioenergy at x EJ/yr for the
  first and allows y EJ/yr for the second; the names differ by one token.
  The cap therefore belongs to HighRE and to def respectively, and to neither
  of the other tokens. Attaching it to the study, or to Budg1100, would make
  it wrong for half the family. The same trap applies to the SSP baseline
  (SSP1, SSP2 and SSP5 members often share one paper), to the carbon-price
  path shape, to the budget value, and so on.

WHAT TO EXTRACT
   token_glossary one entry per DISTINCT token appearing in the supplied
                  scenario names that the paper defines. Use the token exactly
                  as it appears in the name. Omit tokens the paper does not
                  define; do not invent meanings.
                  `meaning` is the token's FULL explanation, as informative as
                  the literature allows - not a one-word gloss. Everything the
                  paper attaches to that token as part of the SCENARIO
                  DEFINITION belongs in it: the constraint it sets, the
                  assumption it switches on, the accounting rule it selects,
                  the parameter it fixes. Exclude anything the paper reports as
                  a computed result of the scenarios carrying this token.
                  For a token representing a budget, forcing level or target,
                  this means explaining what the number actually represents:
                  whether it is cumulative or peak, the accounting start year,
                  the endpoint (for example the net-zero year or 2100),
                  whether overshoot or net-negative emissions are allowed, and
                  the coverage it is accounted over - which gases (CO₂ only,
                  or Kyoto gases as CO₂-eq, and under which GWP metric) and
                  which sectors (fossil fuel and industry only, or including
                  land-use / AFOLU CO₂). "1100 GtCO₂" may not convey the full
                  meaning; "cumulative fossil-fuel and industrial CO₂
                  emissions from 2011 to 2100" does.
                  For a policy token (NPi, NDC, INDC, CurPol, Reference,
                  baseline ...): the cut-off date, the submission vintage, the
                  policy start year, and how effort is extrapolated after the
                  target year.

                  A NUMERIC TOKEN IS DEFINED ONLY WHERE THE PAPER STATES ITS
                  NUMBER. Before writing an entry for a token containing
                  digits, find those digits in the text, attached to this
                  scenario. The `evidence` for such a token must carry the
                  token's own number - either stating it outright ("a peak
                  budget of 900 GtCO2"), or giving a range or a series the
                  number plainly falls in ("budgets of 500, 900 and 1300
                  GtCO2"; "budgets from 400 to 1200 GtCO2 in 50 GtCO2
                  steps"; "budgets from 450 to 950 GtCO2"). Omit the token
                  otherwise.

                  CHECK THE VALUE AGAINST THE TOKEN. A token that carries a
                  number is naming that number. If the value you are about to
                  write differs from the token's own digits, stop - you have
                  either found the SAME quantity under a DIFFERENT ACCOUNTING,
                  or you have found a statement about a different scenario.
                    - Different accounting is real and common, and then the
                      entry carries BOTH numbers with the accounting each
                      belongs to, quoting the paper's own statement of the
                      correspondence: "an end-of-century budget of 1100 GtCO₂
                      from 2011 to 2100, which the study states as a peak
                      budget of 500 GtCO₂ from 2020 to net-zero". NEVER
                      replace the token's number with the other one - a
                      description that says 500 for a token named 1100, and
                      never mentions 1100, is wrong however good the source.
                    - No explicit correspondence means you do not have this
                      token's definition. Omit it. A paper that discusses
                      budgets of 500 and 900 has not thereby defined Budg1100
                      or Budg1300, and the nearest available number is not a
                      definition.
                  Tokens that differ only in their number must never come out
                  with the same value. If two of your readings collide, at
                  least one is wrong: omit both rather than keep the collision.

                  READING A NUMBER. A bare number in a name can be a year, a
                  price, a budget or a forcing level, and which one it is
                  decides what you should be searching the paper FOR. The cues:
                    - price cues ("carbon tax", "/tCO2", "US$",
                      "15 US$2010/tCO2") -> search for the level, its start
                      year and its escalation rule;
                    - budget or emissions cues ("CB", "Budg", "PkBudg",
                      "budget", "CO2", "cumulative emissions", "net
                      emissions") -> search for the value AND its accounting,
                      as described above;
                    - a number reading as a year (35 for 2035, 2040 for
                      itself) -> search for the role that year plays, and name
                      it: policy start year, target year, until year, end
                      year;
                    - a number reading as a forcing level (19 for 1.9, 26 for
                      2.6, 34 for 3.4, 45 for 4.5 W/m²) -> search for the
                      paper's own statement of that level;
                    - a trailing letter on an otherwise numeric token (1000f
                      beside 1000, 19I beside 19) marks a variant of the same
                      quantity -> find what the paper changes between the two
                      and put that difference in BOTH entries. "f" commonly
                      marks a full budget that may be temporarily overspent,
                      but confirm that against the paper rather than assuming
                      it;
                    - anything else -> read it neutrally, explain it based on
                      the evidence, and do not make unsupported guesses.
                  EVERY READING ABOVE IS A HYPOTHESIS, NOT A MEANING. A cue
                  tells you what to search for and hands you a candidate to
                  confirm against the paper; it is not itself evidence. A
                  reading the paper does not bear out is not a glossary entry:
                  omit the token. Nor is a sibling token a substitute - that 15
                  and 50 are carbon-tax levels in this family does not make 80
                  one.

                  Write only what the paper attaches to THIS token. Do not put
                  a statement about the whole study, or about a different
                  token, into a token's meaning.
                  WORDING. Where the token IS a standard term, open the meaning
                  with that term's standard name and put the study-specific
                  substance after it: "Middle of the Road (SSP2), in which
                  ..." rather than a paraphrase such as "a middle-of-the-road
                  socioeconomic pathway". The standard names are Sustainability
                  (SSP1), Middle of the Road (SSP2), Regional Rivalry (SSP3),
                  Inequality (SSP4), Fossil-fueled Development (SSP5);
                  Implemented National Policies (NPi/NPI), Nationally Determined
                  Contributions (NDC), Intended Nationally Determined
                  Contribution (INDC), Shared Policy Assumptions (SPA),
                  COVID-19 (COV), Business as Usual (BAU). Write CO₂ / GtCO₂ /
                  tCO₂ with the subscript ₂ and no thousands separators, and
                  never use quotation marks anywhere in `meaning` - the
                  descriptions built from it inherit its wording and are not
                  allowed to contain them.
                  If the name for a token differs from the name
                  the paper uses for the same thing (a paper may call a
                  variant "-x" where the database calls it "-y"), that
                  is still just a glossary entry: key it on the DATABASE
                  token and give the paper's meaning. Only do this where the
                  paper's own scenario table makes the correspondence
                  explicit; never guess a mapping.

                  ACRONYMS AND THE LEADING FAMILY TOKEN. A token whose
                  content is an abbreviation the paper spells out is a
                  glossary entry like any other, and its meaning is that
                  expansion. Look for it in the main text as well as in the
                  title, the abstract and the acknowledgements. This applies
                  to the LEADING token of each name.

Return JSON with exactly this shape:
{
  "tokenization": [{"scenario": "...", "tokens": ["...", "..."]}],
  "token_glossary": [{"token": "...", "meaning": "...", "evidence": "..."}],
  "model_name_tokens": ["..."]
}
"tokenization" carries every scenario name you were given.
"token_glossary" carries only the tokens the paper defines, and every one of
its `token` values must be a token that appears in "tokenization".
"model_name_tokens" carries the tokens you recognised as an IAM model name or
version and therefore did not define. Empty list if none.
Nothing else needs reporting: a token that appears in "tokenization" and in
neither of the other two lists is, by that fact alone, one you could not
define, and the steps after you read it off that difference.
"""

def profile_to_prompt_block(profile):
    """Render a profile as a compact, quote-free fact block. The quotes stay
    in the xlsx for auditing; sending them again would only add tokens and
    tempt the model to quote the literature in the description, which the
    output rules forbid."""
    if not profile:
        return "(no study facts available for this group)"
    out = []

    if profile.get("token_glossary"):
        out.append("TOKEN GLOSSARY, established for this scenario family from "
                   "the literature supplied below. Each entry applies ONLY to "
                   "a scenario whose name contains that token. Use the wording "
                   "verbatim and identically in every scenario that carries "
                   "the token, and do NOT apply an entry to a scenario that "
                   "lacks it:")
        for t in profile["token_glossary"]:
            out.append(f"  - {t.get('token')}: {t.get('meaning')}")
    return "\n".join(out)


def tokens_of(profile, scenario):
    """The tokens of one scenario name, as the PROFILE PASS split it.

    Splitting a name is a semantic judgement - "-" separates in BEG-Budg600-EG
    and joins in CD-LINKS - so it is made ONCE, by the model that can see the
    surrounding names and the papers, and every later step is handed the
    result. Nothing in this file splits a name on punctuation; doing so would
    be a second, dumber tokenizer that cannot see what the first one decided,
    and the two would disagree exactly where the judgement was hard."""
    for e in (profile or {}).get("tokenization") or []:
        if match_key(e.get("scenario")) == match_key(scenario):
            return [str(t).strip() for t in (e.get("tokens") or []) if str(t).strip()]
    return []


def open_tokens_of(profile, known, scenario):
    """The tokens of `scenario` that no glossary entry covers.

    DERIVED, NEVER DECLARED. The profile pass reports every token of every name
    and every token it could define, so what is open is a subtraction.

    Subtracting the group glossary is safe because every glossary entry exists
    BEFORE any description is written: the retry rounds finish first, and the
    generation pass runs last, once.

    Model names are excluded: they are recognised, deliberately left verbatim,
    and never open."""
    skip = {str(t).strip().lower()
            for t in (profile or {}).get("model_name_tokens") or []}
    return [t for t in tokens_of(profile, scenario)
            if t not in (known or {}) and t.lower() not in skip]


def render_tokenization(profile, scenarios):
    """The per-scenario token list, as the generation and spec passes see it."""
    lines = []
    for s in sorted(scenarios):
        toks = tokens_of(profile, s)
        if toks:
            lines.append(f"  {s}: {' | '.join(toks)}")
    return "\n".join(lines)


def build_study_profile(client, lit_ids, scenarios):
    """One call per literature group -> the token glossary for that group."""
    blocks = [f"[{lid}]\n{get_pdf_text(lid)}" for lid in lit_ids
              if get_pdf_text(lid)]
    context = "\n\n".join(blocks)
    names = "\n".join(f"  {s}" for s in sorted(scenarios))
    prompt = f"""
    Scenario names in this family (from the AR6 database). Split every one of
    them, then define the tokens the literature defines:
{names}

    Work in that order. Split the names first, comparing them against each
    other - a segment that varies while the rest of a name stays fixed is a
    token, and the same segment must be split the same way in every name.
    Then go to the literature for each distinct token you produced.
    A token's meaning comes from the literature and from nowhere else. The
    shape of a name tells you what to search for, never what a token means: if
    the papers below do not settle a token, omit it, and do not fall back on
    what the digits or the spelling suggest.

    Literature:
    {context}
    """
    for attempt in (1, 2):
        try:
            obj = extract_json(call_model(client, prompt,
                                          system=PROFILE_SYSTEM_PROMPT,
                                          reasoning=REASONING_PROFILE))
            return obj
        except Exception as e:
            print(f"  PROFILE ERROR attempt {attempt}: {e}")
    return {}


def retry_study_profile(client, lit_ids, profile, open_toks):
    """A SECOND profile call, aimed at the tokens the first one left open.

    Same system prompt, same output shape, same evidence rule - only the
    request differs, and only in telling the model which tokens to go after.
    It is the profile pass run again with a shorter list, so it produces
    ordinary glossary entries. The writing is left to the generation pass,
    which runs once, last, with the finished glossary."""
    blocks = [f"[{lid}]\n{get_pdf_text(lid)}" for lid in lit_ids
              if get_pdf_text(lid)]
    context = "\n\n".join(blocks)
    tok_block = render_tokenization(profile, [e.get("scenario") for e
                                              in (profile or {}).get("tokenization") or []])
    known = "\n".join(f"  - {t.get('token')}: {t.get('meaning')}"
                      for t in (profile or {}).get("token_glossary") or [])
    prompt = f"""
    The names in this family are already split, and most of their tokens are
    already defined. Your job now is ONLY the tokens still open.

    Tokens still open: {", ".join(sorted(open_toks))}

    Return "tokenization" exactly as given below, unchanged, and a
    "token_glossary" containing ONLY the open tokens you can now define. Do not
    repeat entries that already exist, and do not revise them.

    How the names are split:
{tok_block}

    Already defined, for context - these are settled, and a token's neighbours
    often say what to search for:
{known or "  (none)"}

    Search the literature below for each open token. Useful places, in order:
    an explicit scenario-naming convention, a scenario-design or scenario-
    overview table, the methods section, and figure captions. For a token that
    looks like an abbreviation, and for the LEADING token of a name, look
    instead where a paper names its project: the title, the abstract, the
    acknowledgements and the opening paragraphs.
    Look for a DEFINITION of the token, not for results reported under it.

    An open token you still cannot define is a correct answer: leave it out.
    A fabricated one is not.

    Literature:
    {context}
    """
    for attempt in (1, 2):
        try:
            obj = extract_json(call_model(client, prompt,
                                          system=PROFILE_SYSTEM_PROMPT,
                                          reasoning=REASONING_SPEC))
            return [e for e in (obj.get("token_glossary") or [])
                    if isinstance(e, dict) and str(e.get("token", "")).strip()]
        except Exception as e:
            print(f"  PROFILE RETRY ERROR attempt {attempt}: {e}")
    return []


def compose(scenario, from_tokens, other_info):
    """Polish both halves of a description and join them.

    Returns (final_text, polished_from_tokens, polished_other_info). The two
    polished halves are what gets stored, so the 'other_info' sheet shows the
    exact words that ended up in the description rather than the raw model
    output - otherwise the sheet would still read "1,300 GtCO2" where the
    description reads "1300 GtCO₂"."""
    a = canonical_start(scenario, polish(from_tokens or ""))
    b = polish(other_info or "")
    return ((a + " " + b).strip() if b else a), a, b


def _merged_profile(profiles, glossary, lit_ids):
    """The profile pass's own output PLUS everything the retry rounds added.

    A profile is what the FIRST call returned, so on its own it goes stale the
    moment a retry round defines something. Round 2 handed only that would go
    back to the PDFs for a token round 1 had already found. Merging keeps every
    later step - the next retry, and the generation pass - working from one
    up-to-date view."""
    base = profiles.get(tuple(lit_ids)) or {}
    learned = glossary.get(tuple(lit_ids), {})
    if not learned:
        return base
    known = {str(e.get("token", "")).strip()
             for e in (base.get("token_glossary") or [])}
    extra = [{"token": t, "meaning": e.get("meaning", "")}
             for t, e in sorted(learned.items())
             if t not in known and e.get("meaning")]
    if not extra:
        return base
    return {**base,
            "token_glossary": list(base.get("token_glossary") or []) + extra}


def complete_glossaries(client, group_list, profiles, glossary, rounds=None):
    """Fill the gaps the first profile pass left, before any description is
    written.

    A group's open tokens are its tokenization minus what the glossary covers
    minus the model names - a subtraction over the WHOLE group, with no
    per-scenario state, because nothing has been written yet. Each round asks
    the literature about that list and folds what comes back into `glossary`;
    the next round's subtraction is therefore the pruning.

    Stops as soon as a round adds nothing, and runs at most `rounds` times: a
    third attempt on tokens two searches already failed to find is a full-text
    call spent on a question already answered."""
    rounds = SPEC_ROUNDS if rounds is None else rounds
    for rnd in range(1, rounds + 1):
        learned = 0
        for lit_ids, sc_list in group_list:
            if not profiles.get(tuple(lit_ids)):
                continue
            known = glossary.setdefault(tuple(lit_ids), {})
            prof = _merged_profile(profiles, glossary, lit_ids)
            open_toks = sorted({t for s in sc_list
                                for t in open_tokens_of(prof, known, s)})
            if not open_toks:
                continue
            print(f"glossary retry r{rnd}: {len(open_toks)} open token(s) "
                  f"{open_toks} | lits {list(lit_ids)}")
            for e in retry_study_profile(client, list(lit_ids), prof, open_toks):
                t = canon_token_from(open_toks, e.get("token"))
                if t and t not in known and str(e.get("meaning", "")).strip():
                    known[t] = {"meaning": str(e.get("meaning", "")).strip(),
                                "source": f"glossary retry round {rnd}",
                                "evidence": str(e.get("evidence", "")).strip()}
                    learned += 1
                    print(f"  DEFINED {t}: {known[t]['meaning'][:70]}")
        print(f"glossary retry round {rnd}: {learned} new token meaning(s)")
        if learned == 0:
            break


# ============================================================
# PDF loading - FULL text, no per-PDF cap
# ============================================================
_pdf_cache = {}


def get_pdf_text(lit_id):
    if lit_id in _pdf_cache:
        return _pdf_cache[lit_id]
    from pypdf import PdfReader
    path = PDF_DIR / f"{lit_id}.pdf"
    if not path.exists():
        print(f"  WARNING: PDF missing: {path}")
        _pdf_cache[lit_id] = ""
        return ""
    texts = []
    for p in PdfReader(str(path)).pages:
        try:
            t = p.extract_text()
            if t:
                texts.append(str(t).strip())
        except Exception:
            pass
    _pdf_cache[lit_id] = "\n".join(texts)
    return _pdf_cache[lit_id]


# ============================================================
# literature map
# ============================================================
def build_lit_map():
    """scenario -> refs from SCENREF_XLSX; ref -> literature_ids from LIT_FILE
    (multi-ref cells split on ';' and paired positionally with their ids)."""
    sr = pd.read_excel(SCENREF_XLSX, sheet_name="scenario_reference")
    scenario_to_refs = defaultdict(set)
    for _, row in sr.iterrows():
        sc = norm(row.get("scenario"))
        for ref in str(row.get("literature_reference", "")).split(";"):
            ref = norm(ref)
            if sc and ref and ref.lower() != "nan":
                scenario_to_refs[sc].add(ref)

    sheets = pd.read_excel(LIT_FILE, sheet_name=None)
    if "literature_reference" in sheets:
        id_df = sheets["literature_reference"]
    else:                                    # older workbooks: fall back to position
        id_df = sheets[list(sheets.keys())[1]]
    ref_to_ids = defaultdict(set)
    for _, row in id_df.iterrows():
        refs = [norm(p) for p in str(row.get("literature_reference", "")).split(";")
                if norm(p)]
        ids = [x.strip() for x in re.split(r"[;,]", str(row.get("literature_id", "")))
               if x.strip()]
        if not refs or not ids:
            continue
        if len(refs) == len(ids):
            for ref, lid in zip(refs, ids):
                ref_to_ids[ref].add(lid)
        else:
            for ref in refs:
                ref_to_ids[ref].update(ids)

    canon = {re.sub(r"\s+", "", r).lower(): r for r in ref_to_ids}

    def resolve(ref):
        if ref in ref_to_ids:
            return ref
        return canon.get(re.sub(r"\s+", "", ref).lower())

    unmatched = sorted({r for refs in scenario_to_refs.values() for r in refs
                        if resolve(r) is None})
    if unmatched:
        print(f"WARNING: {len(unmatched)} literature_reference value(s) have no "
              f"literature_id in {LIT_FILE.name}:")
        for r in unmatched:
            print("   -", r[:100])

    resolved = {sc: {resolve(r) for r in refs if resolve(r)}
                for sc, refs in scenario_to_refs.items()}
    return resolved, ref_to_ids


def load_target_scenarios():
    """Unique scenarios (and their current descriptions) from TARGET_XLSX."""
    sheets = pd.read_excel(TARGET_XLSX, sheet_name=None)
    baseline = {}
    for df in sheets.values():
        if "scenario" not in df.columns:
            continue
        for _, row in df.iterrows():
            sc = norm(row["scenario"])
            if not sc or sc in baseline:
                continue
            desc = row.get("scenario_description")
            baseline[sc] = "" if pd.isna(desc) else str(desc).strip()
    return baseline


def build_groups(scenarios, scenario_to_refs, ref_to_ids):
    groups = defaultdict(list)
    no_lit = []
    for s in scenarios:
        ids = set()
        for r in scenario_to_refs.get(s, set()):
            ids |= ref_to_ids.get(r, set())
        if ids:
            groups[tuple(sorted(ids))].append(s)
        else:
            no_lit.append(s)
    return groups, no_lit


def process_batch(client, lit_ids, scenarios, retry=False, profile=None,
                  known=None):
    """One LLM call: study facts + full literature context of the group + a
    batch of scenarios. retry=True marks scenarios a previous call failed to
    return. `known` is the glossary for this group, used to mark which of each
    name's tokens have no meaning yet - the model is told, not asked."""
    blocks = []
    for lid in lit_ids:
        t = get_pdf_text(lid)
        if t:
            blocks.append(f"[{lid}]\n{t}")
    context = "\n\n".join(blocks)

    note = ("\nA previous request did not return a description for this "
            "scenario; it MUST be included in the results." if retry else "")
    models = {str(t).strip() for t in (profile or {}).get("model_name_tokens") or []}

    def _block(s):
        toks = tokens_of(profile, s)
        open_ = open_tokens_of(profile,
                               [t for t in tokens_of(profile, s) if t in (known or {})],
                               s)
        mods = [t for t in toks if t in models]
        out = [f"Current scenario: {s}",
               f"Tokens: {' | '.join(toks) or '(not tokenized)'}"]
        if open_:
            out.append(f"No meaning established: {', '.join(open_)}")
        if mods:
            out.append(f"Model name or version: {', '.join(mods)}")
        out.append(f"Write one description only for this current scenario.{note}")
        return "\n".join(out)

    scenario_blocks = "\n\n".join(_block(s) for s in scenarios)
    # profile_to_prompt_block() labels itself, so no wrapper header here
    facts_block = (profile_to_prompt_block(profile) + "\n") if profile else ""

    prompt = f"""
    For each block below, treat "Current scenario" as the scenario being described.

    Each block gives that name's TOKENS, in order, already split. Use them as
    they stand: they are the complete list, the family token is the first of
    them, and no other substring of the name is a token. Do not re-split.

    {facts_block}
    {scenario_blocks}

    Literature:
    {context}
    """
    for attempt in (1, 2):
        try:
            obj = extract_json(call_model(client, prompt))
            return obj.get("results", [])
        except Exception as e:
            print(f"  BATCH ERROR attempt {attempt} ({len(scenarios)} scenarios): {e}")
    return []


def generate_all(client, scenarios, scenario_to_refs, ref_to_ids, limit=0):
    """Profile pass, glossary retries, then generation - in that order, so the
    generation pass writes once, last, from a glossary nothing will add to.
    Returns (desc, parts, glossary, profiles, no_lit, group_list)."""
    groups, no_lit = build_groups(scenarios, scenario_to_refs, ref_to_ids)
    print(f"groups: {len(groups)} | scenarios without literature: {len(no_lit)}")

    group_list = sorted(groups.items())
    if limit:
        group_list = group_list[:limit]
        print(f"(SD_LIMIT={limit}: only the first {len(group_list)} group(s))")

    # ---- STAGE A: study-profile pass ----------------------------------
    profiles = {}
    if PROFILE_PASS:
        for gi, (lit_ids, sc_list) in enumerate(group_list, 1):
            print(f"Profile {gi}/{len(group_list)}: lits {list(lit_ids)} | "
                  f"{len(sc_list)} scenarios")
            prof = build_study_profile(client, list(lit_ids), sc_list)
            profiles[tuple(lit_ids)] = prof
            print(f"  {len(prof.get('token_glossary', []))} token(s) in glossary")

    new_desc, parts, oi_ev = {}, {}, {}
    # token -> meaning, per literature group. Seeded by the profile pass and
    # completed by the retry rounds below; frozen before anything is written.
    glossary = {}
    for lit_ids, prof in profiles.items():
        for e in prof.get("token_glossary") or []:
            t = str(e.get("token", "")).strip()
            if t:
                glossary.setdefault(lit_ids, {})[t] = {
                    "meaning": str(e.get("meaning", "")).strip(),
                    "source": "profile pass",
                    "evidence": str(e.get("evidence", "")).strip()}

    # ---- STAGE A2: glossary retries -----------------------------------
    # Everything that will ever be in the glossary goes in HERE, before a
    # single description exists. That ordering is the whole point: the
    # generation pass then writes each scenario once, from a finished
    # glossary, instead of writing a fallback and being patched afterwards.
    if SPEC_PASS:
        complete_glossaries(client, group_list, profiles, glossary)

    def run_group(lit_ids, todo, label, retry=False):
        n_batches = (len(todo) + SCEN_BATCH - 1) // SCEN_BATCH
        prof = _merged_profile(profiles, glossary, lit_ids)
        for b in range(n_batches):
            batch = todo[b * SCEN_BATCH:(b + 1) * SCEN_BATCH]
            res = process_batch(client, list(lit_ids), batch, retry=retry,
                                profile=prof,
                                known=glossary.get(tuple(lit_ids), {}))
            key2orig = {match_key(s): s for s in batch}
            got = 0
            for r in res:
                s = key2orig.get(match_key(r.get("scenario")))
                ft = str(r.get("scenario_from_tokens", "")).strip()
                oi = str(r.get("other_info", "")).strip()
                if not s or not ft or s in new_desc:
                    continue
                new_desc[s], part_a, part_b = compose(s, ft, oi)
                parts[s] = (part_a, part_b)
                if OI_EVIDENCE:
                    ev = r.get("other_info_evidence") or []
                    if isinstance(ev, str):
                        ev = [ev]
                    oi_ev[s] = [q for q in (str(x).strip() for x in ev) if q]
                got += 1
            print(f"  {label} batch {b + 1}/{n_batches}: {len(batch)} scenarios "
                  f"-> {got} descriptions")

    # ---- STAGE B: generation ------------------------------------------
    for gi, (lit_ids, sc_list) in enumerate(group_list, 1):
        ctx_chars = sum(len(get_pdf_text(l)) for l in lit_ids)
        print(f"Group {gi}/{len(group_list)}: {len(sc_list)} scenarios | "
              f"lits {list(lit_ids)} | context {ctx_chars:,} chars")
        run_group(lit_ids, sc_list, "initial")

    # ---- retry rounds for scenarios the model did not return ----------
    for rnd in range(1, RETRY_ROUNDS + 1):
        n_todo = sum(1 for _, sc_list in group_list
                     for s in sc_list if s not in new_desc)
        if not n_todo:
            break
        print(f"retry round {rnd}/{RETRY_ROUNDS}: {n_todo} missing scenario(s)")
        for lit_ids, sc_list in group_list:
            todo = [s for s in sc_list if s not in new_desc]
            if todo:
                run_group(lit_ids, todo, f"retry{rnd}", retry=True)

    unresolved = [s for _, sc_list in group_list for s in sc_list
                  if s not in new_desc]
    if unresolved:
        print(f"STILL MISSING after {RETRY_ROUNDS} retry round(s): {unresolved}")
    n_open = 0
    for lit_ids, sc_list in group_list:
        prof = profiles.get(tuple(lit_ids))
        for s in sc_list:
            if s in new_desc and open_tokens_of(prof, glossary.get(tuple(lit_ids), {}), s):
                n_open += 1
    print(f"generation: {n_open} scenario(s) carry at least one token with no "
          f"meaning yet")
    return new_desc, parts, oi_ev, glossary, profiles, no_lit, group_list


def save_output(new_desc, parts, oi_ev, baseline, glossary, group_list,
                no_lit, profiles, out_xlsx):
    """Four sheets, in the order you read them:

      tokens        what the PROFILE PASS and the SPEC ROUNDS decided a token
                    means, and which of the two decided it. Those are the only
                    two passes that work meanings out, so this sheet plus
                    "unresolved" account for every token (see the module
                    docstring for the one leak left)
      unresolved    what is still declared unresolved after the spec rounds
      other_info    the Step 2 half of each description, on its own, with the
                    verbatim quote(s) the model copied to support it
      descriptions  the final text that goes into scenario_description
    """
    scenarios = sorted(baseline)
    scen2grp = {sc: lit for lit, members in group_list for sc in members}
    # token -> scenarios, read off the profile pass's tokenization (the only
    # tokenization there is) rather than re-split here.
    occ = defaultdict(list)
    for lit_ids, sc_list in group_list:
        prof = profiles.get(tuple(lit_ids))
        for sc in sorted(sc_list):
            for t in tokens_of(prof, sc):
                if sc not in occ[t]:
                    occ[t].append(sc)

    tok_rows = []
    for lit_ids, entries in sorted(glossary.items()):
        for token, e in sorted(entries.items()):
            here = [sc for sc in occ.get(token, [])
                    if scen2grp.get(sc) == lit_ids]
            tok_rows.append({
                "token": token,
                "n_scenarios": len(here),
                "scenarios": ", ".join(here),
                "meaning": e.get("meaning", ""),
                "source": e.get("source", ""),
                "literature_group": ", ".join(lit_ids),
                "evidence_quote": e.get("evidence", ""),
            })

    # Derived here exactly as the generation pass derived it: a token of the
    # name that no glossary entry covers and that is not a model name. Nothing
    # declares it, so nothing can forget to.
    unres_rows = []
    for lit_ids, sc_list in group_list:
        prof = profiles.get(tuple(lit_ids))
        known = glossary.get(tuple(lit_ids), {})
        for sc in sorted(sc_list):
            toks = tokens_of(prof, sc)
            for t in open_tokens_of(prof, known, sc):
                unres_rows.append({
                    "scenario": sc, "token": t,
                    "form": "family token" if toks and t == toks[0] else "inner token",
                    "literature_group": ", ".join(lit_ids)})

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        pd.DataFrame(tok_rows, columns=[
            "token", "n_scenarios", "scenarios", "meaning", "source",
            "literature_group", "evidence_quote"]).to_excel(
            w, index=False, sheet_name="tokens")
        pd.DataFrame(unres_rows, columns=[
            "scenario", "token", "form", "literature_group"]).to_excel(
            w, index=False, sheet_name="unresolved")
        pd.DataFrame([{"scenario": sc,
                       "other_info": parts.get(sc, ("", ""))[1],
                       "n_evidence": len(oi_ev.get(sc, [])),
                       "evidence_quote": "\n\n".join(oi_ev.get(sc, []))}
                      for sc in scenarios],
                     columns=["scenario", "other_info", "n_evidence",
                              "evidence_quote"]).to_excel(
            w, index=False, sheet_name="other_info")
        pd.DataFrame([{"scenario": sc, "description": new_desc.get(sc, "")}
                      for sc in scenarios]).to_excel(
            w, index=False, sheet_name="descriptions")

    missing = [sc for sc in scenarios if not new_desc.get(sc)]
    print(f"Saved: {out_xlsx}  (generated {len(scenarios) - len(missing)}/"
          f"{len(scenarios)})")
    if missing:
        nolit = [sc for sc in missing if sc in set(no_lit)]
        skipped = [sc for sc in missing
                   if sc not in scen2grp and sc not in set(no_lit)]
        failed = [sc for sc in missing if sc in scen2grp]
        print(f"  WARNING: {len(missing)} scenario(s) have an EMPTY "
              f"description in the 'descriptions' sheet")
        if nolit:
            print(f"    {len(nolit)} have no literature mapping at all, "
                  f"e.g. {nolit[0]}")
        if skipped:
            print(f"    {len(skipped)} were outside the processed groups "
                  f"(SD_LIMIT), e.g. {skipped[0]}")
        if failed:
            print(f"    {len(failed)} were not returned by the model after "
                  f"{RETRY_ROUNDS} retry round(s), e.g. {failed[0]}")
    print(f"  tokens {len(tok_rows)} defined | {len(unres_rows)} scenario/token "
          f"pair(s) still unresolved")


def main():
    from openai import OpenAI
    client = OpenAI(api_key=load_api_key())

    for p, what in [(TARGET_XLSX, "target dataset"), (SCENREF_XLSX, "scenario-reference file"),
                    (LIT_FILE, "literature map"), (PDF_DIR, "PDF folder")]:
        if not p.exists():
            raise SystemExit(f"{what} not found: {p}")

    baseline = load_target_scenarios()
    print(f"target: {TARGET_XLSX.name} | unique scenarios: {len(baseline)}")
    scenario_to_refs, ref_to_ids = build_lit_map()

    (new_desc, parts, oi_ev, glossary, profiles, no_lit,
     group_list) = generate_all(
        client, sorted(baseline), scenario_to_refs, ref_to_ids, limit=LIMIT)

    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    out_xlsx = OUT_DIR / f"scenario_desc_gen_{MODEL}_{stamp}.xlsx"
    save_output(new_desc, parts, oi_ev, baseline, glossary,
                group_list, no_lit, profiles, out_xlsx)


if __name__ == "__main__":
    main()
