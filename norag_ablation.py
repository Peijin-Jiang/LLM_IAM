"""
norag_ablation.py - the NO-RETRIEVAL ablation of the RAG pipeline.

The LLM gets ONLY the TARGET's scenario_description_text and
model_fingerprint_text (plus the task/output contract) - no neighbours, no
trajectories from any other case. Everything else is rag_core's own machinery,
imported and reused: same data loading, same Gemini client and JSON-schema
constrained decoding, same concurrency/retry/usage accounting, same pred-CSV
format and the same multi-sheet eval report. So the ONLY difference between
this run and a RAG run is the evidence in the prompt.

This is a stronger ablation than topk-zero inside rag_core: there, the prompt
keeps all the neighbour-reading instructions with an empty NEIGHBOR_ALL block;
here the prompt is rewritten so the model is not told to look for evidence that
is not there.

PROMPT: what was kept and what was cut (relative to rag_core.build_prompt)
  kept   - role line, task definition, years, output-JSON-only rule
         - the fingerprint semantics paragraph (covers only some variables;
           "Not Available" / unmentioned = no information, never "average")
         - the anchors-first decision process (2100 level, turning point),
           minus every reference to neighbours
         - sectoral-consistency constraints, non-negativity, output rules,
           exact variable list, strict output format, TARGET (JSON)
  cut    - "EVIDENCE PROVIDED" neighbour inventory
         - all of Step 1 (neighbour-reading rules, neighbor_type semantics,
           cross-family fingerprint translation, stringency comparison
           AGAINST neighbours)
         - neighbour-anchored parts of Steps 2-3
         - "Use only the provided evidence" (replaced: ground the numbers in
           the two TARGET texts and general knowledge of this model family
           and scenario class; stay conservative)

Usage (from the project root): same env contract as rag_core -
    IAM_LLM_BATCH_SHEET_PATH=make_datasets/output/split/AR6_R10sd_test_list.xlsx \
    python3 norag_ablation.py
Output: <dataset>_pred_<stamp>_<model>_norag_fp-..._desc-on_..._sd22.csv + eval
xlsx with the usual run_config timing/cost rows, under outputs/.
"""

import os

# Before importing rag_core: no retrieval work, no rerank calls. The retrieval
# path still executes (synthesize_one_target calls it) but returns empty
# frames instantly and spends nothing; the norag prompt ignores it anyway.
os.environ["IAM_LLM_TOPK_SCEN"] = "0"
os.environ["IAM_LLM_TOPK_MODEL"] = "0"
os.environ["IAM_LLM_RERANK"] = "0"
os.environ["IAM_LLM_RERANK_MODE"] = "off"   # keeps the RUN_TAG honest (no _rrll15x3 suffix)

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag_core

YEARS = rag_core.YEARS if hasattr(rag_core, "YEARS") else [str(y) for y in range(2010, 2110, 10)]


def build_prompt_norag(target_block, neighbors_all, vars_out, total_var,
                       tech_vars, include_per_var_rationale):
    """Drop-in replacement for rag_core.build_prompt (same signature, called by
    synthesize_one_target). neighbors_all is ignored by design."""
    fp_scope = rag_core.fp_scope_for(vars_out)
    fp_label = rag_core.FP_PREF_LABEL

    soft_section = ""
    _vs_set = {str(v).strip() for v in vars_out}
    if rag_core.VARSET == "sd22" and "Final Energy|Industry|Electricity" in _vs_set:
        soft_section = (
            "NUMERIC CONSISTENCY CONSTRAINTS:\n\n"
            "Sectoral consistency (must hold for each year y):\n"
            ' - "Final Energy|Industry|Electricity"[y] <= "Final Energy|Industry"[y]\n'
            ' - "Final Energy|Transportation|Electricity"[y] <= "Final Energy|Transportation"[y]\n'
            ' - "Final Energy|Residential and Commercial|Electricity"[y] <= '
            '"Final Energy|Residential and Commercial"[y]\n\n')

    target_json = json.dumps(target_block, ensure_ascii=False, indent=2)
    vars_json = json.dumps([str(v) for v in vars_out], ensure_ascii=False)

    per_var_block = ""
    pre_fields = ""
    if include_per_var_rationale:
        per_var_block = f"""

PER-VARIABLE RATIONALES REQUIREMENT:
- Provide a concise rationale for EACH variable trajectory in 2-4 sentences: (1) what the TARGET scenario_description_text implies for HOW FAST and HOW FAR this variable changes; (2) where the {fp_label} speaks to this variable, how it shifts the LEVEL - and where it does not, say so in one clause. Do not produce vague statements or claims the two TARGET texts cannot support.
""".rstrip()
        pre_fields = ('\n  "per_variable_rationales": { "<exact variable name>": '
                      '"<rationale; write these FIRST>", "...": "..." },')

    prompt = f"""
You are an expert in integrated assessment model (IAM) scenario synthesis for IPCC AR6-style databases.
Your task is to generate decade-step numeric trajectories for a TARGET model_family × scenario from the TARGET information.

You must reason carefully and step by step internally, but you must output JSON only.
Do NOT include any extra text before or after the JSON.

TASK:
Generate, for the TARGET model_family × scenario, full trajectories (no interpolation) for ALL required variables, for years: {", ".join(YEARS)}


DECISION PROCESS (FOLLOW INTERNALLY IN THIS ORDER):

Step 1: Read the two TARGET texts.
- Use the scenario_description_text to determine the mitigation stringency and constraints - climate target, carbon budget, technology availability, policy timing. These determine HOW FAST and HOW FAR each variable changes.
- Use the model_fingerprint_text to adjust LEVELS: {fp_label} states, for a few of the predicted variables ({fp_scope}), whether this model family tends to sit at a higher or lower LEVEL than other models.
- The fingerprint may read "Not Available", or may simply not mention a variable. That means no cross-model evidence exists for it: the information is missing, and this is never itself a property of the model. Do not treat it as average, neutral or mid-range.
- Beyond these two texts, ground magnitudes in your general knowledge of scenarios for this region, model family and scenario class. Values are in EJ/yr for the stated region, not global totals.

Step 2: Fix the long-horizon anchors before writing any numbers.
- For each variable, decide the 2100 LEVEL implied by the scenario's mitigation stringency and the region's plausible scale, then the TURNING POINT (the decade where the variable peaks, or where it begins its terminal decline, or "monotone" if it has neither).

Step 3: Generate the decade path subject to all consistency constraints.
- Fill in the decades so the path passes through the turning point from Step 2 and lands on its 2100 level.
- Ensure non-negativity.
- Avoid unrealistic decade-to-decade oscillations.

{soft_section}NON-NEGATIVITY:
- Every value must be >= 0 for every year.

GENERAL OUTPUT RULES:
- Output MUST be valid JSON.
- Values must be numeric only.
- Do not output NaN.
- Do not output strings for numeric values.
- Avoid null whenever possible.
- Do not omit required variables.
- Use exact variable names.
- If the evidence is weak, stay conservative: smooth, plausible paths at realistic regional magnitudes rather than invented detail.{per_var_block}

VARIABLE LIST (must include EXACTLY these variables, each exactly once):
{vars_json}

OUTPUT JSON FORMAT (STRICT; FOLLOW EXACTLY, INCLUDING FIELD ORDER):
{{{pre_fields}
  "trajectories": [
    {{
      "variable": "<exact variable name>",
      "unit": "<unit or null>",
      "values": {{
        "2010": <number>,
        "2020": <number>,
        "2030": <number>,
        "2040": <number>,
        "2050": <number>,
        "2060": <number>,
        "2070": <number>,
        "2080": <number>,
        "2090": <number>,
        "2100": <number>
      }}
    }}
  ]
}}

TARGET (JSON):
{target_json}

Return JSON only.
""".strip()
    return prompt


# ── patch rag_core: prompt + run naming ──────────────────────────────────────
rag_core.build_prompt = build_prompt_norag
_new_tag = rag_core.RUN_TAG.replace("_topk000", "_norag")
assert _new_tag != rag_core.RUN_TAG, "expected _topk000 in RUN_TAG (topk envs are set above)"
rag_core.RUN_TAG = _new_tag
rag_core.OUT_BATCH_CSV = os.path.join(rag_core.OUTPUTS_DIR,
                                      f"{rag_core.DATASET_TAG}_pred_{_new_tag}.csv")
# run_evaluation() reads these module globals, bound at rag_core import time -
# rebind them to the norag names (unless the caller overrode them via env)
if not os.environ.get("IAM_LLM_PRED_PATH"):
    rag_core.PRED_PATH = rag_core.OUT_BATCH_CSV
if not os.environ.get("IAM_LLM_EVAL_PATH"):
    rag_core.OUT_XLSX = os.path.join(rag_core.OUT_DIR,
                                     f"{rag_core.DATASET_TAG}_eval_{_new_tag}.xlsx")

if __name__ == "__main__":
    # mirrors rag_core's own MODE=="batch" __main__ flow, step for step
    rag_core.INCLUDE_PER_VAR_RATIONALE = False       # batch runs disable rationales
    rag_core.print_resolved_config()
    print(f"[NORAG] neighbour retrieval disabled; prompt = target texts only\n"
          f"[NORAG] pred -> {rag_core.OUT_BATCH_CSV}")
    dfv = rag_core.load_validation_xlsx(rag_core.BATCH_SHEET_PATH)
    _lim = int(os.environ.get("TEST_LIMIT", "0"))
    if _lim > 0:
        dfv = dfv.head(_lim)
        print(f"[NOTE] TEST_LIMIT={_lim}: running only the first {_lim} targets.")
    if rag_core.SAMPLE_N:
        dfv = rag_core.stratified_sample(dfv, rag_core.SAMPLE_N, rag_core.SAMPLE_SEED)
    rag_core.assert_batch_gt_consistent(dfv, rag_core.GT_PATH)

    _t0 = time.time()
    out_df = asyncio.run(rag_core.run_batch(dfv))
    rag_core.RUN_STATS["elapsed_sec"] = round(time.time() - _t0, 1)
    _n = len(dfv)
    if _n:
        rag_core.RUN_STATS["sec_per_case_throughput"] = round(
            rag_core.RUN_STATS["elapsed_sec"] / _n, 4)
    if rag_core.CASE_SECS:
        cs = rag_core.CASE_SECS
        rag_core.RUN_STATS["sec_per_case_latency_measured"] = round(sum(cs) / len(cs), 2)
        rag_core.RUN_STATS["sec_per_case_latency_p50"] = round(statistics.median(cs), 2)
        if len(cs) >= 20:
            rag_core.RUN_STATS["sec_per_case_latency_p95"] = round(
                statistics.quantiles(cs, n=20)[18], 2)
        rag_core.RUN_STATS["sec_per_case_latency_max"] = round(max(cs), 2)
    rag_core.RUN_STATS.update(rag_core.USAGE)
    rag_core.RUN_STATS["est_cost_usd"] = round(rag_core.usage_cost(), 4)
    rag_core.RUN_STATS["est_cost_lost_max_usd"] = round(rag_core.lost_cost_upper_bound(), 4)

    out_df = rag_core.round_year_cols(out_df)
    out_df.to_csv(rag_core.OUT_BATCH_CSV, index=False)
    print(f"Saved batch CSV: {rag_core.OUT_BATCH_CSV}")
    print(f"[RUN STATS] wall {rag_core.RUN_STATS['elapsed_sec']}s | "
          f"cost ${rag_core.RUN_STATS['est_cost_usd']} | "
          f"calls {rag_core.USAGE['calls']} (failed {rag_core.USAGE['failed_calls']})")

    rag_core.run_evaluation()
