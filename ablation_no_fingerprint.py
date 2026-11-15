#!/usr/bin/env python3
"""
ablation_no_fingerprint.py - "no model fingerprint" ablation for rag_core.

WHAT THIS IS
============
Companion to ablation_no_scenario_desc.py. Also a thin wrapper, not a fork.

The fingerprints come from an Excel file that can be replaced, and
no_fingerprint.xlsx holds the same 9 model families with every field set to
"Not Available". The whole ablation is therefore two environment variables,
set BEFORE rag_core is imported (it reads them at module level):

    IAM_LLM_FINGERPRINT_PATH = <project root>/no_fingerprint.xlsx
    IAM_LLM_FP_SHEET         = no_fingerprint

The sheet inside no_fingerprint.xlsx is called "no_fingerprint", not
"fingerprint", so both variables are required.

FULL-CHAIN, LIKE THE DESC ABLATION
==================================
build_fp_text() turns each row into

    Energy System Preference: Not Available

which is non-empty, so get_model_family_fp_text()'s "empty fingerprint" guard
does not fire - the text is *unavailable*, not *missing*. Because every family
now shares one identical string, the ablation reaches both paths:

  RETRIEVAL   step2's family ranking embeds fp_text per family and scores by
              cosine similarity. One distinct text -> one vector -> every
              family ties, and the ordering degenerates to source order.
              Deterministic, but carrying no signal.

              The LLM MODEL reranker still runs and is still billed; it is shown
              RERANK_POOL_MODEL identical strings and returns an arbitrary
              order. Pipeline unchanged, evidence gone - same stance as the desc
              ablation. IAM_LLM_RERANK_MODE=off is cheaper but is then a
              different pipeline from the baseline.

              The SCENARIO reranker ranks by description text and is untouched.

  SYNTHESIS   build_target_block() and build_neighbors_all_evidence_hard() both
              write the "Not Available" fingerprint into the prompt, for the
              TARGET and for every NEIGHBOR. The prompt's reasoning rules still
              refer to model_fingerprint_text on purpose: rewriting them would
              change the instruction as well as the evidence.

WHAT IS *NOT* ABLATED
=====================
scenario_description_text, the neighbour trajectories and everything else stay
intact. Run this and ablation_no_scenario_desc.py separately to attribute the
two text channels; running both at once needs a third script and answers a
different question.

OUTPUTS
=======
rag_core derives RUN_TAG's fp segment from the fingerprint filename whenever
IAM_LLM_FINGERPRINT_PATH is set, so this run tags itself `_fp-no_fingerprint`
automatically and cannot collide with a `_fp-papers-text_...` baseline. No
retagging needed here (unlike the desc ablation, where the tag would have
claimed `_desc-on`).

USAGE
=====
    python3 ablation_no_fingerprint.py --check       # free: assert the swap took
    TEST_LIMIT=20 python3 ablation_no_fingerprint.py    # smoke test
    python3 ablation_no_fingerprint.py                  # full batch + eval
    IAM_LLM_MAX_CONCURRENCY=16 python3 ablation_no_fingerprint.py

Every other rag_core env var still applies - set them exactly as for the
baseline run so the two differ in one thing only.

Extra knobs this wrapper adds:
    IAM_LLM_NOFP_PATH      point at a different blanked fingerprint workbook
    IAM_LLM_NOFP_SHEET     ... and its sheet name
    IAM_LLM_SKIP_EVAL=1    write the pred CSV, skip run_evaluation()
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

NOFP_PATH = os.environ.get("IAM_LLM_NOFP_PATH", str(_HERE / "no_fingerprint.xlsx"))
NOFP_SHEET = os.environ.get("IAM_LLM_NOFP_SHEET", "no_fingerprint")

if not Path(NOFP_PATH).exists():
    raise SystemExit(f"blanked fingerprint workbook not found: {NOFP_PATH}\n"
                     f"(override with IAM_LLM_NOFP_PATH)")

# Warn before we stomp on an explicit choice - silently overriding the thing the
# caller just exported is how an ablation ends up scoring the baseline.
_prev = os.environ.get("IAM_LLM_FINGERPRINT_PATH", "")
if _prev and Path(_prev).resolve() != Path(NOFP_PATH).resolve():
    print(f"[ABLATION] overriding inherited IAM_LLM_FINGERPRINT_PATH\n"
          f"           was: {_prev}\n           now: {NOFP_PATH}")

# MUST happen before `import rag_core`: both are read at module import time.
os.environ["IAM_LLM_FINGERPRINT_PATH"] = NOFP_PATH
os.environ["IAM_LLM_FP_SHEET"] = NOFP_SHEET

import rag_core as rc  # noqa: E402


# ----------------------------------------------------------------------------
# free self-check (no synthesis calls)
# ----------------------------------------------------------------------------
def check() -> int:
    """Assert the fingerprint swap took, without spending anything."""
    ok = True

    same_path = Path(rc.FINGERPRINT_PATH).resolve() == Path(NOFP_PATH).resolve()
    print(f"  [{'ok ' if same_path else 'FAIL'}] FINGERPRINT_PATH = "
          f"{Path(rc.FINGERPRINT_PATH).name}")
    print(f"  [{'ok ' if rc.FP_SHEET_NAME == NOFP_SHEET else 'FAIL'}] FP_SHEET_NAME  = "
          f"{rc.FP_SHEET_NAME!r}")
    ok &= same_path and rc.FP_SHEET_NAME == NOFP_SHEET

    texts = {fam: rc.get_model_family_fp_text(fam)
             for fam in sorted(rc.model_family2fp_text)}
    n_distinct = len(set(texts.values()))
    print(f"  [{'ok ' if n_distinct == 1 else 'FAIL'}] distinct fp_text across "
          f"{len(texts)} families: {n_distinct} (want 1)")
    ok &= (n_distinct == 1)

    sample = next(iter(texts.values())) if texts else ""
    has_na = "not available" in sample.lower()
    print(f"  [{'ok ' if has_na else 'FAIL'}] fp_text = {sample!r}")
    ok &= has_na

    # every family the case list actually uses must resolve, or the run dies
    # mid-batch after it has already been billed
    dfv = rc.load_validation_xlsx(rc.BATCH_SHEET_PATH)
    fams = sorted(dfv["model_family"].astype(str).str.strip().unique())
    missing = [f for f in fams if f not in rc.model_family2fp_text]
    print(f"  [{'ok ' if not missing else 'FAIL'}] all {len(fams)} families in "
          f"{Path(rc.BATCH_SHEET_PATH).name} covered"
          + (f" - MISSING {missing}" if missing else ""))
    ok &= not missing

    # the target block is what actually lands in the prompt
    row = dfv.iloc[0]
    tb = rc.build_target_block(row["model_family"], row["scenario"], row["region"])
    got = tb.get("model_fingerprint_text")
    hit = isinstance(got, str) and "not available" in got.lower()
    print(f"  [{'ok ' if hit else 'FAIL'}] build_target_block -> "
          f"model_fingerprint_text={got!r}")
    ok &= hit

    desc = tb.get("scenario_description_text") or ""
    kept = bool(desc.strip()) and desc.strip().lower() != "not available"
    print(f"  [{'ok ' if kept else 'WARN'}] scenario_description_text still real "
          f"({len(desc)} chars) - not ablated by this script")

    print(f"  [....] run tag will be: {rc.RUN_TAG}")
    print("\n[CHECK] PASS" if ok else "\n[CHECK] FAIL")
    return 0 if ok else 1


# ----------------------------------------------------------------------------
# batch run (mirrors rag_core's __main__ batch branch)
# ----------------------------------------------------------------------------
def main() -> int:
    if rc.MODE != "batch":
        raise SystemExit(f"rag_core.MODE is {rc.MODE!r}; this ablation only "
                         f"runs in batch mode.")

    rc.INCLUDE_PER_VAR_RATIONALE = False

    print(f"[ABLATION] fingerprints from {Path(rc.FINGERPRINT_PATH).name} "
          f"(sheet {rc.FP_SHEET_NAME!r}), {len(rc.model_family2fp_text)} families, "
          f"{len(set(rc.model_family2fp_text.values()))} distinct text(s)")
    if rc.RERANK_MODE != "off":
        print(f"[ABLATION] NOTE rerank_mode={rc.RERANK_MODE!r} is still ON: the "
              f"model reranker will see {rc.RERANK_POOL_MODEL} identical "
              f"strings and return an arbitrary order (still billed). The "
              f"scenario reranker is unaffected - it reads descriptions.")

    if "--check" in sys.argv:
        return check()

    rc.print_resolved_config()

    dfv = rc.load_validation_xlsx(rc.BATCH_SHEET_PATH)
    lim = int(os.environ.get("TEST_LIMIT", "0"))
    if lim > 0:
        dfv = dfv.head(lim)
        print(f"[NOTE] TEST_LIMIT={lim}: running only the first {lim} targets.")
    if rc.SAMPLE_N:
        dfv = rc.stratified_sample(dfv, rc.SAMPLE_N, rc.SAMPLE_SEED)

    # fail before spending anything if the list and the GT disagree
    rc.assert_batch_gt_consistent(dfv, rc.GT_PATH)

    t0 = time.time()
    out_df = asyncio.run(rc.run_batch(dfv))
    rc.RUN_STATS["elapsed_sec"] = round(time.time() - t0, 1)
    if len(dfv):
        rc.RUN_STATS["sec_per_case_throughput"] = round(
            rc.RUN_STATS["elapsed_sec"] / len(dfv), 4)
    if rc.CASE_SECS:
        import statistics as _st
        rc.RUN_STATS["sec_per_case_latency_measured"] = round(
            sum(rc.CASE_SECS) / len(rc.CASE_SECS), 2)
        rc.RUN_STATS["sec_per_case_latency_p50"] = round(_st.median(rc.CASE_SECS), 2)
        if len(rc.CASE_SECS) >= 20:
            rc.RUN_STATS["sec_per_case_latency_p95"] = round(
                _st.quantiles(rc.CASE_SECS, n=20)[18], 2)
        rc.RUN_STATS["sec_per_case_latency_max"] = round(max(rc.CASE_SECS), 2)
    rc.RUN_STATS.update(rc.USAGE)
    rc.RUN_STATS["est_cost_usd"] = round(rc.usage_cost(), 4)
    rc.RUN_STATS["est_cost_lost_max_usd"] = round(rc.lost_cost_upper_bound(), 4)

    # emit at the GT's 2-decimal precision, same as rag_core.py's batch
    # branch - otherwise this ablation's CSV is the only full-precision one.
    out_df = rc.round_year_cols(out_df)
    out_df.to_csv(rc.OUT_BATCH_CSV, index=False)
    print(f"Saved batch CSV ({rc.OUTPUT_DECIMALS} decimals):", rc.OUT_BATCH_CSV)

    if rc.SAVE_PROMPTS and rc.PROMPT_STORE:
        with open(rc.OUT_PROMPTS_JSONL, "w", encoding="utf-8") as pf:
            for i in sorted(rc.PROMPT_STORE):
                pf.write(json.dumps({"id": i, "prompt": rc.PROMPT_STORE[i]},
                                    ensure_ascii=False) + "\n")
        print(f"Saved prompts ({len(rc.PROMPT_STORE)} cases):", rc.OUT_PROMPTS_JSONL)

    u = rc.USAGE
    _c = rc.RUN_STATS["est_cost_usd"]; _n = len(dfv)
    print(f"\n[REPORT] {_n} cases @ concurrency {rc.MAX_CONCURRENCY}")
    print(f"  1. total cost            ${_c:,.2f}")
    print(f"  2. cost per case         ${_c / _n:.5f}")
    print(f"  3. total wall clock      {rc.RUN_STATS['elapsed_sec']:,.0f}s "
          f"({rc.RUN_STATS['elapsed_sec'] / 60:.1f} min)")
    print(f"  4. wall clock per case   "
          f"{rc.RUN_STATS.get('sec_per_case_throughput', float('nan')):.2f}s "
          f"(batch rate at concurrency {rc.MAX_CONCURRENCY}; not a per-case cost)")
    print(f"  5. sec per trajectory    "
          f"{rc.RUN_STATS.get('sec_per_case_latency_p50', float('nan')):.2f}s "
          f"(median, one case start to finish, concurrency-independent)")
    print(f"[DIAG] calls {u['calls']} | tokens in {u['tokens_in']:,} / "
          f"out {u['tokens_out']:,} / thinking {u['tokens_thinking']:,} | "
          f"latency mean {rc.RUN_STATS.get('sec_per_case_latency_measured', float('nan')):.2f} "
          f"p95 {rc.RUN_STATS.get('sec_per_case_latency_p95', float('nan')):.1f} "
          f"max {rc.RUN_STATS.get('sec_per_case_latency_max', float('nan')):.1f}")
    if u["failed_calls"]:
        print(f"[RUN STATS] failed calls {u['failed_calls']} "
              f"(503/rate {u['failed_503']} | disconnect {u['failed_disconnect']} "
              f"| other {u['failed_other']}) | lost input est "
              f"{u['tokens_lost_est']:,} tok | un-tracked billing "
              f"<= ${rc.RUN_STATS['est_cost_lost_max_usd']:.4f}")

    if rc.FAILED_CASES:
        print(f"\n[FAILED] {len(rc.FAILED_CASES)} of {len(dfv)} cases exhausted "
              f"RETRY_MAX and are MISSING from the CSV above:")
        for _cid, _cerr in rc.FAILED_CASES[:20]:
            print(f"    id={_cid}  {_cerr}")
        if len(rc.FAILED_CASES) > 20:
            print(f"    ... and {len(rc.FAILED_CASES) - 20} more")
        print("\nABORT: batch incomplete -> run_evaluation() SKIPPED "
              "(scoring it would shrink the denominator and break "
              "comparability with the baseline runs).")
        return 1

    if os.environ.get("IAM_LLM_SKIP_EVAL", "0") != "1":
        rc.run_evaluation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
