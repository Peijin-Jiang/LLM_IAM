#!/usr/bin/env python3
"""
ablation_no_scenario_desc.py - "no scenario description" ablation for rag_core.

WHAT THIS IS
============
A thin wrapper, NOT a fork. It imports rag_core, replaces every scenario
description with the literal string "Not Available", then runs the same batch +
evaluation that rag_core's own __main__ block runs. Nothing else changes, and
because rag_core is imported rather than copied, this script follows any
change of rag_core.

FULL-CHAIN ABLATION
===================
The blanking happens in `scenario2desc`, the single table every description
consumer reads at call time, so the text is gone from BOTH paths:

  RETRIEVAL   step1_rank_scenarios_region_family() ranks candidate scenarios by
              cosine similarity of description embeddings. With one distinct
              description left, every similarity ties at 1.0, np.argsort is
              stable, and step1 degenerates to the candidate order the source
              dataframe happens to have - deterministic, but carrying no signal.

              The LLM SCENARIO reranker still runs and still costs one call per
              target. It is now shown RERANK_POOL_SCEN identical "Not Available"
              strings and returns an arbitrary permutation. That is the honest
              degradation: the pipeline is unchanged, only the evidence is gone.
              Set IAM_LLM_RERANK_MODE=off for the cheaper, lower-variance
              variant - but that is then a DIFFERENT pipeline from the baseline
              and the comparison stops isolating a single factor.

              The MODEL reranker ranks by fingerprint text, not by scenario
              description, so it is untouched. Same for step2's family pool.

  SYNTHESIS   build_target_block() and build_neighbors_all_evidence_hard() both
              write "Not Available" into the prompt - for the TARGET and for
              every NEIGHBOR.

              The prompt's reasoning rules still refer to
              scenario_description_text. That is deliberate: rewriting them
              would change the INSTRUCTION as well as the EVIDENCE, which is a
              second variable. The model is told to use a field whose value is
              "Not Available", exactly as it would be if the dataset shipped
              without descriptions.

WHAT IS *NOT* ABLATED
=====================
model_fingerprint_text and the neighbour trajectories are untouched. This
ablation answers "how much does the scenario description text contribute?",
not "how much does text contribute?".

OUTPUTS
=======
RUN_TAG's `_desc-on` segment becomes `_desc-NA`, so the pred CSV, the prompt
sidecar and the eval XLSX cannot collide with a baseline run in outputs/.

USAGE
=====
    python3 ablation_no_scenario_desc.py --check      # free: assert the blanking took
    TEST_LIMIT=20 python3 ablation_no_scenario_desc.py   # ~20-target smoke test
    python3 ablation_no_scenario_desc.py                 # full batch + eval
    IAM_LLM_MAX_CONCURRENCY=16 python3 ablation_no_scenario_desc.py

Every rag_core env var (IAM_LLM_BATCH_SHEET_PATH, IAM_LLM_GT_PATH, the topk
knobs, IAM_LLM_MAX_CONCURRENCY, ...) still applies. Set them exactly as you did
for the baseline run so the two differ in one thing only.

Extra knobs this wrapper adds:
    IAM_LLM_DESC_NA_TEXT   placeholder string        (default "Not Available")
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

NA_TEXT = os.environ.get("IAM_LLM_DESC_NA_TEXT", "Not Available").strip()
if not NA_TEXT:
    raise SystemExit("IAM_LLM_DESC_NA_TEXT must not be empty - an empty "
                     "description makes get_scenario_desc() return None, which "
                     "rag_core treats as 'missing' and hard-fails on, instead "
                     "of as 'unavailable'.")

# Importing rag_core runs its module-level setup (data load + embedding lookup).
# The description embeddings it builds there are thrown away below; they come
# from the on-disk cache, so this costs nothing beyond the read.
import rag_core as rc  # noqa: E402


# ----------------------------------------------------------------------------
# the ablation itself
# ----------------------------------------------------------------------------
def blank_all_scenario_descriptions(na: str = NA_TEXT) -> None:
    """Replace every scenario description with `na`, everywhere."""
    n_scen = len(rc.scenario2desc)
    n_distinct = len({str(v).strip() for v in rc.scenario2desc.values()
                      if str(v).strip()})

    # 1) the lookup that get_scenario_desc() (retrieval side) and
    #    get_scenario_desc_text() (prompt side) both read at call time.
    rc.scenario2desc = {k: na for k in rc.scenario2desc}

    # 2) the frames, so no description can reach a prompt by some other route.
    for _name in ("df", "df_gt", "df_desc_global"):
        _f = getattr(rc, _name, None)
        if _f is not None and getattr(rc, "col_desc", None) in getattr(_f, "columns", []):
            _f[rc.col_desc] = na

    # 3) the embedding table step1 indexes by description STRING. One text ->
    #    one vector -> every candidate similarity ties.
    cache = rc.load_json(rc.SCEN_EMB_CACHE)
    embs = rc.embed_texts([na], cache, prefix="SCEN_DESC")
    rc.save_json(cache, rc.SCEN_EMB_CACHE)
    rc.all_desc = [na]
    rc.desc_embs = embs
    rc.desc2emb = {na: embs[0]}

    print(f"[ABLATION] scenario descriptions: {n_distinct} distinct -> 1 "
          f"({na!r}), across {n_scen} scenarios")
    print(f"[ABLATION] desc2emb: {len(rc.desc2emb)} vector(s) -> step1 "
          f"similarity ties for every candidate")
    if rc.RERANK_MODE != "off":
        print(f"[ABLATION] NOTE rerank_mode={rc.RERANK_MODE!r} is still ON: the "
              f"scenario reranker will see {rc.RERANK_POOL_SCEN} identical "
              f"strings and return an arbitrary order (still billed). The "
              f"model reranker is unaffected - it reads fingerprints.")


def retag_outputs(marker: str = "NA") -> None:
    """Move this run's output filenames off the baseline's names."""
    tag = rc.RUN_TAG
    for old in ("_desc-on", "_desc-off"):
        tag = tag.replace(old, f"_desc-{marker}")
    if f"_desc-{marker}" not in tag:
        tag = f"{tag}_desc-{marker}"
    rc.RUN_TAG = tag

    out_dir = getattr(rc, "OUT_DIR", rc.OUTPUTS_DIR)
    rc.OUT_BATCH_CSV = os.path.join(
        rc.OUTPUTS_DIR, f"{rc.DATASET_TAG}_pred_{rc.RUN_TAG}.csv")
    rc.OUT_PROMPTS_JSONL = os.path.join(
        rc.OUTPUTS_DIR, f"{rc.DATASET_TAG}_prompts_{rc.RUN_TAG}.jsonl")
    rc.OUT_XLSX = os.path.join(
        out_dir, f"{rc.DATASET_TAG}_eval_{rc.RUN_TAG}.xlsx")

    # Follow the new CSV only if the caller did not pin one explicitly.
    if not os.environ.get("IAM_LLM_PRED_PATH", "").strip():
        rc.PRED_PATH = rc.OUT_BATCH_CSV

    print(f"[ABLATION] run tag : {rc.RUN_TAG}")
    print(f"[ABLATION] pred CSV: {Path(rc.OUT_BATCH_CSV).name}")
    print(f"[ABLATION] eval    : {Path(rc.OUT_XLSX).name}")


# ----------------------------------------------------------------------------
# free self-check (no API calls)
# ----------------------------------------------------------------------------
def check() -> int:
    """Assert the blanking actually took, without spending anything."""
    sample = list(rc.scenario2desc)[:5]
    ok = True

    for s in sample:
        a, b = rc.get_scenario_desc(s), rc.get_scenario_desc_text(s)
        flag = "ok " if a == b == NA_TEXT else "FAIL"
        ok &= (a == b == NA_TEXT)
        print(f"  [{flag}] {s[:44]:44s} retrieval={a!r} prompt={b!r}")

    n = len({str(v).strip() for v in rc.scenario2desc.values()})
    print(f"  [{'ok ' if n == 1 else 'FAIL'}] distinct descriptions remaining: {n} (want 1)")
    ok &= (n == 1)

    print(f"  [{'ok ' if len(rc.desc2emb) == 1 else 'FAIL'}] desc2emb entries: "
          f"{len(rc.desc2emb)} (want 1)")
    ok &= (len(rc.desc2emb) == 1)

    # the target block is what actually lands in the prompt
    row = rc.load_validation_xlsx(rc.BATCH_SHEET_PATH).iloc[0]
    tb = rc.build_target_block(row["model_family"], row["scenario"], row["region"])
    got = tb.get("scenario_description_text")
    print(f"  [{'ok ' if got == NA_TEXT else 'FAIL'}] build_target_block -> "
          f"scenario_description_text={got!r}")
    ok &= (got == NA_TEXT)
    fp = (tb.get("model_fingerprint_text") or "")
    print(f"  [ok ] model_fingerprint_text still present ({len(fp)} chars) - not ablated")

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
    blank_all_scenario_descriptions()

    if "--check" in sys.argv:
        return check()

    retag_outputs()
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
    rc.RUN_STATS.update(rc.USAGE)
    rc.RUN_STATS["est_cost_usd"] = round(rc.usage_cost(), 4)
    rc.RUN_STATS["est_cost_lost_max_usd"] = round(rc.lost_cost_upper_bound(), 4)

    out_df.to_csv(rc.OUT_BATCH_CSV, index=False)
    print("Saved batch CSV:", rc.OUT_BATCH_CSV)

    if rc.SAVE_PROMPTS and rc.PROMPT_STORE:
        with open(rc.OUT_PROMPTS_JSONL, "w", encoding="utf-8") as pf:
            for i in sorted(rc.PROMPT_STORE):
                pf.write(json.dumps({"id": i, "prompt": rc.PROMPT_STORE[i]},
                                    ensure_ascii=False) + "\n")
        print(f"Saved prompts ({len(rc.PROMPT_STORE)} cases):", rc.OUT_PROMPTS_JSONL)

    u = rc.USAGE
    print(f"\n[RUN STATS] elapsed {rc.RUN_STATS['elapsed_sec']}s | "
          f"calls {u['calls']} | tokens in {u['tokens_in']:,} / "
          f"out {u['tokens_out']:,} / thinking {u['tokens_thinking']:,} | "
          f"est cost ${rc.RUN_STATS['est_cost_usd']:.4f}")
    if u["failed_calls"]:
        print(f"[RUN STATS] failed calls {u['failed_calls']} "
              f"(503/rate {u['failed_503']} | disconnect {u['failed_disconnect']} "
              f"| other {u['failed_other']}) | lost input est "
              f"{u['tokens_lost_est']:,} tok | un-tracked billing "
              f"<= ${rc.RUN_STATS['est_cost_lost_max_usd']:.4f}")

    if os.environ.get("IAM_LLM_SKIP_EVAL", "0") != "1":
        rc.run_evaluation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
