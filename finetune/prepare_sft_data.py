"""
prepare_sft_data.py
───────────────────
Build the finetuning datasets (JSONL) from the training, validation and test
sets in make_datasets/output/split, with the neighbour retrieval of rag_core.

Building the neighbour sets calls the LLM reranker of rag_core twice per case.
IAM_LLM_RERANK=0 uses the embedding-only retrieval instead. Importing rag_core
requires OPENAI_API_KEY.

Each sample:
  input  = the prompt of rag_core. build_user_prompt() calls
           rag_core.build_prompt() with the arguments that
           rag_core.synthesize_one_target() passes in batch mode, so the
           finetuned model and the zero-shot RAG baseline read the same target
           block, neighbour block, decision steps and output format. The
           retrieval excludes the target itself: the same-family step excludes
           the target scenario and the same-scenario step excludes the target
           family.
  output = the AR6 trajectories of the case as JSON (variables x 10 years),
           rounded to OUTPUT_DECIMALS (2) decimals.

Neighbour values are passed at the precision of the source files (2 decimals).
Each prompt, its neighbour block and its target cover the required variables
of the target IAM family.

Outputs -> finetune/data/ (each also as a *_nonb variant without neighbours):
  sft_train.jsonl      training cases, with targets
  sft_val.jsonl        validation cases, with targets (validation loss)
  sft_val_infer.jsonl  the same validation cases, prompt only (inference and scoring)
  sft_test.jsonl       test cases, prompt only
  sft_test_sub01..05.jsonl  the five test subsets, prompt only

Usage (from the project root):
    python3 finetune/prepare_sft_data.py
    # retrieval timing only (writes data/RETRIEVAL_TIMING_<split>_n<N>.json):
    SFT_TIME_ONLY=0 python3 finetune/prepare_sft_data.py
    #   SFT_TIME_ONLY=N = first N cases; SFT_TIME_SPLIT=test|val10|train70; SFT_TIME_CONC=100
"""

import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

import rag_core  # noqa: E402  (loads data + embedding caches; pool = train70)

OUT = _HERE / "data"
OUT.mkdir(parents=True, exist_ok=True)

YEARS = rag_core.YEARS

# Token estimation for the length report below. chars/4 is the usual English
# rule of thumb, but these prompts are mostly JSON punctuation and digits,
# which tokenize more densely. The default was calibrated against the measured
# Qwen token lengths of this dataset.
CHARS_PER_TOKEN = float(os.environ.get("SFT_CHARS_PER_TOKEN", "1.75"))
# Decimals kept in the assistant targets; see gt_answer(). Must match
# colab_infer_vllm.py's OUTPUT_DECIMALS so training and inference agree.
OUTPUT_DECIMALS = 2
# keep in sync with colab_train.py SFT_MAX_SEQ_LEN / colab_infer_vllm.py MAX_SEQ_LEN
SEQ_BUDGET = int(os.environ.get("SFT_MAX_SEQ_LEN", "32768"))
# keep in sync with colab_infer_vllm.py MAX_NEW_TOKENS. For prompt-only splits the
# prompt must leave room for the generated answer inside the same window, so
# those files are checked against SEQ_BUDGET - MAX_NEW_TOKENS.
MAX_NEW_TOKENS = int(os.environ.get("SFT_MAX_NEW_TOKENS", "4000"))
# Unbuildable cases abort the run by default. Set SFT_ALLOW_SKIP=1 only if you
# have read the skip list and accept a split that no longer covers every case.
ALLOW_SKIP = os.environ.get("SFT_ALLOW_SKIP", "0") == "1"

# rag_core sends exactly this as the system message on the OpenAI path
# (rag_core._run_once) and prepends the identical string on the Gemini path.
SYSTEM = "Return JSON only."


def build_user_prompt(target, include_neighbors=True):
    """The RAG prompt, verbatim. Same call rag_core.synthesize_one_target()
    makes in batch mode - do not paraphrase it here, or the FT-vs-RAG
    comparison silently becomes a comparison of two different prompts."""
    vars_t = rag_core.vars_for_family(target["model_family"])
    tb = rag_core.build_target_block(target["model_family"], target["scenario"],
                                     target["region"])

    # Same guard as rag_core.synthesize_one_target: a target with no scenario
    # description would silently become a null in the prompt. Raise instead -
    # make_split() catches it and reports the case.
    desc = tb.get("scenario_description_text")
    if rag_core.INCLUDE_SCENARIO_DESC and not str(desc or "").strip():
        raise KeyError(f"TARGET scenario '{target['scenario']}' has no "
                       f"scenario_description")

    if include_neighbors:
        steps = rag_core.compute_steps_for_target(target)
        nb_raw = rag_core.concat_neighbor_raw(
            steps["STEP1_DF"], steps["STEP2_DF"],
            steps["STEP3_DF"] if not steps["STEP3_DF"].empty else None)
        neighbors = rag_core.build_neighbors_all_evidence_hard(nb_raw, vars_t)
    else:
        # _nonb ablation: same prompt, empty evidence block
        neighbors = []

    return rag_core.build_prompt(
        target_block=tb,
        neighbors_all=neighbors,
        vars_out=vars_t,
        total_var=rag_core.TOTAL_ELECTRICITY_VAR,
        tech_vars=rag_core.TECH_ELECTRICITY_VARS,
        # batch/eval mode: rag_core.__main__ forces this False for the runs the
        # fine-tuned model is scored against
        include_per_var_rationale=False,
    )


def gt_answer(gt_block, vars_t):
    names = gt_block["variable"].astype(str).str.strip()
    in_scope = names[names.isin(vars_t)]
    # A dict comprehension would silently keep the LAST duplicate; rag_core
    # dedups keep="first". Rather than pick a side, refuse: a duplicated
    # (id, variable) row means the GT file is wrong and the target would be
    # one model's trajectory while the neighbour evidence is another's.
    if in_scope.duplicated().any():
        dups = sorted(in_scope[in_scope.duplicated()].unique())
        raise ValueError(f"GT block has duplicate rows for {len(dups)} "
                         f"variable(s): {dups[:3]}")
    lookup = {n: r for n, (_, r) in zip(names, gt_block.iterrows())}
    missing = [v for v in vars_t if v not in lookup]
    if missing:
        raise KeyError(f"GT block missing {len(missing)} variable(s): "
                       f"{missing[:3]}{'...' if len(missing) > 3 else ''}")
    trajs = []
    for v in vars_t:
        r = lookup[v]
        vals = {}
        for y in YEARS:
            try:
                fv = float(r[y])
            except (TypeError, ValueError):
                raise ValueError(f"GT variable '{v}' year {y} is not numeric: "
                                 f"{r[y]!r}")
            if not np.isfinite(fv):
                raise ValueError(f"GT variable '{v}' year {y} is not finite")
            vals[y] = round(fv, OUTPUT_DECIMALS)
        trajs.append({"variable": v, "values": vals})
    return json.dumps({"trajectories": trajs}, ensure_ascii=False)


def check_block_matches(block, target, cid):
    """The (prompt, answer) pair is joined on `id` alone. If the list and GT
    files were regenerated at different times the join still succeeds and every
    sample gets a plausible but WRONG target - the variable-name check above
    would not notice, because every core-5 case carries the same 22 variables.
    The GT files carry the keys, so verify them."""
    for col in ("region", "model_family", "scenario"):
        if col not in block.columns:
            continue
        got = set(block[col].astype(str).str.strip())
        if got != {target[col]}:
            raise ValueError(f"GT id={cid} has {col}={sorted(got)} but the case "
                             f"list says {target[col]!r} - list/GT out of sync, "
                             f"re-run make_datasets/split_r10_sd.py")


def load_gt(path):
    df = pd.read_excel(path)
    df.columns = [str(c) for c in df.columns]
    if "id" not in df.columns:
        raise KeyError(f"{path} has no 'id' column - re-run "
                       f"make_datasets/split_r10_sd.py")
    return df


def make_split(list_path, gt_path, out_path, with_target=True, limit=0,
               include_neighbors=True):
    lst = pd.read_excel(list_path)
    lst.columns = [str(c).strip() for c in lst.columns]
    need = ["id", "region", "model_family", "scenario"]
    miss = [c for c in need if c not in lst.columns]
    if miss:
        raise KeyError(f"{list_path} is missing required columns: {miss}")

    # only the *_target splits need the ground truth
    gt = load_gt(gt_path) if with_target else None

    if limit:
        # a smoke run (SFT_LIMIT) is written under its own file name, so that it
        # never overwrites the full dataset
        out_path = out_path.with_name(f"{out_path.stem}_smoke{out_path.suffix}")
        print(f"  [SFT_LIMIT={limit}] SMOKE RUN -> {out_path.name}")

    # write to a temp file and rename only on success, so an aborted run cannot
    # leave a short-but-valid JSONL behind for colab_train.py / colab_infer_vllm.py to read
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    n_done, skipped, lens = 0, [], []
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in lst.itertuples(index=False):
            cid = int(row.id)
            target = {"region": str(row.region).strip(),
                      "model_family": str(row.model_family).strip(),
                      "scenario": str(row.scenario).strip()}
            # the target JSON is built inside the same try as the prompt, so a
            # malformed GT block skips ONE case instead of killing the run
            try:
                user = build_user_prompt(target, include_neighbors)
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user}]
                if with_target:
                    block = gt[gt["id"] == cid]
                    if block.empty:
                        raise KeyError(f"no GT rows for id={cid}")
                    check_block_matches(block, target, cid)
                    msgs.append({"role": "assistant", "content": gt_answer(
                        block, rag_core.vars_for_family(target["model_family"]))})
            except Exception as e:
                skipped.append((cid, f"{type(e).__name__}: {e}"))
                continue
            rec = {"id": cid, **target, "messages": msgs}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            lens.append(sum(len(m["content"]) for m in msgs))
            n_done += 1
            if n_done % 250 == 0:
                print(f"  {out_path.name}: {n_done}")
            if limit and n_done >= limit:
                break

    print(f"{out_path.name}: {n_done} samples, {len(skipped)} skipped")

    # length report BEFORE any abort - it is most useful precisely when the
    # run is failing. A prompt-only split must also leave room for the answer.
    budget = SEQ_BUDGET if with_target else SEQ_BUDGET - MAX_NEW_TOKENS
    if lens:
        lens.sort()
        est = lambda c: c / CHARS_PER_TOKEN
        p50, p90, mx = (est(lens[len(lens) // 2]),
                        est(lens[int(0.9 * len(lens))]), est(lens[-1]))
        what = "prompt+answer" if with_target else f"prompt (+{MAX_NEW_TOKENS} to generate)"
        print(f"    length[{what}]: median ~{p50:,.0f} tok | p90 ~{p90:,.0f} | "
              f"max ~{mx:,.0f}  (est., budget {budget:,})")
        if mx > budget:
            print(f"    !! max ~{mx:,.0f} EXCEEDS the {budget:,} budget. Raise "
                  f"SFT_MAX_SEQ_LEN in colab_train.py AND colab_infer_vllm.py, or "
                  f"the longest cases get truncated.")

    if skipped:
        for cid, why in skipped:
            print(f"    SKIPPED id={cid}: {why}")
        # Collect every problem first (so one run tells you about all of them),
        # then refuse to leave a short file on disk. A silently-shortened test
        # or val split is the dangerous case: colab_infer_vllm.py would predict fewer
        # ids and rag_core.run_evaluation would just print "[NOTE] Partial run"
        # and score the fine-tuned model on a smaller set than the RAG
        # baseline. rag_core itself hard-fails on an unusable target (run_batch
        # uses asyncio.gather without return_exceptions), so this matches it.
        if not ALLOW_SKIP:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"{out_path.name}: {len(skipped)} case(s) could not be built "
                f"(listed above); nothing written. Fix the data, or set "
                f"SFT_ALLOW_SKIP=1 to write the short file anyway - but then "
                f"the eval set no longer matches the RAG baseline's."
            )

    os.replace(tmp_path, out_path)
    return n_done, len(skipped)


def time_retrieval(list_path, n_cases, out_json, workers=100):
    """Measure the retrieval side of the fine-tuned pipeline - the part
    prepare_sft_data does offline and bakes into the prompts, which the vLLM
    run_info therefore does NOT contain. Writes nothing to the sft files.

    ONE concurrent pass (thread pool), instrumented the way rag_core's
    run_batch is: wall clock over the whole batch, and each case's own time
    from the moment a worker picks it up to the moment its prompt is built.
    Reports the same five rpt_* numbers rag_core's eval report carries:
      rpt_total_cost_usd      tokens actually spent (rag_core.USAGE delta,
                              priced by rag_core.PRICE_IN/OUT)
      rpt_cost_per_case_usd   cost / cases
      rpt_total_wallclock_sec batch wall clock
      rpt_sec_per_case_batch  wall clock / cases (throughput; depends on workers)
      rpt_sec_per_trajectory  p50 of the measured per-case latency
                              (concurrency-independent; rag_core's definition)
    Embedding caches are warm, as for every RAG run; the time and cost are the
    two rerank LLM calls per case.
    """
    import time, statistics
    from concurrent.futures import ThreadPoolExecutor
    lst = pd.read_excel(list_path)
    lst.columns = [str(c).strip() for c in lst.columns]
    rows = list(lst.itertuples(index=False))
    if n_cases:
        rows = rows[:n_cases]
    targets = [{"region": str(r.region).strip(), "model_family": str(r.model_family).strip(),
                "scenario": str(r.scenario).strip()} for r in rows]

    def one(t):
        t0 = time.perf_counter()
        build_user_prompt(t, include_neighbors=True)
        return time.perf_counter() - t0

    usage0 = dict(rag_core.USAGE)
    cost0 = rag_core.usage_cost()
    conc = min(int(workers), len(targets))
    print(f"[timing] {len(targets)} cases, {conc} workers ...")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        secs = list(ex.map(one, targets))
    wall = time.perf_counter() - t0
    cost = rag_core.usage_cost() - cost0
    n = len(targets)

    q = statistics.quantiles(secs, n=20) if n >= 20 else None
    info = {
        "what": "retrieval only (STEP1/STEP2 embedding ranking + LLM rerank + prompt build); "
                "no generation. Add the vLLM run_info generate_per_case for end-to-end.",
        "list": str(list_path), "n_cases": n, "workers": conc,
        "topk": f"{rag_core.TOPK_SCEN}{rag_core.TOPK_MODEL}0",
        "rerank": bool(getattr(rag_core, "RERANK", False)),
        "rpt_total_cost_usd": round(cost, 4),
        "rpt_cost_per_case_usd": round(cost / n, 6),
        "rpt_total_wallclock_sec": round(wall, 1),
        "rpt_sec_per_case_batch": round(wall / n, 4),
        "rpt_sec_per_trajectory": round(statistics.median(secs), 3),
        "diag_latency": {"mean": round(statistics.mean(secs), 3),
                         "p95": round(q[18], 3) if q else None,
                         "max": round(max(secs), 3)},
        "diag_llm_calls": rag_core.USAGE["calls"] - usage0["calls"],
        "diag_tokens_in": rag_core.USAGE["tokens_in"] - usage0["tokens_in"],
        "diag_tokens_out": rag_core.USAGE["tokens_out"] - usage0["tokens_out"],
        "price_per_mtok": {"in": rag_core.PRICE_IN, "out": rag_core.PRICE_OUT},
        "measured": datetime.datetime.now().isoformat(timespec="minutes"),
    }
    json.dump(info, open(out_json, "w"), indent=2)
    print(json.dumps(info, indent=2))
    print("[timing] ->", out_json)


if __name__ == "__main__":
    # ── timing-only mode: SFT_TIME_ONLY=0 = the whole list; N = first N (smoke) ──
    if os.environ.get("SFT_TIME_ONLY") is not None:
        _n = int(os.environ["SFT_TIME_ONLY"])
        _split = os.environ.get("SFT_TIME_SPLIT", "test")      # test | val10 | train70
        _conc = int(os.environ.get("SFT_TIME_CONC", "100"))
        _lists = {"val10": "AR6_R10sd_val10_list.xlsx", "test": "AR6_R10sd_test_list.xlsx",
                  "train70": "AR6_R10sd_train70_list.xlsx"}
        _lp = _ROOT / "make_datasets" / "output" / "split" / _lists[_split]
        time_retrieval(_lp, _n, OUT / f"RETRIEVAL_TIMING_{_split}_n{_n or 'all'}_c{_conc}.json",
                       workers=_conc)
        raise SystemExit(0)

    if getattr(rag_core, "RERANK", False):
        print(f"[NOTE] rag_core LLM rerank is ON (pool "
              f"{rag_core.RERANK_POOL_SCEN}/{rag_core.RERANK_POOL_MODEL}). Neighbour "
              f"selection now costs 2 LLM calls per case, and the neighbours baked "
              f"into these files are the ones THIS run selected - a later RAG run "
              f"re-ranks independently and may land on a slightly different set, so "
              f"the SFT and RAG prompts are same-distribution rather than "
              f"byte-identical. Set IAM_LLM_RERANK=0 to restore the deterministic "
              f"embedding-only retrieval.")
    limit = int(os.environ.get("SFT_LIMIT", "0"))   # smoke-test with e.g. SFT_LIMIT=20
    D = _ROOT / "make_datasets" / "output" / "split"

    # test splits: full test set ("") + the 5 sub splits
    TEST_TAGS = [""] + [f"_sub{i:02d}" for i in range(1, 6)]

    # sfx "" = with neighbors; "_nonb" = no-neighbor ablation (pure parametric emulator)
    for include_nb, sfx in ((True, ""), (False, "_nonb")):
        make_split(D / "AR6_R10sd_train70_list.xlsx",
                   D / "AR6_R10sd_train70_ground_truth.xlsx",
                   OUT / f"sft_train{sfx}.jsonl", limit=limit,
                   include_neighbors=include_nb)
        # val10: with assistant targets (eval loss / checkpoint selection)
        make_split(D / "AR6_R10sd_val10_list.xlsx",
                   D / "AR6_R10sd_val10_gt.xlsx",
                   OUT / f"sft_val{sfx}.jsonl", limit=limit,
                   include_neighbors=include_nb)
        # val10: prompt-only (run inference, then score against gt)
        make_split(D / "AR6_R10sd_val10_list.xlsx",
                   D / "AR6_R10sd_val10_gt.xlsx",
                   OUT / f"sft_val_infer{sfx}.jsonl", with_target=False, limit=limit,
                   include_neighbors=include_nb)
        # test: prompt-only, full set + all 5 subs (N_SUB in split_r10_sd.py)
        for tag in TEST_TAGS:
            make_split(D / f"AR6_R10sd{tag}_test_list.xlsx",
                       D / f"AR6_R10sd{tag}_test_gt.xlsx",
                       OUT / f"sft_test{tag}{sfx}.jsonl", with_target=False,
                       limit=limit, include_neighbors=include_nb)
    # provenance: which retrieval budget / split these files were built from.
    # The notebooks' EXPECT_SPLIT guard and the "does the SFT prompt match the
    # RAG baseline's topk" question both need this, and nothing else records it.
    import datetime
    _n = {k: sum(1 for _ in open(OUT / f"sft_{k}.jsonl"))
          for k in ("train", "val", "val_infer", "test")}
    info = {"built": datetime.datetime.now().isoformat(timespec="minutes"),
            # same spelling as rag_core's run tag: the 3rd digit is 0 unless STEP3 runs
            "topk": (f"{rag_core.TOPK_SCEN}{rag_core.TOPK_MODEL}"
                     f"{rag_core.TOPK_CROSS if getattr(rag_core, 'RUN_STEP3', False) else 0}"),
            "topk_scen": rag_core.TOPK_SCEN, "topk_model": rag_core.TOPK_MODEL,
            "topk_cross": rag_core.TOPK_CROSS,
            "rerank": bool(getattr(rag_core, "RERANK", False)),
            "fingerprint_sheet": getattr(rag_core, "FP_SHEET_NAME", None),
            "split_dir": str(D), "n_cases": _n, "limit": limit}
    (OUT / "BUILD_INFO.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print("build info ->", OUT / "BUILD_INFO.json", info)
    print("done ->", OUT)
