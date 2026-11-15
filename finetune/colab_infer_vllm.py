"""
colab_infer_vllm.py - fine-tuned-adapter inference with vLLM (continuous
batching + paged KV + built-in JSON-schema structured output). The output CSV
has the format that score_preds.py and rag_core.run_evaluation() read.

Why merge instead of vLLM's LoRA mode: vLLM's Qwen3.5 LoRA path only maps the
fused module names (qkv_proj / gate_up_proj) and silently ignores gate_proj /
up_proj from a PEFT adapter (vllm issue #38085). Merging the adapter into the
bf16 weights avoids this, and the
merged model is what vLLM serves as a plain HF checkpoint.

Run in a SEPARATE Colab runtime from training (vLLM pins its own torch /
transformers; do not mix with the unsloth training pins):
    !pip install -q vllm peft
    from google.colab import drive; drive.mount('/content/drive')
    %env SFT_BASE_MODEL=Qwen/Qwen3.5-9B
    %env SFT_LR=5e-4                         # -> TAG -> adapters/<TAG>
    %env SFT_SPLIT=val_infer                 # val_infer | test | test_sub01..05
    !python colab_infer_vllm.py              # a real subprocess, NOT %run (fileno)
    # another checkpoint of the same run:
    %env SFT_ADAPTER=/content/drive/MyDrive/finetune/checkpoints/<TAG>/checkpoint-600

Env vars:
    SFT_BASE_MODEL / SFT_LR / SFT_LORA_R / SFT_SEED / SFT_SFX / SFT_TAG / SFT_LOAD_4BIT
    SFT_ADAPTER      adapter dir (default <DRIVE_ROOT>/adapters/<TAG>)
    SFT_SPLIT        which sft_<split><SFX>.jsonl (default val_infer)
    SFT_LIMIT        0 = all cases (REPORT); N = probe on the first N
    SFT_CONSTRAIN    1 = JSON-schema structured output (REPORT); 0 = free decoding
    SFT_MERGED_DIR   where the merged bf16 model goes (default /content/merged/<TAG>_<ckpt>,
                     local disk; ~16-18GB, rebuilt per checkpoint, never on Drive)
    SFT_GPU_UTIL     vLLM gpu_memory_utilization (default 0.90)
    SFT_MAX_SEQ_LEN  vLLM max_model_len (default 20480; prompts are up to 15.2k tokens, plus the 4k answer)
"""

import os, json, re, time, glob, shutil
from datetime import datetime as _dt

DATA_DIR = os.environ.get("SFT_DATA_DIR", "/content/drive/MyDrive/finetune/data")
DRIVE_ROOT = os.environ.get("SFT_DRIVE_ROOT", "/content/drive/MyDrive/finetune")
SFX = os.environ.get("SFT_SFX", "")
BASE_MODEL = os.environ.get("SFT_BASE_MODEL", "Qwen/Qwen3.5-9B")
LOAD_4BIT = os.environ.get("SFT_LOAD_4BIT", "0") == "1"      # only feeds the TAG
LR = float(os.environ.get("SFT_LR", "5e-4"))
LORA_R = int(os.environ.get("SFT_LORA_R", "16"))
TRAIN_SEED = int(os.environ.get("SFT_SEED", "42"))
BASE_SHORT = BASE_MODEL.split("/")[-1].replace("-unsloth-bnb-4bit", "").replace("-bnb-4bit", "").lower()
TAG = os.environ.get("SFT_TAG", f"{BASE_SHORT}_lr{LR}_r{LORA_R}{SFX}" + ("_4bit" if LOAD_4BIT else "_bf16")
                     + (f"_seed{TRAIN_SEED}" if TRAIN_SEED != 42 else ""))
ADAPTER = os.environ.get("SFT_ADAPTER") or f"{DRIVE_ROOT}/adapters/{TAG}"
SPLIT = os.environ.get("SFT_SPLIT", "val_infer")
DATA_FILE = f"{DATA_DIR}/sft_{SPLIT}{SFX}.jsonl"
OUT_DIR = os.environ.get("SFT_OUT_DIR") or f"{DRIVE_ROOT}/preds"
os.makedirs(OUT_DIR, exist_ok=True)
# 20480 = longest prompt of the SFT data (15.2k tokens) + 4k answer, with margin.
# vLLM must fit ONE sequence of this length in its KV pool at start-up, so a smaller
# value = less reserved memory = more concurrent sequences (and it fits a 40GB A100).
MAX_SEQ_LEN = int(os.environ.get("SFT_MAX_SEQ_LEN", "20480"))
MAX_NEW_TOKENS = int(os.environ.get("SFT_MAX_NEW_TOKENS", "4000"))
LIMIT = int(os.environ.get("SFT_LIMIT", "0"))
CONSTRAIN = os.environ.get("SFT_CONSTRAIN", "1") == "1"
GPU_UTIL = float(os.environ.get("SFT_GPU_UTIL", "0.90"))
OUTPUT_DECIMALS = int(os.environ.get("IAM_LLM_OUTPUT_DECIMALS", "2"))

_ckpt = os.path.basename(ADAPTER.rstrip("/"))
if _ckpt == TAG:
    _ckpt = "best"
MERGED_DIR = os.environ.get("SFT_MERGED_DIR") or f"/content/merged/{TAG}_{_ckpt}"
_probe = f"_n{LIMIT}" if LIMIT else ""
_flags = ("" if CONSTRAIN else "_free") + _probe
# "_vllm" marks the inference backend so a
# vLLM CSV and an HF-generate CSV can never be confused in a results table
OUT_CSV = (f"{OUT_DIR}/ft_{SPLIT}_{TAG}_{_ckpt}{_flags}_vllm_"
           f"{_dt.now().strftime('%Y%m%d_%H%M')}.csv")
assert os.path.exists(DATA_FILE), DATA_FILE
assert os.path.exists(f"{ADAPTER}/adapter_config.json"), f"no adapter at {ADAPTER}"
print("adapter :", ADAPTER, "\nmerged  :", MERGED_DIR, "\nout     :", os.path.basename(OUT_CSV))

# ============================================================
# 1) merge LoRA -> bf16 weights (generic PEFT; cached per checkpoint)
# ============================================================
import torch
from transformers import AutoConfig, AutoTokenizer

def _loaders():
    """Candidate base classes, tried in order. The adapter's module paths must
    match the class it was trained on: text-only causal LM for Qwen3, and for
    Qwen3.5 (a unified VLM checkpoint) possibly the image-text class. A wrong
    class does not crash PEFT - the LoRA-B weights just stay zero - so each
    candidate is VERIFIED by checking that merging actually changed a weight."""
    from transformers import AutoModelForCausalLM
    yield "AutoModelForCausalLM", AutoModelForCausalLM
    try:
        from transformers import AutoModelForImageTextToText
        yield "AutoModelForImageTextToText", AutoModelForImageTextToText
    except ImportError:
        pass

def _try_merge(cls_name, cls):
    from peft import PeftModel
    try:
        # merge on the GPU: the bf16 weights plus the temporary copy of the merge
        # can exceed the CPU RAM of a standard Colab runtime
        base = cls.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                   low_cpu_mem_usage=True, device_map={"": 0})
    except Exception as e:
        print(f"[merge] {cls_name}: cannot load base ({type(e).__name__}: {str(e)[:120]})")
        return None
    probe_name = next(n for n, _ in base.named_parameters() if n.endswith("q_proj.weight"))
    before = dict(base.named_parameters())[probe_name].detach().clone()
    model = PeftModel.from_pretrained(base, ADAPTER)
    n_lora = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
    model = model.merge_and_unload()
    delta = (dict(model.named_parameters())[probe_name].detach().float() - before.float()).abs().max().item()
    print(f"[merge] {cls_name}: {n_lora} LoRA tensors, max |delta| in {probe_name} = {delta:.3e}")
    if n_lora == 0 or delta == 0.0:
        del model, base; torch.cuda.empty_cache()
        return None                    # adapter did not land on this class's modules
    return model, n_lora

if not os.path.exists(f"{MERGED_DIR}/config.json"):
    t0 = time.time()
    merged = None
    for cls_name, cls in _loaders():
        merged = _try_merge(cls_name, cls)
        if merged:
            break
    assert merged, ("adapter could not be applied to any base class - module paths in "
                    f"{ADAPTER} match neither AutoModelForCausalLM nor AutoModelForImageTextToText")
    model, n_lora = merged
    merged = None   # drop the extra reference - with it alive, `del model` below cannot
                    # release the ~16-18GB of merged weights, and vLLM then fails its
                    # free-memory check
    os.makedirs(MERGED_DIR, exist_ok=True)
    model.save_pretrained(MERGED_DIR, safe_serialization=True)
    try:                               # tokenizer/processor: what training saved, else base
        AutoTokenizer.from_pretrained(ADAPTER).save_pretrained(MERGED_DIR)
    except Exception:
        AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(MERGED_DIR)
    json.dump({"base_model": BASE_MODEL, "adapter": ADAPTER, "tag": TAG, "ckpt": _ckpt,
               "merged_at": _dt.now().isoformat(timespec="minutes"), "n_lora_tensors": n_lora},
              open(f"{MERGED_DIR}/MERGE_INFO.json", "w"), indent=2)
    del model
    import gc; gc.collect(); torch.cuda.empty_cache()   # hand the whole GPU to vLLM
    _merge_sec = time.time() - t0
    print(f"[merge] saved to {MERGED_DIR} in {_merge_sec / 60:.1f} min")
else:
    _merge_sec = 0.0        # cached from an earlier call in this runtime
    print("[merge] reusing", MERGED_DIR, json.load(open(f"{MERGED_DIR}/MERGE_INFO.json")))

# Qwen3.5 is a VLM checkpoint: vLLM requires the image and video processor config
# files at engine start, so the config files of the base repository are copied
# into the merged model (its own config, generation config and weights are kept)
try:
    from huggingface_hub import snapshot_download
    _src = snapshot_download(BASE_MODEL,
                             allow_patterns=["*.json", "*.jinja", "*.txt", "vocab*", "merges.txt"])
    _copied = []
    for _f in glob.glob(_src + "/*"):
        _b = os.path.basename(_f)
        if _b in ("config.json", "generation_config.json") or _b.startswith("model.safetensors"):
            continue
        if os.path.isfile(_f):
            shutil.copy(_f, os.path.join(MERGED_DIR, _b))
            _copied.append(_b)
    print("[merge] base-repo config files ensured:", sorted(_copied))
except Exception as _pe:
    print(f"[merge] config-file top-up skipped ({type(_pe).__name__}: {_pe})")

# ============================================================
# 2) schema per family
# ============================================================
YEARS = [str(y) for y in range(2010, 2110, 10)]
VARS_22 = [
    "Primary Energy|Coal|w/ CCS", "Primary Energy|Coal|w/o CCS",
    "Primary Energy|Gas|w/ CCS", "Primary Energy|Gas|w/o CCS",
    "Primary Energy|Oil|w/ CCS", "Primary Energy|Oil|w/o CCS",
    "Primary Energy|Solar", "Primary Energy|Wind", "Primary Energy|Hydro",
    "Primary Energy|Nuclear", "Primary Energy|Biomass",
    "Final Energy|Liquids", "Final Energy|Gases",
    "Final Energy|Industry", "Final Energy|Transportation",
    "Final Energy|Residential and Commercial", "Final Energy|Hydrogen",
    "Final Energy|Heat", "Final Energy|Solids",
    "Final Energy|Industry|Electricity", "Final Energy|Transportation|Electricity",
    "Final Energy|Residential and Commercial|Electricity",
]
SUPPLY_11, DEMAND_11 = VARS_22[:11], VARS_22[11:]
SUPPLY_ONLY_FAMILIES = ["COFFEE", "WITCH"]
DEMAND_ONLY_FAMILIES = ["GCAM", "TIAM"]

def vars_for_family(fam):
    fam = str(fam).strip()
    if fam in SUPPLY_ONLY_FAMILIES: return SUPPLY_11
    if fam in DEMAND_ONLY_FAMILIES: return DEMAND_11
    return VARS_22

def scope_key(fam):
    fam = str(fam).strip()
    return "sup" if fam in SUPPLY_ONLY_FAMILIES else "dem" if fam in DEMAND_ONLY_FAMILIES else "core"

def build_schema(vars_t):
    return {"type": "object",
            "properties": {"trajectories": {
                "type": "array", "minItems": len(vars_t), "maxItems": len(vars_t),
                "items": {"type": "object",
                          "properties": {"variable": {"type": "string", "enum": vars_t},
                                         "unit": {"type": "string"},
                                         "values": {"type": "object",
                                                    "properties": {y: {"type": "number"} for y in YEARS},
                                                    "required": YEARS}},
                          "required": ["variable", "values"]}}},
            "required": ["trajectories"]}

SCHEMAS = {k: build_schema(v) for k, v in (("core", VARS_22), ("sup", SUPPLY_11), ("dem", DEMAND_11))}

def extract_json(s):
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", (s or "").strip())
    i, j = s.find("{"), s.rfind("}")
    return json.loads(re.sub(r",\s*([}\]])", r"\1", s[i:j + 1]))

fmt = {"parse_fail": 0, "scope_mismatch": 0}

def rows_for(rec, text):
    out_rows = []
    meta = {"id": rec["id"], "region": rec["region"],
            "model_family": rec["model_family"], "scenario": rec["scenario"]}
    try:
        obj = extract_json(text)
        v2t = {str(t["variable"]).strip(): t["values"] for t in obj["trajectories"]}
        expected = vars_for_family(rec["model_family"])
        extra = [v for v in v2t if v not in expected]
        missing = [v for v in expected if v not in v2t]
        if extra or missing:
            fmt["scope_mismatch"] += 1
            print(f"case {rec['id']}: scope mismatch (+{len(extra)}, -{len(missing)})")
        for var in expected:
            vals = v2t.get(var, {})
            row = dict(meta, variable=var, unit="EJ/yr")
            for y in YEARS:
                try:
                    row[y] = round(max(float(vals.get(y, 0.0)), 0.0), OUTPUT_DECIMALS)
                except (TypeError, ValueError):
                    row[y] = 0.0
            out_rows.append(row)
    except Exception as e:
        fmt["parse_fail"] += 1
        print(f"case {rec['id']}: parse failed ({e}) - filling zeros")
        for var in vars_for_family(rec["model_family"]):
            row = dict(meta, variable=var, unit="EJ/yr")
            for y in YEARS:
                row[y] = 0.0
            out_rows.append(row)
    return out_rows

# ============================================================
# 3) vLLM generation
# ============================================================
# run the engine in-process, so that its errors are visible in the notebook
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# override the attention backend from the notebook if a kernel is missing for the
# GPU architecture (e.g. Blackwell): SFT_VLLM_ATTN=FLASHINFER | TRITON_ATTN | FLASH_ATTN
if os.environ.get("SFT_VLLM_ATTN"):
    os.environ["VLLM_ATTENTION_BACKEND"] = os.environ["SFT_VLLM_ATTN"]
from vllm import LLM, SamplingParams

records = [json.loads(l) for l in open(DATA_FILE)]
if LIMIT:
    records = records[:LIMIT]

tok = AutoTokenizer.from_pretrained(MERGED_DIR)
prompts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) for r in records]
# chat-template guard: the prompt must end with the assistant marker (and an
# empty think block when the template emits one), same as training
assert prompts[0].rstrip().endswith(("<|im_start|>assistant", "<think>\n\n</think>")), repr(prompts[0][-80:])
plens = [len(x) for x in tok(prompts)["input_ids"]]
n_truncated = sum(p + MAX_NEW_TOKENS > MAX_SEQ_LEN for p in plens)
print(f"prompts: n={len(prompts)} max={max(plens)} tok; over budget: {n_truncated}")

_eos = {tok.eos_token_id}
for t in ("<|im_end|>", "<|endoftext|>"):
    i = tok.convert_tokens_to_ids(t)
    if isinstance(i, int) and i >= 0:
        _eos.add(i)
_eos = sorted(x for x in _eos if x is not None)

_t_eng = time.time()
llm = LLM(model=MERGED_DIR, dtype="bfloat16", max_model_len=MAX_SEQ_LEN,
          gpu_memory_utilization=GPU_UTIL, enable_prefix_caching=True,
          trust_remote_code=True)
_engine_sec = time.time() - _t_eng

def sampling_for(fam):
    kw = dict(temperature=0.0, max_tokens=MAX_NEW_TOKENS, stop_token_ids=_eos)
    if CONSTRAIN:
        schema = SCHEMAS[scope_key(fam)]
        try:                                    # vLLM >= 0.11 API
            from vllm.sampling_params import StructuredOutputsParams
            kw["structured_outputs"] = StructuredOutputsParams(json=schema)
        except ImportError:                     # older vLLM
            from vllm.sampling_params import GuidedDecodingParams
            kw["guided_decoding"] = GuidedDecodingParams(json=schema)
    return SamplingParams(**kw)

params = [sampling_for(r["model_family"]) for r in records]
print("=" * 72)
print(f"  backend              : vLLM, structured output {'ON' if CONSTRAIN else 'OFF'}")
print(f"  cases                : {len(records)}")
print(f"  model                : {MERGED_DIR}")
print(f"  data                 : {DATA_FILE}")
print("=" * 72)
t0 = time.time()
outs = llm.generate(prompts, params)        # continuous batching over all cases
_el = time.time() - t0

rows, n_cap = [], 0
for rec, o in zip(records, outs):
    text = o.outputs[0].text
    if len(o.outputs[0].token_ids) >= MAX_NEW_TOKENS:
        n_cap += 1
    rows += rows_for(rec, text)

import pandas as pd
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
_n = len(records)
print("saved", OUT_CSV)
print("=" * 72)
print(f"  schema-valid JSON    : {_n - fmt['parse_fail']}/{_n}")
print(f"  scope mismatches     : {fmt['scope_mismatch']}/{_n}")
print(f"  hit max_tokens       : {n_cap}/{_n}")
print(f"  wall clock           : {_el / 60:.1f} min  ({_el / _n:.1f}s/case)")
print("=" * 72)
if n_truncated:
    print(f"WARNING: {n_truncated} prompts + {MAX_NEW_TOKENS} new tokens exceed max_model_len={MAX_SEQ_LEN}")

# ── run record next to the CSV: timing + provenance, for the paper's
# "inference cost" numbers and for telling apart runs that differ only in backend
import vllm as _vllm, transformers as _tf, platform, statistics as _st
_gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
# per-request latency under load: first scheduled -> finished, from vLLM's own
# request metrics. Queue wait excluded, sharing the GPU with concurrent requests
# included - the same convention as rag_core's measured sec_per_trajectory
# (its worker latency is also taken under MAX_CONCURRENCY load).
_lat = []
for o in outs:
    m = getattr(o, "metrics", None)
    t_end = getattr(m, "finished_time", None) if m else None
    t_sch = getattr(m, "first_scheduled_time", None) if m else None
    if t_end and t_sch:
        _lat.append(t_end - t_sch)
_lat_stats = ({"p50": round(_st.median(_lat), 2), "mean": round(_st.mean(_lat), 2),
               "p95": round(_st.quantiles(_lat, n=20)[18], 2) if len(_lat) >= 20 else None,
               "max": round(max(_lat), 2), "n": len(_lat)}
              if _lat else None)      # some vLLM versions do not populate metrics
_run_info = {
    "pred_csv": os.path.basename(OUT_CSV),
    "split": SPLIT, "n_cases": _n, "limit": LIMIT,
    "base_model": BASE_MODEL, "tag": TAG, "adapter": ADAPTER, "checkpoint": _ckpt,
    "backend": "vllm", "vllm_version": _vllm.__version__, "transformers_version": _tf.__version__,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "constrained": CONSTRAIN, "max_model_len": MAX_SEQ_LEN, "max_new_tokens": MAX_NEW_TOKENS,
    "gpu_memory_utilization": GPU_UTIL,
    "timing_sec": {"merge": round(_merge_sec, 1), "engine_build": round(_engine_sec, 1),
                   "generate": round(_el, 1), "generate_per_case": round(_el / _n, 2),
                   "per_case_latency_under_load": _lat_stats},
    "prompt_tokens": {"max": int(max(plens)), "mean": round(sum(plens) / len(plens), 1), "total": int(sum(plens))},
    "generated_tokens": {"total": int(_gen_tokens), "mean": round(_gen_tokens / _n, 1),
                         "throughput_tok_per_sec": round(_gen_tokens / _el, 1)},
    "schema_valid": _n - fmt["parse_fail"], "scope_mismatch": fmt["scope_mismatch"],
    "hit_max_tokens": n_cap, "over_length_prompts": n_truncated,
    "finished": _dt.now().isoformat(timespec="seconds"),
}
_info_path = OUT_CSV[:-4] + "_run_info.json"
json.dump(_run_info, open(_info_path, "w"), indent=2)
print("run info ->", os.path.basename(_info_path))

# ── explicit shutdown ────────────────────────────────────────────────────────
# Without this the process can exit with NCCL's "destroy_process_group() was
# not called" warning and occasionally linger holding GPU memory, which makes
# the NEXT run fail its gpu_memory_utilization check until the runtime is
# restarted. Everything below is best-effort: the results are already on disk.
try:
    del llm
except Exception:
    pass
import gc
gc.collect()
try:
    torch.distributed.destroy_process_group()
except Exception:
    pass
torch.cuda.empty_cache()
print("clean shutdown")
