"""
colab_train.py - LoRA fine-tuning on Colab (A100), unsloth framework.
Script version of colab_finetune_qwen3.5-9b.ipynb: same recipe, with every
setting selectable through environment variables.

Colab:
    # pinned stack:
    !pip install -q "unsloth==2026.7.5" "unsloth_zoo==2026.7.6" "transformers==5.5.0" "trl==0.24.0" lm-format-enforcer
    # only if Qwen3.5 fails to load under that pin:
    #   !pip install -q --upgrade unsloth unsloth_zoo "transformers>=5.5.0" "trl>=0.24.0"
    from google.colab import drive; drive.mount('/content/drive')
    %env SFT_DATA_DIR=/content/drive/MyDrive/finetune/data
    %env SFT_DRIVE_ROOT=/content/drive/MyDrive/finetune
    %env SFT_LR=5e-4
    %run colab_train.py

Env vars (defaults = the Config cell of the notebook):
    SFT_BASE_MODEL   base model (default Qwen/Qwen3.5-9B)
    SFT_LOAD_4BIT    1 = QLoRA 4-bit; default 0 = bf16 LoRA (Qwen3.5: keep 0)
    SFT_LR / SFT_EPOCHS / SFT_LORA_R / SFT_SEED / SFT_SFX ("" | "_nonb")
    SFT_DRIVE_ROOT   if set: checkpoints mirrored to <root>/checkpoints/<TAG>,
                     adapter backed up to <root>/adapters/<TAG> at the end
    SFT_RESUME=1     continue from the last mirrored checkpoint (same TAG)
    SFT_PROBE=1      8 longest samples x 5 steps -> peak VRAM, then stop
    SFT_EXPECT_TRAIN / SFT_EXPECT_VAL   split-version guard (see below)

What it does:
  - official Qwen weights, bf16 LoRA r=16/alpha=32 on the 7 projections
  - loss on assistant tokens only (train_on_responses_only)
  - eval loss on val10 every 150 steps; keeps best checkpoint;
    early stopping after 3 evals without improvement
"""

import os

# less fragmentation on a 32k-token context (recommended by the CUDA OOM message)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DATA_DIR = os.environ.get("SFT_DATA_DIR", "/content/drive/MyDrive/finetune/data")
DRIVE_ROOT = os.environ.get("SFT_DRIVE_ROOT", "/content/drive/MyDrive/finetune")   # "" = local /content only
SFX = os.environ.get("SFT_SFX", "")            # "" = with neighbors; "_nonb" = no-neighbor ablation
BASE_MODEL = os.environ.get("SFT_BASE_MODEL", "Qwen/Qwen3.5-9B")   # official weights
LOAD_4BIT = os.environ.get("SFT_LOAD_4BIT", "0") == "1"          # default bf16 LoRA
IS_QWEN35 = "qwen3.5" in BASE_MODEL.lower()
if IS_QWEN35 and LOAD_4BIT:
    raise SystemExit("Unsloth advises against QLoRA (4-bit) on Qwen3.5 - use bf16 (SFT_LOAD_4BIT=0)")
MAX_SEQ_LEN = int(os.environ.get("SFT_MAX_SEQ_LEN", "32768"))   # longest training sample: 18.3k tokens
# hyperparameters
LR = float(os.environ.get("SFT_LR", "5e-4"))
EPOCHS = float(os.environ.get("SFT_EPOCHS", "3"))
LORA_R = int(os.environ.get("SFT_LORA_R", "16"))
TRAIN_SEED = int(os.environ.get("SFT_SEED", "42"))
BATCH_SIZE = 1
GRAD_ACCUM = 16 // BATCH_SIZE
SAVE_STEPS = int(os.environ.get("SFT_SAVE_STEPS", "150"))
EVAL_STEPS = int(os.environ.get("SFT_EVAL_STEPS", "150"))  # SAVE_STEPS must be a multiple of this
PATIENCE = int(os.environ.get("SFT_PATIENCE", "3"))       # 3 evals x 150 steps = 450 steps
PROBE = os.environ.get("SFT_PROBE", "0") == "1"
RESUME = os.environ.get("SFT_RESUME", "0") == "1"
# the short model name is part of the tag
BASE_SHORT = BASE_MODEL.split("/")[-1].replace("-unsloth-bnb-4bit", "").replace("-bnb-4bit", "").lower()
TAG = os.environ.get("SFT_TAG", f"{BASE_SHORT}_lr{LR}_r{LORA_R}{SFX}" + ("_4bit" if LOAD_4BIT else "_bf16")
                     + (f"_seed{TRAIN_SEED}" if TRAIN_SEED != 42 else ""))
OUTPUT_DIR = os.environ.get("SFT_OUT", f"/content/iam_lora_{TAG}")
DRIVE_CKPT = f"{DRIVE_ROOT}/checkpoints/{TAG}" if DRIVE_ROOT else ""

assert os.path.exists(f"{DATA_DIR}/sft_train{SFX}.jsonl"), f"data not found under {DATA_DIR}"
# ── split-version guard ──────────────────────────────────────────────────────
# The split is regenerated in place (ids reassigned), and finetune/data must be
# rebuilt with prepare_sft_data.py afterwards. Pin the split the data came from.
EXPECT_SPLIT = {"train": int(os.environ.get("SFT_EXPECT_TRAIN", "4747")),
                "val": int(os.environ.get("SFT_EXPECT_VAL", "663"))}
_n = {k: sum(1 for _ in open(f"{DATA_DIR}/sft_{k}{SFX}.jsonl")) for k in EXPECT_SPLIT}
assert _n == EXPECT_SPLIT, (f"sft data is {_n}, expected {EXPECT_SPLIT} - finetune/data was built "
                            f"from an older split; re-run prepare_sft_data.py and re-upload data/")
print(TAG, "| base:", BASE_MODEL, "| 4bit:", LOAD_4BIT, "| data:", DATA_DIR, "| out:", OUTPUT_DIR)

import json
import glob
import shutil
import inspect

from unsloth import FastLanguageModel   # also the documented path for Qwen3.5 text SFT
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from transformers import EarlyStoppingCallback, TrainerCallback
from trl import SFTTrainer, SFTConfig

_load_kw = dict(max_seq_length=MAX_SEQ_LEN, load_in_4bit=LOAD_4BIT, dtype=None)
if IS_QWEN35:
    _load_kw["load_in_16bit"] = not LOAD_4BIT   # explicit bf16, per Unsloth's Qwen3.5 guide
model, tokenizer = FastLanguageModel.from_pretrained(BASE_MODEL, **_load_kw)
# some builds hand back a Processor for VLM checkpoints (Qwen3.5); text-only here
tok = getattr(tokenizer, "tokenizer", tokenizer)

model = FastLanguageModel.get_peft_model(
    model, r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth")

# ── LoRA placement report ───────────────────────────────────────────────────
# Qwen3.5-9B: 8 full-attention + 24 Gated-DeltaNet layers (in_proj_*/out_proj,
# not q/k/v/o) -> attention LoRA on 8 layers only, MLP LoRA on all 32.
_lora = [n for n, _ in model.named_modules() if n.endswith("lora_A")]
_by = {}
for n in _lora:
    _by[n.split(".")[-2]] = _by.get(n.split(".")[-2], 0) + 1
print("LoRA modules by name:", _by, "| total", len(_lora))
assert not any("visual" in n or "vision" in n for n in _lora), "LoRA landed on the vision tower"
model.print_trainable_parameters()

ds = load_dataset("json", data_files={
    "train": f"{DATA_DIR}/sft_train{SFX}.jsonl",
    "val": f"{DATA_DIR}/sft_val{SFX}.jsonl"})


def to_text(ex):
    return {"text": tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False,
        enable_thinking=False)}


# chat-template guard: train_on_responses_only matches literal markers; if the
# template stops emitting them the run silently trains on the whole sequence.
_r = tokenizer.apply_chat_template([{"role": "user", "content": "ping"},
                                    {"role": "assistant", "content": "pong"}],
                                   tokenize=False, add_generation_prompt=False,
                                   enable_thinking=False)
for _m in ("<|im_start|>user\n", "<|im_start|>assistant\n"):
    assert _m in _r, f"chat template no longer emits {_m!r}: {_r!r}"
print("chat template OK:", repr(_r[-120:]))

ds = ds.map(to_text, remove_columns=[c for c in ds["train"].column_names
                                     if c != "text"])

# token-length health check BEFORE training: everything must fit in MAX_SEQ_LEN,
# otherwise the assistant marker gets truncated and the sample is dropped
import numpy as np
lens = [len(x) for x in tok(list(ds["train"]["text"]))["input_ids"]]
print(f"train token lengths: p50={np.percentile(lens,50):.0f} "
      f"p95={np.percentile(lens,95):.0f} p99={np.percentile(lens,99):.0f} "
      f"max={max(lens)} | over MAX_SEQ_LEN({MAX_SEQ_LEN}): "
      f"{sum(l > MAX_SEQ_LEN for l in lens)}")
assert sum(l > MAX_SEQ_LEN for l in lens) == 0, "some samples exceed MAX_SEQ_LEN - raise it"
vlens = [len(x) for x in tok(list(ds["val"]["text"]))["input_ids"]]
print(f"val   token lengths: p50={np.percentile(vlens,50):.0f} "
      f"p95={np.percentile(vlens,95):.0f} max={max(vlens)} | "
      f"over MAX_SEQ_LEN({MAX_SEQ_LEN}): {sum(l > MAX_SEQ_LEN for l in vlens)}")
assert sum(l > MAX_SEQ_LEN for l in vlens) == 0, "some val samples exceed MAX_SEQ_LEN - raise it"

# ── version-safe SFTConfig kwargs (TRL/transformers keep renaming these) ─────
_sft_params = inspect.signature(SFTConfig.__init__).parameters
_cfg = {"max_length" if "max_length" in _sft_params else "max_seq_length": MAX_SEQ_LEN}


class DriveCheckpointSync(TrainerCallback):
    """Mirror OUTPUT_DIR's checkpoint-* dirs to DRIVE_CKPT after every save,
    so a runtime cutoff loses at most save_steps worth of work."""

    def on_save(self, args, state, control, **kwargs):
        os.makedirs(DRIVE_CKPT, exist_ok=True)
        for p in glob.glob(f"{DRIVE_CKPT}/*.tmp"):          # leftover from a mid-copy cutoff
            shutil.rmtree(p, ignore_errors=True)
        local = {os.path.basename(p)
                 for p in glob.glob(f"{args.output_dir}/checkpoint-*")}
        for name in sorted(local):                          # copy new ones (tmp+rename = atomic)
            dst = f"{DRIVE_CKPT}/{name}"
            if not os.path.exists(dst):
                shutil.copytree(f"{args.output_dir}/{name}", dst + ".tmp")
                os.rename(dst + ".tmp", dst)
        # keep EVERY checkpoint's adapter weights on Drive (so any step can be
        # scored on val_infer later), but only the LATEST keeps optimizer /
        # scheduler / rng state (that is what RESUME needs; ~400MB per ckpt).
        cks = sorted(glob.glob(f"{DRIVE_CKPT}/checkpoint-*"),
                     key=lambda p: int(p.rsplit("-", 1)[1]))
        for p in cks[:-1]:
            for f in glob.glob(f"{p}/optimizer.pt") + glob.glob(f"{p}/scheduler.pt") \
                     + glob.glob(f"{p}/rng_state*.pth"):
                os.remove(f)
        return control


_callbacks = [] if PROBE else [EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
if DRIVE_CKPT and not PROBE:
    _callbacks.append(DriveCheckpointSync())

_cfg.update(dict(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,   # effective batch 16
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        per_device_eval_batch_size=1,       # default is 8! eval logits at 10-21k tokens
                                            # ([8, seq, 152k vocab] fp32) alone are ~47GB
        prediction_loss_only=True,          # we only read eval_loss: never keep/gather logits
        eval_strategy="steps", eval_steps=EVAL_STEPS,
        save_strategy="steps", save_steps=SAVE_STEPS,
        eval_on_start=True,                 # fail fast: eval problems show up before step 1
        save_total_limit=None,              # keep every checkpoint (see DriveCheckpointSync)
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        dataset_text_field="text",
        report_to="none",
        seed=TRAIN_SEED,
))
# ── PROBE mode: 8 longest samples, 5 steps, report peak VRAM, stop ───────────
if PROBE:
    _longest = [int(i) for i in np.argsort(lens)[::-1][:8]]
    ds["train"] = ds["train"].select(_longest)
    ds["val"] = ds["val"].select(range(2))
    _cfg.update(max_steps=5, num_train_epochs=1, eval_steps=5, save_steps=5,
                eval_on_start=False, save_total_limit=1, load_best_model_at_end=False)
    print(f"[PROBE] 8 longest train samples: {[lens[i] for i in _longest]} tokens")

_drop = [k for k in _cfg if k not in _sft_params]
if _drop:   # never crash on a renamed/removed arg - report and continue
    print("[warn] SFTConfig does not accept", _drop, "- dropping")
    for k in _drop:
        _cfg.pop(k)

# TRL renamed tokenizer= -> processing_class= (Unsloth patches the old name in
# some builds, not all); pick whichever the installed signature has.
_tk_kw = ("processing_class" if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters
          else "tokenizer")
trainer = SFTTrainer(
    model=model, **{_tk_kw: tok},
    train_dataset=ds["train"], eval_dataset=ds["val"],
    callbacks=_callbacks,
    args=SFTConfig(**_cfg),
)

# loss only on the assistant part (Qwen chat template markers)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n")

resume_ckpt = None
if RESUME:
    assert DRIVE_CKPT, "SFT_RESUME=1 needs SFT_DRIVE_ROOT"
    cks = sorted(glob.glob(f"{DRIVE_CKPT}/checkpoint-*"),
                 key=lambda p: int(p.rsplit("-", 1)[1]))
    assert cks, f"SFT_RESUME=1 but no checkpoints under {DRIVE_CKPT}"
    for p in cks:                       # bring them back to fast local disk
        dst = f"{OUTPUT_DIR}/{os.path.basename(p)}"
        if not os.path.exists(dst):
            shutil.copytree(p, dst)
    resume_ckpt = f"{OUTPUT_DIR}/{os.path.basename(cks[-1])}"
    print("resuming from", resume_ckpt)
trainer.train(resume_from_checkpoint=resume_ckpt)

import torch
if PROBE:
    print(f"[PROBE] PEAK VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
          f" / {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
    raise SystemExit("probe finished - unset SFT_PROBE for the real run")

# eval_loss per checkpoint - the table you pick from (best-by-eval_loss is
# picked automatically below; the others are kept for scoring on val_infer)
print("step   eval_loss")
for _e in trainer.state.log_history:
    if "eval_loss" in _e:
        print(f"{int(_e['step']):>5}  {_e['eval_loss']:.6f}")
model.save_pretrained(OUTPUT_DIR + "/best")
tokenizer.save_pretrained(OUTPUT_DIR + "/best")
# persist the train/eval-loss history alongside the adapter
with open(OUTPUT_DIR + "/best/train_log.json", "w") as _f:
    json.dump({"log_history": trainer.state.log_history,
               "best_metric": trainer.state.best_metric,
               "best_model_checkpoint": trainer.state.best_model_checkpoint,
               "base_model": BASE_MODEL, "load_4bit": LOAD_4BIT, "tag": TAG,
               "expect_split": EXPECT_SPLIT},
              _f, indent=2)
_bm = trainer.state.best_metric      # None only if no eval ever ran
print(f"[{TAG}] best val loss: "
      + (f"{_bm:.4f}" if _bm is not None else "n/a")
      + f" (checkpoint {trainer.state.best_model_checkpoint})")
print("adapter saved to", OUTPUT_DIR + "/best")
print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB / "
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")

if DRIVE_ROOT:   # back up the adapter, then free the mid-training checkpoints
    DRIVE_ADAPTER = f"{DRIVE_ROOT}/adapters/{TAG}"
    os.makedirs(os.path.dirname(DRIVE_ADAPTER), exist_ok=True)
    shutil.copytree(OUTPUT_DIR + "/best", DRIVE_ADAPTER, dirs_exist_ok=True)
    print("adapter backed up to", DRIVE_ADAPTER)
    print("kept per-step checkpoints:",
          sorted(os.path.basename(p) for p in glob.glob(f"{DRIVE_CKPT}/checkpoint-*")))
