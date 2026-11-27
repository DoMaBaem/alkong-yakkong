#!/usr/bin/env python3
"""알콩약콩 LoRA / QLoRA 미세조정.

    python scripts/train_lora.py --model Qwen/Qwen3.5-2B --out runs/qwen35-2b-qlora
    python scripts/train_lora.py --model kakaocorp/kanana-2-3b-instruct --out runs/kanana-qlora
    python scripts/train_lora.py --model Qwen/Qwen3.5-2B --out runs/smoke --limit 16 --epochs 1

기본은 QLoRA(4-bit NF4 모델 위에 LoRA)이다. 베이스라인 평가가 같은 4-bit 모델로 답변을 생성하므로
학습과 평가 조건을 맞추기 위해서다. --no-4bit를 주면 bf16 모델 위에서 LoRA를 학습한다.

- 프롬프트는 baseline_eval.py의 ANSWER_PROMPT와 입력 형식을 그대로 쓴다.
- 손실은 답변(reference) 토큰에만 건다.
- epoch마다 Val loss를 재고, 가장 낮은 epoch의 어댑터만 <out>/adapter에 남긴다.
- 학습 기록(하이퍼파라미터, 학습 파라미터 수, 학습 시간, 최대 GPU 메모리, 버전)은 <out>/train_config.json에 저장한다.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from baseline_eval import ANSWER_PROMPT, load_data  # noqa: E402  베이스라인과 같은 프롬프트·로더

# 4B 이하 공개 모델이라 attention·MLP의 모든 선형층에 LoRA를 붙인다. Qwen3.5의 비전 인코더는 제외한다.
EXCLUDE = r".*(visual|vision).*"


def user_payload(row):
    """baseline_eval.py의 Generator 호출과 같은 사용자 입력."""
    return json.dumps({"question": row["question"], "evidence": row["evidence"] or "제공된 근거 없음"},
                      ensure_ascii=False)


def encode(tokenizer, row, is_qwen35, max_len):
    """프롬프트는 생성 때와 같은 토큰으로, 답변 뒤에는 대화 종료 표시까지 붙여 학습한다."""
    options = {"enable_thinking": False} if is_qwen35 else {}
    messages = [{"role": "system", "content": ANSWER_PROMPT}, {"role": "user", "content": user_payload(row)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **options)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": row["reference"]}],
                                         tokenize=False, **options)
    if full.startswith(prompt) and row["reference"] in full[len(prompt):]:
        completion, matched = full[len(prompt):], True
    else:  # 템플릿이 과거 대화를 다르게 렌더링하면 답변 + EOS로 대신한다
        completion, matched = row["reference"] + tokenizer.eos_token, False
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(c_ids) > max_len:
        raise ValueError(f"{row['id']}: 길이 {len(p_ids) + len(c_ids)} > --max-len {max_len}")
    return {"input_ids": p_ids + c_ids, "labels": [-100] * len(p_ids) + c_ids}, matched, prompt + completion


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [self.pad_id] * (n - len(b["input_ids"])) for b in batch]
        labels = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
        mask = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(mask)}


def version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="새 출력 폴더")
    ap.add_argument("--train", default=str(ROOT / "data/train.jsonl"))
    ap.add_argument("--val", default=str(ROOT / "data/val.jsonl"))
    ap.add_argument("--no-4bit", action="store_true", help="QLoRA 대신 bf16 LoRA")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="시험 실행용: Train/Val 앞 N문항만 사용")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"새 출력 폴더를 지정하세요: {out}")

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments, set_seed)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU가 필요합니다.")
    set_seed(args.seed)
    is_qwen35 = "qwen3.5" in args.model.lower()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # 데이터
    train_rows, val_rows = load_data(args.train), load_data(args.val)
    if args.limit:
        train_rows, val_rows = train_rows[:args.limit], val_rows[:args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded, unmatched = {}, 0
    for name, rows in (("train", train_rows), ("val", val_rows)):
        encoded[name] = []
        for row in rows:
            item, matched, text = encode(tokenizer, row, is_qwen35, args.max_len)
            unmatched += not matched
            encoded[name].append(item)
    lengths = [len(x["input_ids"]) for x in encoded["train"]]
    print(f"[data] train {len(encoded['train'])} / val {len(encoded['val'])}, "
          f"토큰 길이 최대 {max(lengths)} 평균 {sum(lengths) / len(lengths):.0f}, 템플릿 불일치 {unmatched}")
    print("[data] 학습 예시 1개 (마지막 행):\n" + text)

    # 모델
    model_class = AutoModelForImageTextToText if is_qwen35 else AutoModelForCausalLM
    quant = None if args.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    torch.cuda.reset_peak_memory_stats()
    model = model_class.from_pretrained(args.model, device_map={"": 0}, dtype=dtype, quantization_config=quant)
    model.config.use_cache = False
    if quant is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
                      task_type="CAUSAL_LM", target_modules="all-linear", exclude_modules=EXCLUDE)
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"[model] 학습 파라미터 {trainable:,} / 전체 {total:,} ({100 * trainable / total:.3f}%)")

    steps_per_epoch = math.ceil(len(encoded["train"]) / (args.batch_size * args.grad_accum))
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, lr_scheduler_type="cosine",
        warmup_steps=math.ceil(args.warmup_ratio * steps_per_epoch * args.epochs),
        optim="adamw_torch", weight_decay=0.0, max_grad_norm=1.0,
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=5, report_to="none", seed=args.seed, data_seed=args.seed,
        remove_unused_columns=False, dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=encoded["train"], eval_dataset=encoded["val"],
                      data_collator=Collator(tokenizer.pad_token_id))
    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    # 가장 좋은 epoch의 어댑터만 남기고 중간 체크포인트는 지운다(디스크 절약)
    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    best = trainer.state.best_model_checkpoint
    shutil.rmtree(out / "checkpoints", ignore_errors=True)

    history = trainer.state.log_history
    config = {
        "model": args.model,
        "method": "LoRA (bf16)" if args.no_4bit else "QLoRA (4-bit NF4)",
        "hyperparameters": {
            "epochs": args.epochs, "learning_rate": args.lr, "lr_scheduler": "cosine",
            "warmup_steps": targs.warmup_steps, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum, "optimizer": "AdamW (torch)",
            "weight_decay": 0.0, "max_grad_norm": 1.0, "lora_rank": args.rank, "lora_alpha": args.alpha,
            "lora_dropout": args.dropout, "target_modules": "all-linear", "exclude_modules": EXCLUDE,
            "max_len": args.max_len, "gradient_checkpointing": True, "seed": args.seed,
        },
        "trainable_params": trainable, "total_params": total,
        "trainable_ratio": trainable / total,
        "data": {
            "train": args.train, "val": args.val, "limit": args.limit,
            "train_n": len(encoded["train"]), "val_n": len(encoded["val"]),
            "train_sha256": hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
            "val_sha256": hashlib.sha256(Path(args.val).read_bytes()).hexdigest(),
            "max_tokens": max(lengths), "template_unmatched": unmatched,
        },
        "answer_prompt": ANSWER_PROMPT,
        "best_checkpoint": Path(best).name if best else None,
        "best_eval_loss": trainer.state.best_metric,
        "eval_loss_by_epoch": [{"epoch": h["epoch"], "eval_loss": h["eval_loss"]} for h in history if "eval_loss" in h],
        "train_time_sec": round(elapsed, 1),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "gpu": torch.cuda.get_device_name(0),
        "versions": {p: version(p) for p in ("torch", "transformers", "peft", "accelerate", "bitsandbytes")},
        "cuda": torch.version.cuda,
    }
    (out / "train_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "log_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: config[k] for k in ("best_checkpoint", "best_eval_loss", "train_time_sec",
                                             "peak_gpu_memory_gb", "trainable_params")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
