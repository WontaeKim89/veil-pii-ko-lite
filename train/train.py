"""토큰 분류 학습 — HF Trainer. transformers 4.49 / 5.x 호환.

예) python train/train.py --model monologg/koelectra-base-v3-discriminator \
      --train data/unified/kdpii_train_dlg.jsonl data/unified/bccard_train.jsonl \
      --eval data/unified/kdpii_valid.jsonl --out runs/koelectra --epochs 1
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.labels import load_labels
from train.dataset import read_jsonl, build, ListDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only_valid", action="store_true", help="valid=true 행만 학습(clean 셋)")
    ap.add_argument("--repeat", nargs="*", default=[], help="파일명 부분문자열:배수 (예: kdpii:3)")
    ap.add_argument("--limit_eval", type=int, default=4000); ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=0.1); ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--grad_ckpt", action="store_true")
    a = ap.parse_args()

    import torch
    from transformers import (AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer,
                              DataCollatorForTokenClassification, set_seed)
    set_seed(a.seed)
    ents, tags, label2id, id2label = load_labels()
    tok = AutoTokenizer.from_pretrained(a.model)

    # 학습 행 수집 (+ repeat)
    rows = []
    for p in a.train:
        rs = read_jsonl([p])
        if a.only_valid: rs = [r for r in rs if r.get("valid", True)]
        k = 1
        for spec in a.repeat:
            key, mult = spec.split(":")
            if key in Path(p).name: k = int(mult)
        rows.extend(rs * k)
        print(f"train file {Path(p).name}: rows={len(rs)} x{k}", flush=True)
    t0 = time.time()
    train_items = build(rows, tok, label2id, a.max_len, a.stride, a.seed)
    eval_rows = read_jsonl(a.eval)
    eval_items = build(eval_rows, tok, label2id, a.max_len, a.stride, a.seed, limit=a.limit_eval)
    print(f"encoded train windows={len(train_items)} eval windows={len(eval_items)} in {time.time()-t0:.0f}s", flush=True)

    model = AutoModelForTokenClassification.from_pretrained(a.model, num_labels=len(tags), id2label=id2label, label2id=label2id,
                                                            ignore_mismatched_sizes=True)
    if a.grad_ckpt: model.gradient_checkpointing_enable()

    from seqeval.metrics import f1_score, precision_score, recall_score
    from seqeval.scheme import IOBES

    def compute_metrics(p):
        preds = p.predictions.argmax(-1); labels = p.label_ids
        y_true, y_pred = [], []
        for pr, lb in zip(preds, labels):
            t, q = [], []
            for x, y in zip(pr, lb):
                if y == -100: continue
                t.append(tags[y]); q.append(tags[x])
            y_true.append(t); y_pred.append(q)
        try:
            return {"f1": f1_score(y_true, y_pred, mode="strict", scheme=IOBES),
                    "precision": precision_score(y_true, y_pred, mode="strict", scheme=IOBES),
                    "recall": recall_score(y_true, y_pred, mode="strict", scheme=IOBES)}
        except Exception:
            return {"f1": f1_score(y_true, y_pred), "precision": 0.0, "recall": 0.0}

    ta = dict(output_dir=a.out, learning_rate=a.lr, num_train_epochs=a.epochs, per_device_train_batch_size=a.bs,
              per_device_eval_batch_size=64, warmup_ratio=a.warmup, weight_decay=0.01, logging_steps=50,
              save_total_limit=1, seed=a.seed, report_to=[], dataloader_num_workers=4,
              label_smoothing_factor=a.label_smoothing, max_steps=a.max_steps, save_strategy="epoch", save_only_model=True,
              load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
              bf16=(a.precision == "bf16"), fp16=(a.precision == "fp16"))
    try:
        args = TrainingArguments(eval_strategy="epoch", **ta)
    except TypeError:
        args = TrainingArguments(evaluation_strategy="epoch", **ta)
    if a.max_steps > 0:  # 스텝 기반이면 스텝 단위 평가/저장
        for k, v in dict(eval_steps=a.max_steps, save_steps=a.max_steps).items(): setattr(args, k, v)
        args.save_strategy = "steps"
        try: args.eval_strategy = "steps"
        except Exception: args.evaluation_strategy = "steps"

    trainer = Trainer(model=model, args=args, train_dataset=ListDataset(train_items), eval_dataset=ListDataset(eval_items),
                      data_collator=DataCollatorForTokenClassification(tok), compute_metrics=compute_metrics)
    trainer.train()
    m = trainer.evaluate()
    print("FINAL_EVAL", json.dumps(m), flush=True)
    final = Path(a.out) / "final"; trainer.save_model(str(final)); tok.save_pretrained(str(final))
    json.dump({"args": vars(a), "eval": m, "train_windows": len(train_items)}, open(final / "train_meta.json", "w"), ensure_ascii=False, indent=1)
    # 체크포인트 디렉토리 정리(디스크)
    import shutil
    for d in Path(a.out).glob("checkpoint-*"): shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
