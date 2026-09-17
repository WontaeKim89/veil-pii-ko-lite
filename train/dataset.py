"""통합 JSONL → 토큰 BIOES 학습 예제 (슬라이딩 윈도우)."""
import json, random
from pathlib import Path


def read_jsonl(paths):
    rows = []
    for p in paths:
        for line in open(p):
            rows.append(json.loads(line))
    return rows


def encode_row(text, spans, tok, label2id, max_len=512, stride=128):
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=max_len, stride=stride,
              return_overflowing_tokens=True)
    n_win = len(enc["input_ids"])
    for w in range(n_win):
        offs = enc["offset_mapping"][w]
        tags = ["O"] * len(offs)
        for sp in spans:
            idx = [i for i, (s, e) in enumerate(offs) if e > s and s < sp["end"] and e > sp["start"]]
            if not idx: continue
            lab = sp["label"]
            if len(idx) == 1: tags[idx[0]] = f"S-{lab}"
            else:
                tags[idx[0]] = f"B-{lab}"; tags[idx[-1]] = f"E-{lab}"
                for i in idx[1:-1]: tags[i] = f"I-{lab}"
        labels = [label2id[t] if (e > s) else -100 for t, (s, e) in zip(tags, offs)]
        yield {"input_ids": enc["input_ids"][w], "attention_mask": enc["attention_mask"][w], "labels": labels}


def build(rows, tok, label2id, max_len=512, stride=128, seed=0, limit=None):
    R = random.Random(seed); rows = list(rows); R.shuffle(rows)
    if limit: rows = rows[:limit]
    out = []
    for r in rows:
        out.extend(encode_row(r["text"], r["spans"], tok, label2id, max_len, stride))
    return out


class ListDataset:
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]
