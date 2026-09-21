"""Veil-PII-Ko-Lite 추론 — ONNX Runtime + 제약 BIOES Viterbi + 한국어 후처리.

토크나이저는 `tokenizers` 를 기본으로 쓴다(transformers 불필요, 의존성 135MB 절감).
tokenizer.json 이 없는 디렉토리를 주면 transformers 로 폴백한다.
"""
import json
import re
from pathlib import Path

import numpy as np

_PARTICLE_NUM = re.compile(r"(이라고|이라는|으로부터|에서는|에서도|까지는|까지도|부터는|부터|까지|으로|이나|이랑|이며|이고|이다|에서|에게|한테|처럼|보다|이면|이야|이가|이는|은|는|이|가|을|를|로|에|의|도|과|와|랑|만)$")
_PARTICLE_PER = re.compile(r"(이라고|이랑|한테|에게|이는|이가|은|는|을|를|도|의|과|와|랑|야|아|님|씨)$")
_NUM_END = re.compile(r"[0-9일년월호번층동실]$")
_WEAK = ("이", "가", "은", "는", "을", "를", "도", "의", "과", "와", "랑")


def _vocative_ok(v, p):
    """호격 '아/야' 는 앞 음절 받침에 따라 갈린다: 지훈아(받침 O)·철수야(받침 X). 수아·서아의 '아' 는 이름의 일부."""
    if p not in ("아", "야") or len(v) < 2:
        return True
    c = ord(v[-2]); has_final = 0xAC00 <= c <= 0xD7A3 and (c - 0xAC00) % 28 != 0
    return has_final if p == "아" else not has_final


class _FastTokenizer:
    """tokenizers 기반. transformers 의 return_overflowing_tokens 와 같은 출력을 만든다."""

    def __init__(self, tokenizer_dir, max_len, stride):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))
        self.tk.enable_truncation(max_length=max_len, stride=stride)
        self.tk.no_padding()                      # attention_mask 로 거르므로 패딩 불필요

    def __call__(self, text):
        enc = self.tk.encode(text)
        wins = [enc] + list(enc.overflowing)
        return ([w.ids for w in wins], [w.attention_mask for w in wins],
                [[tuple(o) for o in w.offsets] for w in wins])


class _SlowTokenizer:
    """transformers 폴백 — tokenizer.json 이 없는 배포본용."""

    def __init__(self, tokenizer_dir, max_len, stride):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.max_len, self.stride = max_len, stride

    def __call__(self, text):
        enc = self.tok(text, return_offsets_mapping=True, truncation=True, max_length=self.max_len,
                       stride=self.stride, return_overflowing_tokens=True, padding=True)
        return (enc["input_ids"], enc["attention_mask"],
                [[tuple(o) for o in w] for w in enc["offset_mapping"]])


class Veil:
    """한국어 PII 탐지기.

        det = Veil("model.int8.onnx", "model_dir")
        det.predict("담당자 김철수(010-1234-5678)에게 문의")
        det.mask("...")                       # [PERSON] 치환
    """

    def __init__(self, onnx_path, tokenizer_dir=None, max_len=512, stride=128, threads=4,
                 o_bias=0.0, merge_adjacent=("ADDRESS",)):
        import onnxruntime as ort
        self.merge_adjacent = set(merge_adjacent or ())   # 인접한 같은 라벨 스팬(공백 1개 이내) 병합 — 벤치 재현 시 ()
        d = Path(tokenizer_dir or Path(onnx_path).parent)
        self.tok = (_FastTokenizer(d, max_len, stride) if (d / "tokenizer.json").exists()
                    else _SlowTokenizer(d, max_len, stride))
        cfg = json.load(open(d / "config.json", encoding="utf-8")) if (d / "config.json").exists() else None
        if cfg and "id2label" in cfg:
            self.id2label = {int(k): v for k, v in cfg["id2label"].items()}
        else:
            self.id2label = {int(k): v for k, v in json.load(open(d / "id2label.json", encoding="utf-8")).items()}
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.in_names = [i.name for i in self.sess.get_inputs()]
        self.max_len, self.stride, self.o_bias = max_len, stride, o_bias
        self.T, self.s0, self.e0 = self._transition()

    # ---- 제약 BIOES Viterbi ----
    def _transition(self):
        n = len(self.id2label); T = np.zeros((n, n), np.float32)
        sp = lambda t: (t.split("-", 1) + [None])[:2] if t != "O" else ("O", None)
        for i in range(n):
            pa, ta = sp(self.id2label[i])
            for j in range(n):
                pb, tb = sp(self.id2label[j])
                ok = (pa in ("O", "E", "S") and pb in ("O", "B", "S")) or (pa in ("B", "I") and pb in ("I", "E") and ta == tb)
                T[i, j] = 0.0 if ok else -1e4
        s0 = np.array([0.0 if sp(self.id2label[i])[0] in ("O", "B", "S") else -1e4 for i in range(n)], np.float32)
        e0 = np.array([0.0 if sp(self.id2label[i])[0] in ("O", "E", "S") else -1e4 for i in range(n)], np.float32)
        return T, s0, e0

    def _decode(self, lp):
        L, n = lp.shape; score = lp[0] + self.s0; back = np.zeros((L, n), np.int32)
        for t in range(1, L):
            cand = score[:, None] + self.T + lp[t][None, :]; back[t] = cand.argmax(0); score = cand.max(0)
        score += self.e0; path = [int(score.argmax())]
        for t in range(L - 1, 0, -1):
            path.append(int(back[t][path[-1]]))
        return path[::-1]

    @staticmethod
    def _spans(tags, offs, probs, win):
        out, cur = [], None
        for i, (t, (s, e)) in enumerate(zip(tags, offs)):
            if e <= s: continue
            if t == "O":
                if cur: out.append(cur); cur = None
                continue
            pre, lab = t.split("-", 1)
            if pre in ("B", "S") or cur is None or cur["label"] != lab:
                if cur: out.append(cur)
                cur = {"start": s, "end": e, "label": lab, "score": float(probs[i]), "n": 1, "win": win}
                if pre == "S": out.append(cur); cur = None
            else:
                cur["end"] = e; cur["score"] += float(probs[i]); cur["n"] += 1
                if pre == "E": out.append(cur); cur = None
        if cur: out.append(cur)
        for sp in out: sp["score"] /= sp.pop("n")
        return out

    @staticmethod
    def _post(spans, text):
        # 공백 트림 → 창 경계 병합 → 겹침 해소 → 조사 제거
        tr = []
        for sp in spans:
            s, e = sp["start"], sp["end"]
            while s < e and text[s].isspace(): s += 1
            while e > s and text[e - 1].isspace(): e -= 1
            if e > s: tr.append({**sp, "start": s, "end": e})
        out = []
        for sp in sorted(tr, key=lambda x: (x["start"], -x["end"])):
            if out and sp["label"] == out[-1]["label"] and sp["start"] <= out[-1]["end"] + 1 and text[out[-1]["end"]:sp["start"]].strip() == "" and sp["win"] != out[-1]["win"]:
                out[-1]["end"] = max(out[-1]["end"], sp["end"]); out[-1]["score"] = max(out[-1]["score"], sp["score"])
            elif out and sp["start"] < out[-1]["end"]:
                if sp["score"] > out[-1]["score"]: out[-1] = sp
            else:
                out.append(sp)
        res = []
        for sp in out:
            s, e = sp["start"], sp["end"]; v = text[s:e]
            if sp["label"] == "PERSON":
                m = _PARTICLE_PER.search(v)
                if m and not _vocative_ok(v, m.group(1)): m = None
                if m and ((len(v) - len(m.group(1)) >= 2 and m.group(1) not in _WEAK) or len(v) - len(m.group(1)) >= 3):
                    e -= len(m.group(1))
            else:
                m = _PARTICLE_NUM.search(v)
                if m and _NUM_END.search(v[: len(v) - len(m.group(1))] or "") and len(v) - len(m.group(1)) >= 2:
                    e -= len(m.group(1))
            if e > s: res.append({"start": s, "end": e, "label": sp["label"], "score": round(sp["score"], 4)})
        return res

    def predict(self, text, bs=16):
        """문자 오프셋 스팬 목록 — [{"start","end","label","score"}], 서로 겹치지 않는다."""
        if not text: return []
        ids, masks, offsets = self.tok(text)
        n = len(ids); spans = []
        for b in range(0, n, bs):
            chunk_ids, chunk_masks = ids[b:b + bs], masks[b:b + bs]
            width = max(len(x) for x in chunk_ids)
            feeds = {
                "input_ids": np.array([x + [0] * (width - len(x)) for x in chunk_ids], np.int64),
                "attention_mask": np.array([x + [0] * (width - len(x)) for x in chunk_masks], np.int64),
            }
            if "token_type_ids" in self.in_names:
                feeds["token_type_ids"] = np.zeros_like(feeds["input_ids"])
            feeds = {k: v for k, v in feeds.items() if k in self.in_names}
            logits = self.sess.run(None, feeds)[0]
            for w in range(logits.shape[0]):
                wi = b + w
                am = np.array(feeds["attention_mask"][w], bool)
                lg = logits[w][am].astype(np.float32)
                offs = [o for o, m in zip(offsets[wi], am) if m]
                if self.o_bias: lg = lg.copy(); lg[:, 0] -= self.o_bias
                mx = lg.max(-1, keepdims=True); lp = lg - mx - np.log(np.exp(lg - mx).sum(-1, keepdims=True))
                path = self._decode(lp); probs = np.exp(lp[np.arange(len(path)), path])
                lo = 0 if wi == 0 else self.stride // 2
                hi = len(path) if wi == n - 1 else len(path) - self.stride // 2
                tags = [self.id2label[i] if lo <= k < hi else "O" for k, i in enumerate(path)]
                spans.extend(self._spans(tags, offs, probs, wi))
        res = self._post(spans, text)
        if self.merge_adjacent:
            merged = []
            for sp in res:
                if merged and sp["label"] in self.merge_adjacent and sp["label"] == merged[-1]["label"] and sp["start"] - merged[-1]["end"] <= 1 and text[merged[-1]["end"]:sp["start"]].strip() == "":
                    merged[-1]["end"] = sp["end"]; merged[-1]["score"] = min(merged[-1]["score"], sp["score"])
                else:
                    merged.append(sp)
            res = merged
        return res

    def mask(self, text, fmt="[{label}]"):
        """탐지된 자리를 라벨로 치환한 문자열."""
        out, pos = [], 0
        for sp in self.predict(text):
            out.append(text[pos:sp["start"]]); out.append(fmt.format(label=sp["label"])); pos = sp["end"]
        out.append(text[pos:])
        return "".join(out)
