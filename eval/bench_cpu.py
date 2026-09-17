"""CPU 지연 벤치 — 실제 한국어 텍스트를 정확히 N 토큰으로 맞춰 측정(더미 반복 문자열 금지: WordPiece 가 [UNK] 로 축약함).

python eval/bench_cpu.py --model <dir> --onnx a.onnx [b.onnx ...] --threads 4 8 --tokens 128 512
"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--onnx", nargs="+", required=True)
ap.add_argument("--threads", nargs="+", type=int, default=[4]); ap.add_argument("--tokens", nargs="+", type=int, default=[128, 512])
ap.add_argument("--data", default=None); ap.add_argument("--n", type=int, default=100); ap.add_argument("--reps", type=int, default=20)
a = ap.parse_args()
import onnxruntime as ort
tok = AutoTokenizer.from_pretrained(a.model)
SAMPLE = ("고객님 안녕하세요. 지난달 요금 명세서와 관련하여 문의드립니다. 담당자 확인 후 처리 결과를 문자로 안내드리겠습니다. "
          "추가로 주소 변경과 자동이체 계좌 등록 절차도 함께 안내드릴 예정이니 준비 서류를 확인해 주세요. ") * 40
ids_full = tok(SAMPLE, add_special_tokens=False)["input_ids"]
res = []
for onnx_path in a.onnx:
    for th in a.threads:
        so = ort.SessionOptions(); so.intra_op_num_threads = th
        sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"]); names = [i.name for i in sess.get_inputs()]
        row = {"onnx": Path(onnx_path).name, "size_mb": round(Path(onnx_path).stat().st_size / 1e6, 1), "threads": th}
        for L in a.tokens:
            ids = [tok.cls_token_id] + ids_full[: L - 2] + [tok.sep_token_id]
            feeds = {"input_ids": np.array([ids], np.int64), "attention_mask": np.ones((1, len(ids)), np.int64)}
            if "token_type_ids" in names: feeds["token_type_ids"] = np.zeros((1, len(ids)), np.int64)
            feeds = {k: v for k, v in feeds.items() if k in names}
            for _ in range(3): sess.run(None, feeds)
            t0 = time.perf_counter()
            for _ in range(a.reps): sess.run(None, feeds)
            row[f"ms_{L}tok"] = round((time.perf_counter() - t0) / a.reps * 1000, 1)
        res.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
json.dump(res, open(Path(a.onnx[0]).parent / "bench_cpu.json", "w"), indent=1)
