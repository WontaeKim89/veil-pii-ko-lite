#!/usr/bin/env bash
# release/ 를 Hugging Face 모델 리포로 업로드. 사용: HF_REPO=<user>/veil-pii-ko-lite bash scripts/publish_hf.sh
# 사전: `hf auth login` (또는 HF_TOKEN env). MODEL_CARD.md 가 README.md 로 올라간다.
set -euo pipefail
REPO=${HF_REPO:?e.g. user/veil-pii-ko-lite}; cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
$PY - "$REPO" <<'EOF'
import sys, shutil, tempfile, pathlib
from huggingface_hub import HfApi
repo = sys.argv[1]; api = HfApi(); api.create_repo(repo, repo_type="model", exist_ok=True)
src = pathlib.Path("release"); tmp = pathlib.Path(tempfile.mkdtemp())
# fp32 가중치·설정·토크나이저는 루트에 (transformers 표준), INT8 ONNX 와 디코더는 나란히
for f in src.glob("fp32/*"): shutil.copy(f, tmp / f.name)
for f in ["model.int8.onnx", "veil.py", "labels.yaml", "EVIDENCE.md", "reproduce_claims.sh", "requirements-repro.txt"]:
    if (src / f).exists(): shutil.copy(src / f, tmp / f)
shutil.copy(src / "MODEL_CARD.md", tmp / "README.md")
api.upload_folder(repo_id=repo, folder_path=str(tmp), commit_message="Veil-PII-Ko-Lite v4: fp32 safetensors + INT8 ONNX + decoder")
print("uploaded →", f"https://huggingface.co/{repo}")
EOF
