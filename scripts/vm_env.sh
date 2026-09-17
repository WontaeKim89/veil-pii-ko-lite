# VM 공통 환경 — 빌린 인터프리터 + 사설 pylib. source 해서 쓴다.
export PY=$HOME/wt-kure-insurance-v1/.venv/bin/python
export PYTHONPATH=$HOME/kopii-lite/pylib:$HOME/kopii-lite
export HF_HOME=$HOME/kopii-lite/hf
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd $HOME/kopii-lite
