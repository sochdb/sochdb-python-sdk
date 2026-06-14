#!/bin/bash
# Sequential benchmark orchestrator — runs detached via systemd-run on the box.
cd /root/sochdb-python-sdk
source .venv/bin/activate
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
echo "=== RUN_ALL START $(date) ==="

echo ">>> LoCoMo full $(date)"
python -u benchmarks/run_rag_e2e.py --suite locomo \
  --address 127.0.0.1:50061 --out-dir /root/results_locomo_full --max-answer-tokens 96 \
  > /root/locomo_full.log 2>&1
echo ">>> LoCoMo DONE $(date)"

echo ">>> MemoryAgentBench full $(date)"
python -u benchmarks/run_rag_e2e.py --suite mab \
  --address 127.0.0.1:50061 --out-dir /root/results_mab_full --max-answer-tokens 96 \
  > /root/mab_full.log 2>&1
echo ">>> MAB DONE $(date)"

echo ">>> Hybrid ollama-rerank (15 QA) $(date)"
python -u benchmarks/run_rag_e2e.py --suite locomo --limit 15 --rerank \
  --address 127.0.0.1:50061 --out-dir /root/results_hybrid15 --max-answer-tokens 96 \
  > /root/hybrid15.log 2>&1
echo ">>> HYBRID DONE $(date)"
echo "=== RUN_ALL DONE $(date) ==="
