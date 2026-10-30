#!/bin/bash
#SBATCH --job-name=meritnet-eval
#SBATCH --array=0-63            # 64 shards; raise for more parallelism
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_%A_%a.out
#
# Full-scale closed-loop MERIT-Net evaluation as a SLURM array job.
# Each array task runs one shard of the (detector x attack x speed x seed) grid;
# the runner is resumable, so re-submitting completes any unfinished shards.
#
# Before submitting:
#   pip install --user ns3 torch numpy scipy scikit-learn joblib
#   ./build.sh meritnet-scenario.cc meritnet-scenario       # build the NS-3 binary
#   (models already trained -> ../meritnet/ckpt; else run ../meritnet/train.py)
#
# After all tasks finish:
#   python3 ns3_evaluate_merge.py        # merge shard files -> results/ns3_full.jsonl
#   python3 analyze_full.py              # tables + significance tests
#
set -euo pipefail
mkdir -p logs results

export EVAL_NSHARDS=${SLURM_ARRAY_TASK_COUNT:-64}
export EVAL_SHARD=${SLURM_ARRAY_TASK_ID:-0}
export EVAL_WORKERS=${SLURM_CPUS_PER_TASK:-4}

# ---- the full paper grid ----
export EVAL_DETECTORS="Static threshold,FAPRP-style kNN,CNN-BiLSTM-RF,Ref. CNN-LSTM/GRU,MERIT-Net,MERIT no V-Net,MERIT no fusion,MERIT linear λ"
export EVAL_SPEEDS="0,5,10,15,20"
export EVAL_SEEDS="1,2,3,4,5,6,7,8,9,10"
export EVAL_SIM="200"
# node-density sweep: run the same array again with EVAL_NODES overrides if your
# scenario build parameterises N (the generator already does via --nNodes).

python3 ns3_evaluate.py
