# MERIT-Net prototype

Python prototype of MERIT-Net (mobility-explained residual CUSUM + lambda-Net + V-Net + one-bit fusion)
and six reimplemented baselines, compared online inside a window-level MANET simulator.

    python3 gen_data.py     # training/validation sequences (data/, ~1 min)
    python3 train.py        # trains lambda-Net, V-Net, GLM, kNN, Ref CNN-LSTM/GRU, CNN-BiLSTM-RF (~7 min, CPU)
    python3 evaluate.py     # 10 detectors x 7 scenarios x 5 speeds x 3 seeds, online (~16 min, 2 cores)
    python3 analyze.py      # tables + results/summary.json

Files: sim.py (simulator, 4 attacks), models.py, detectors.py (online detectors), train.py,
evaluate.py, analyze.py; ckpt/ trained weights + calibration (config.json); results/runs.jsonl raw per-run metrics.
This is a prototype simulator, not NS-3: use it to check the method, not as paper-final numbers.
