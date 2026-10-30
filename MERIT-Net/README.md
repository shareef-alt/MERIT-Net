# MERIT-Net: Lightweight Mobility-Explained Flooding Detection for MANETs

Code, closed-loop NS-3 simulation results, and the **MANET-FLOOD** dataset for the article

> S. K. Shareef, D. Siri, K. Suresh, M. Adudhodla, V. Bande, and K. Jabeen,
> "MERIT-Net: Lightweight Mobility-Explained Flooding Detection for Mobile Ad Hoc Networks,"
> submitted to *IEEE Open Journal of the Communications Society*, 2026.

MERIT-Net is an on-node flooding detector for mobile ad hoc networks. A small neural network
(λ-Net, 626 parameters) predicts each neighbor's legitimate load from locally overheard
link-break and flow evidence. A per-stream residual CUSUM test flags only the *unexplained*
excess, with false-alarm and detection-delay bounds that do not depend on node speed. A second
small network (V-Net, 2,101 parameters) runs only for suspected neighbors. Neighbors fuse
one-bit decisions, and mitigation is graduated.

## Main results (NS-3 3.44, 1,800 closed-loop runs, 95% bootstrap CIs)

Six detectors × 300 scenarios (six conditions × five speeds 0–20 m/s × ten seeds), 50 nodes,
160 s per run, attacks start at 60 s.

| Detector | Benign false-positive rate | Detection rate | Mean delay (s) | Honest nodes isolated / run | PDR under attack |
|---|---|---|---|---|---|
| **MERIT-Net** | **0.0006** [0.0000, 0.0019] | **1.00** | 6.4 | **1.54** | **0.797** |
| Ref. CNN-LSTM/GRU | 0.0087 [0.0075, 0.0099] | 1.00 | 2.8 | 31.51 | 0.777 |
| CNN-BiLSTM-RF | 0.0161 | 1.00 | 2.9 | 39.33 | 0.753 |
| FAPRP-style kNN | 0.0178 | 1.00 | 5.0 | 44.40 | 0.538 |
| Flow-based CUSUM | 0.0260 | 1.00 | 3.1 | 45.42 | 0.572 |
| Static threshold | 0.0000 | 0.27 | 49.6 | 1.70 | 0.818 |

MERIT-Net's isolation advantage over every baseline is significant at p ≤ 6.9×10⁻⁵⁰ (paired
Wilcoxon signed-rank test, Holm-corrected). Full tables with confidence intervals are in
`analysis/stats.json`.

## Repository layout

```
meritnet/          detector code: sim.py (feature definitions + window simulator), models.py
                   (λ-Net, V-Net, baselines), detectors.py (online detectors), train.py
  ckpt/            trained weights and calibration used for every result in the paper
  results/         prototype-simulator results (design stage, not the paper's numbers)
ns3sim/            NS-3 3.44 scenario and closed-loop evaluation
  meritnet-scenario.cc   50-node AODV/802.11b scenario, four flooding attacks, feature sniffing
  ns3_bridge.py          detector-in-the-loop bridge (decisions fed back into forwarding)
  ns3_evaluate.py        closed-loop runner (seeds 11-12, five detectors)
  run_cusum_seeds11_12.py, run_ablation_seeds11_12.py, run_10seeds.py   remaining runs
  dataset_gen.py, ns3_train_data.py      MANET-FLOOD generation and training-set preparation
  results/               all 2,700 per-run records behind the paper (JSON lines)
analysis/          statistics, figures and theory checks
  stats.py               merges results, bootstrap CIs, Wilcoxon + Holm tests -> stats.json
  paper_numbers.py       every number quoted in the paper, computed from stats.json
  make_figs.py, make_fig1.py   Figs. 1-6
  verify_theory.py       numerical checks of the propositions and threshold calibration
  position_sensitivity.py      λ-Net without the five position-derived features (Sec. VII-H)
data/              MANET-FLOOD dataset (flat CSV, zipped) and its datasheet
figures/           Figs. 1-6 as PNG and PDF
notebooks/         Google Colab notebook to run the 10-seed evaluation on all CPU cores
```

## Quick start: reproduce the paper's tables and figures (about 1 minute)

```bash
pip install -r requirements.txt
python3 analysis/stats.py          # CIs and significance tests -> analysis/stats.json
python3 analysis/make_figs.py      # Figs. 2-6 -> figures/
python3 analysis/make_fig1.py      # Fig. 1   -> figures/
python3 analysis/verify_theory.py  # theory checks (Propositions 1-5, thresholds)
```

These scripts read only the stored run records in `ns3sim/results/`. They regenerate the
confidence intervals, p-values and figures reported in the paper exactly.

## Full re-run of the simulations

Each closed-loop run takes about 2 minutes on one CPU core; all 2,700 runs take about 90
core-hours. The runs are deterministic: the same seed and the same pinned software reproduce
every record exactly.

```bash
pip install -r requirements.txt            # includes the NS-3 3.44 Python wheel
cd ns3sim
./build_portable.sh meritnet-scenario.cc meritnet-scenario
./meritnet-scenario --nNodes=50 --speed=5 --sim=20 --seed=1 --out=/tmp/smoke.bin   # smoke test

# seeds 11-12, five detectors
EVAL_SPEEDS=0,5,10,15,20 EVAL_SEEDS=11,12 EVAL_WORKERS=$(nproc) python3 ns3_evaluate.py
mv results/ns3_runs.jsonl results/ns3_scaled_300runs.jsonl
python3 run_cusum_seeds11_12.py            # flow-based CUSUM, seeds 11-12
python3 run_ablation_seeds11_12.py         # three ablation variants, seeds 11-12
WORKERS=$(nproc) python3 run_10seeds.py    # all nine detectors, seeds 13-20 (resumable)
```

All runners are resumable: if a run is interrupted, start it again and it continues where it
stopped. `run_10seeds.py` accepts `RESULTS_DIR`, `WORKERS`, `SEEDS` and `SEED_ORDER`
(`asc`/`desc`), so two machines can split the seeds. `notebooks/MERIT-Net_colab_10seeds.ipynb`
does this on Google Colab.

### Retraining the models (optional)

The checkpoints in `meritnet/ckpt/` are the ones used in the paper. To retrain from scratch:

```bash
cd ns3sim && python3 dataset_gen.py && python3 ns3_train_data.py   # NS-3 runs -> meritnet/data/
cd ../meritnet && python3 train.py                                 # λ-Net, V-Net, baselines -> ckpt/
```

## MANET-FLOOD dataset

`data/manet_flood_csv.zip` contains about 618,000 labeled per-neighbor, per-window records from
168 NS-3 runs. These cover benign traffic and RREQ, stealthy RREQ, hello, data and SYN
flooding, at five speeds. Every record is what one node can observe about one neighbor in one
1-second window (23 features). See `data/DATASHEET.md` for the features, labels, collection
process and known gaps. The full raw tensors (about 200 MB) are archived on Zenodo; they can
also be regenerated with `ns3sim/dataset_gen.py`.

## License

Code: MIT (`LICENSE`). Dataset and simulation results: CC BY 4.0 (`data/LICENSE.md`).

## Citation

Please cite the article above if you use this code or the dataset. Machine-readable metadata
is in `CITATION.cff`.

## Note on AI assistance

As disclosed in the article's Acknowledgment, the code in this repository was written with
the assistance of Claude (Anthropic). The authors reviewed it and take responsibility for it.
