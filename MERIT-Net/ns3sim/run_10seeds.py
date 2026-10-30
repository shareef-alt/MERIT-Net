"""10-seed closed-loop evaluation (seeds 11-20), resumable, seed-major so every detector
always has the same number of completed seeds. 160 s runs, identical to the original grid.
Main detectors -> results/ns3_10seed_main.jsonl ; ablation variants -> results/ns3_ablation_180runs.jsonl
(seeds 11-12 of the main detectors already live in results/ns3_scaled_300runs.jsonl and results/ns3_cusum_60runs.jsonl)."""
import os, sys, json
os.environ["EVAL_SIM"] = "160"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ns3_evaluate as E
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = ["Static threshold", "Plain CUSUM", "FAPRP-style kNN", "CNN-BiLSTM-RF", "Ref. CNN-LSTM/GRU", "MERIT-Net"]
ABL = ["MERIT no V-Net", "MERIT no fusion", "MERIT linear λ"]
RES = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))
WORKERS = int(os.environ.get("WORKERS", os.cpu_count() or 2))
OUT_MAIN = os.path.join(RES, "ns3_10seed_main.jsonl")
OUT_ABL = os.path.join(RES, "ns3_ablation_180runs.jsonl")
SPEEDS = [0, 5, 10, 15, 20]
SEEDS = [int(x) for x in os.environ.get("SEEDS", ",".join(map(str, range(11, 21)))).split(",")]
if os.environ.get("SEED_ORDER", "asc") == "desc":      # run from seed 20 downwards (other machine runs upwards)
    SEEDS = sorted(SEEDS, reverse=True)


def load(path, rename=None):
    out = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            out.add((rename or r["detector"], r["attack"], r["speed"], r["seed"]))
    return out


done = load(OUT_MAIN) | load(OUT_ABL) | load(os.path.join(RES, "ns3_scaled_300runs.jsonl")) \
       | load(os.path.join(RES, "ns3_cusum_60runs.jsonl"), rename="Plain CUSUM")
jobs = [(d, a, rate, s, sd) for sd in SEEDS for d in MAIN + ABL for a, rate in E.SCEN for s in SPEEDS
        if (d, a, s, sd) not in done]
print("remaining jobs", len(jobs), "workers", WORKERS, flush=True)
with Pool(WORKERS, initializer=E._init) as p, open(OUT_MAIN, "a") as fm, open(OUT_ABL, "a") as fa:
    for i, r in enumerate(p.imap(E.run_job, jobs), 1):      # imap keeps seed-major order of completion
        f = fa if r["detector"] in ABL else fm
        f.write(json.dumps(r) + "\n"); f.flush()
        print(i, r["detector"], r["attack"], r["speed"], r["seed"], flush=True)
print("DONE", flush=True)
