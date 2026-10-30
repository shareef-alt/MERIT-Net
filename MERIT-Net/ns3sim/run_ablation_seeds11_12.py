"""Seeds 11-12 of the three ablation variants (60 runs each, 180 total) -> results/ns3_ablation_180runs.jsonl.
Seeds 13-20 are produced by run_10seeds.py. Resumable."""
import os, sys, json
os.environ.setdefault("EVAL_SIM", "160")          # 160-s closed-loop runs, attacks start at 60 s
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ns3_evaluate as E
from multiprocessing import Pool
OUT = os.path.join(HERE, "results", "ns3_ablation_180runs.jsonl")
WORKERS = int(os.environ.get("WORKERS", os.cpu_count() or 2))
VAR = ["MERIT no V-Net", "MERIT no fusion", "MERIT linear λ"]
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l); done.add((r["detector"], r["attack"], r["speed"], r["seed"]))
jobs = [(v, a, rate, s, sd) for v in VAR for a, rate in E.SCEN for s in [0, 5, 10, 15, 20] for sd in [11, 12]
        if (v, a, s, sd) not in done]
print("jobs", len(jobs), flush=True)
if __name__ == "__main__":
    with Pool(WORKERS, initializer=E._init) as p, open(OUT, "a") as f:
        for r in p.imap_unordered(E.run_job, jobs):
            f.write(json.dumps(r) + "\n"); f.flush(); print(r["detector"], r["attack"], r["speed"], r["seed"], flush=True)
    print("DONE", flush=True)
