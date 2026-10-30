"""Merge per-shard result files (results/ns3_runs_*.jsonl) into one, de-duplicating."""
import glob, json, os

seen = {}
for f in sorted(glob.glob("results/ns3_runs_*.jsonl")) + (
        ["results/ns3_runs.jsonl"] if os.path.exists("results/ns3_runs.jsonl") else []):
    for line in open(f):
        r = json.loads(line)
        seen[(r["detector"], r["attack"], r["speed"], r["seed"])] = r
with open("results/ns3_full.jsonl", "w") as out:
    for r in seen.values():
        out.write(json.dumps(r) + "\n")
print(f"merged {len(seen)} unique runs -> results/ns3_full.jsonl")
