# NS-3 closed-loop evaluation

NS-3 3.44 port of the MERIT-Net evaluation. Every detector runs **inside the loop**: its
decisions are fed back into packet forwarding, so a wrong conviction really cuts an honest
node's traffic.

## Files

| File | Purpose |
|---|---|
| `meritnet-scenario.cc` | 50 nodes, 1000 m × 1000 m, random waypoint, AODV over 802.11b, CBR/UDP + TCP traffic, four flooding attacks, promiscuous per-neighbor feature sniffing (23 features per observer-neighbor pair per 1-s window) |
| `build_portable.sh`, `build.sh` | compile a `.cc` against the pip-installed NS-3 3.44 |
| `ns3_bridge.py` | runs NS-3 and a Python detector in lockstep (features out, drop decisions in) |
| `ns3_evaluate.py` | closed-loop runner; records false positives, detection delay, isolation and PDR per run |
| `run_cusum_seeds11_12.py`, `run_ablation_seeds11_12.py`, `run_10seeds.py` | the remaining runs behind the paper (all resumable) |
| `ns3_evaluate_merge.py`, `cluster_submit.slurm`, `analyze_full.py` | optional SLURM sharding and merging |
| `dataset_gen.py`, `ns3_train_data.py` | generate MANET-FLOOD and turn it into training sequences for `meritnet/train.py` |
| `ns3_loader.py`, `range.cc` | tensor loader; radio-range calibration (~250 m) |
| `results/` | per-run records for all 2,700 runs in the paper |

## Attacks

Selected with `--attack` and driven by `--rate`; attacks start at `--aStart` (60 s in the paper):

- **rreq / rreq_stealth**: the attacker keeps requesting routes it lacks, so AODV floods RREQs.
  The stealth variant targets real addresses to hide the unknown-destination cue.
- **hello**: the attacker's AODV hello interval is shortened and its transmit power raised.
- **data**: a colluding attacker pair exchanges high-rate UDP traffic through honest relays.
- **syn**: the attacker opens many TCP connections to a victim without completing them.

## Enforcement

A per-node post-reception filter drops received packets whose originator the node has
convicted. Graduated mitigation forwards a suspected neighbor's RREQs with probability
`1 − π`. In this threat model the flooder is the source of its own traffic, so a conviction
blocks the attacker. An honest node wrongly convicted loses its own traffic, which is why
"honest nodes isolated" is a meaningful metric.

## Running

See the top-level `README.md` for the commands that reproduce every run in the paper.
Example single run:

```
./build_portable.sh meritnet-scenario.cc meritnet-scenario
./meritnet-scenario --attack=rreq --attackers=3 --rate=20 --aStart=60 \
    --nNodes=50 --speed=5 --sim=160 --seed=11 --out=win.bin
```

Runs are deterministic for a given seed and software version. Records were regenerated
independently on a second machine with identical pinned software and matched exactly.
