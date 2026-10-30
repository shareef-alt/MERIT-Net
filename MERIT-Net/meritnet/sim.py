"""
Window-level packet MANET simulator for MERIT-Net experiments.

Model (one window = 1 s):
  * N nodes, random-waypoint mobility in an AREA x AREA field, radio range R.
  * AODV-style routing: route discovery (RREQ flood over the connected component),
    route breaks detected per window -> RERR from the upstream node, forwarded back
    to the source, then re-discovery with retries/backoff when the destination is
    unreachable; route timeout after idle periods.
  * HELLO every window from every node.
  * Legitimate traffic: CBR/UDP on-off flows and short TCP connections (SYN,
    handshake completion = SYN-ACK observed, ACK feedback).
  * Shared-channel congestion: each node's local load = transmissions it can hear;
    load above capacity C causes proportional loss.
  * Attacks: RREQ flood (fake destination IDs), Hello flood (high-power HELLO, reach
    2R, creates one-way fake links), data flood (colluding pair, UDP, no feedback),
    SYN flood (half-open connections at a victim).
  * Every node observes every neighbour j (packets thinned by a distance-dependent
    reception probability) and produces a 23-feature view X[i, j].
  * A detector plugged into the loop returns per-(observer, suspect) forwarding
    probabilities and convictions, which feed back into flooding, routing and data.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, breadth_first_order

# ----------------------------------------------------------------- constants
N_DEFAULT = 50
AREA = 1000.0
R = 250.0
CAPACITY = 260.0          # transmissions per second a neighbourhood can carry (802.11 broadcast-limited)
HELLO_PER_WIN = 1
CBR_FLOWS = 20
CBR_RATE = 4.0
TCP_START_RATE = 0.02     # new TCP connections per honest node per second
TCP_RATE = 4.0
TCP_MEAN_DUR = 20.0
SYN_BACKLOG = 64
ACTIVE_ROUTE_TIMEOUT = 3

# feature indices
F_RREQ, F_HELLO, F_DATA, F_SYN = 0, 1, 2, 3
F_RERR, F_HMISS, F_DEGJ, F_CHURN, F_NEWFLOW, F_ACTFLOW, F_LINKAGE, F_DDIST, F_DEGI, F_COMP, F_FB = range(4, 15)
F_UNKNOWN, F_ENTROPY, F_DUP, F_HSYM, F_PERFLOW, F_FBRATIO, F_HALFOPEN = range(15, 22)
F_MASK = 22
NF = 23
CAUSE_IDX = list(range(4, 15))
SYMPTOM_IDX = list(range(15, 22))
COUNT_IDX = [0, 1, 2, 3]

ATTACK_TYPES = {"benign": 0, "rreq": 1, "hello": 2, "data": 3, "syn": 4}


class Flow:
    __slots__ = ("src", "dst", "rate", "kind", "on", "t_end", "t_next_toggle", "route",
                 "retry_at", "backoff", "idle", "syn_pending", "syn_tries", "established",
                 "attack", "_pending")

    def __init__(self, src, dst, rate, kind, attack=False):
        self.src, self.dst, self.rate, self.kind = src, dst, rate, kind
        self.on = True
        self.t_end = np.inf
        self.t_next_toggle = np.inf
        self.route = None
        self.retry_at = 0
        self.backoff = 0
        self.idle = 0
        self.syn_pending = kind == "tcp"
        self.syn_tries = 0
        self.established = kind != "tcp"
        self.attack = attack
        self._pending = None


class ManetSim:
    def __init__(self, n=N_DEFAULT, speed=5.0, attack="benign", attack_rate=None,
                 attack_start=60, T=200, seed=0, n_attackers=1, stealth=False):
        self.n, self.speed, self.attack, self.T = n, speed, attack, T
        self.attack_start = attack_start
        self.stealth = stealth
        self.rng = np.random.default_rng(seed)
        rng = self.rng
        self.pos = rng.uniform(0, AREA, (n, 2))
        self.wp = rng.uniform(0, AREA, (n, 2))
        self.vel = self._draw_speed(n)
        # attackers
        self.attackers = []
        if attack != "benign":
            k = 2 if attack == "data" else n_attackers
            self.attackers = list(rng.choice(n, size=k, replace=False))
        self.is_attacker = np.zeros(n, bool)
        self.is_attacker[self.attackers] = True
        defaults = {"rreq": 20.0, "hello": 10.0, "data": 40.0, "syn": 20.0}
        self.attack_rate = attack_rate if attack_rate is not None else defaults.get(attack, 0.0)
        self.honest = np.where(~self.is_attacker)[0]
        self.victim = int(rng.choice(self.honest)) if attack == "syn" else -1
        # flows
        self.flows = []
        for _ in range(CBR_FLOWS):
            s, d = rng.choice(self.honest, 2, replace=False)
            f = Flow(int(s), int(d), CBR_RATE, "cbr")
            f.on = rng.random() < 0.6
            f.t_next_toggle = rng.exponential(30.0 if f.on else 15.0)
            self.flows.append(f)
        if attack == "data":
            a, b = self.attackers
            for s, d in ((a, b), (b, a)):
                f = Flow(int(s), int(d), self.attack_rate, "flood", attack=True)
                f.on = False
                self.flows.append(f)
        if attack == "syn":
            f = Flow(int(self.attackers[0]), self.victim, self.attack_rate, "synflood", attack=True)
            f.on = False
            self.flows.append(f)
        # state for observations
        self.prev_adj = np.zeros((n, n), bool)
        self.link_age = np.zeros((n, n))
        self.prev_dist = None
        self.prev_nbrs = [set() for _ in range(n)]
        self.half_open = 0.0
        self.t = 0
        # metrics
        self.m = {k: 0.0 for k in ("data_sent", "data_recv", "ctrl_tx", "tcp_attempt",
                                   "tcp_estab", "attack_tx")}
        self.blacklist_effects = None

    # ------------------------------------------------------------- mobility
    def _draw_speed(self, k):
        if self.speed <= 0:
            return np.zeros(k)
        return self.rng.uniform(0.5 * self.speed, 1.5 * self.speed, k)

    def _move(self):
        if self.speed <= 0:
            return
        d = self.wp - self.pos
        dist = np.linalg.norm(d, axis=1)
        step = np.minimum(self.vel, dist)
        with np.errstate(invalid="ignore", divide="ignore"):
            u = np.where(dist[:, None] > 1e-9, d / dist[:, None], 0.0)
        self.pos = self.pos + u * step[:, None]
        arrived = dist <= self.vel
        if arrived.any():
            k = int(arrived.sum())
            self.wp[arrived] = self.rng.uniform(0, AREA, (k, 2))
            self.vel[arrived] = self._draw_speed(k)

    # ------------------------------------------------------------- routing
    @staticmethod
    def _bfs_path(graph_csr, src, dst):
        order, pred = breadth_first_order(graph_csr, src, directed=True, return_predecessors=True)
        if pred[dst] < 0 and dst != src:
            return None
        path = [dst]
        while path[-1] != src:
            path.append(pred[path[-1]])
        return path[::-1]

    # ------------------------------------------------------------- main step
    def step(self, pi=None, convict=None):
        """Advance one window.

        pi[i, j]      forwarding probability node i applies to traffic originated by j
        convict[i, j] True if i has blacklisted j (drops all of j's packets, ignores
                      j's HELLOs for routing)
        Returns X (n, n, NF) observation tensor for this window.
        """
        n, rng = self.n, self.rng
        if pi is None:
            pi = np.ones((n, n))
        if convict is None:
            convict = np.zeros((n, n), bool)
        t = self.t
        attack_on = self.attack != "benign" and t >= self.attack_start
        self._move()
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        adj = (dist <= R) & ~np.eye(n, dtype=bool)
        deg = adj.sum(1)

        # Hello flood: attacker heard up to 2R -> one-way fake links (u believes a is a neighbour)
        fake = np.zeros((n, n), bool)
        hello_range = np.full(n, R)
        if attack_on and self.attack == "hello":
            for a in self.attackers:
                hello_range[a] = 2 * R
                far = (dist[a] > R) & (dist[a] <= 2 * R)
                fake[:, a] |= far          # u -> a believed link
                fake[a, :] |= far          # a appears reachable both ways in routing tables
        # routing graph seen by honest nodes: true links, plus fake ones not yet convicted;
        # links to convicted nodes are removed by the convicting node
        route_graph = (adj | fake) & ~convict & ~convict.T
        comp_n, comp_lbl = connected_components(csr_matrix(adj), directed=False)
        comp_size = np.bincount(comp_lbl)[comp_lbl]
        g_route = csr_matrix(route_graph)

        # per-node counters this window
        rreq = np.zeros(n); rreq_unknown = np.zeros(n); rreq_dests = [[] for _ in range(n)]
        rerr = np.zeros(n); hello = np.full(n, float(HELLO_PER_WIN))
        data = np.zeros(n); syn = np.zeros(n); comp = np.zeros(n); fb = np.zeros(n)
        newflow = np.zeros(n); flow_dsts = [set() for _ in range(n)]
        tx = np.zeros(n)                # transmissions per node (for load)
        flood_weight = np.zeros(n)      # RREQ floods originated, weighted by forwarding

        def fwd_frac(j):
            nb = np.where(adj[:, j])[0]
            if len(nb) == 0:
                return 0.0
            return 1.0 - np.prod(1.0 - pi[nb, j])

        def discover(f):
            """RREQ from f.src; returns route or None."""
            s = f.src
            rreq[s] += 1
            rreq_dests[s].append(f.dst)
            flood_weight[s] += fwd_frac(s)
            if fwd_frac(s) < rng.random():      # every neighbour dropped it
                return None
            return self._bfs_path(g_route, s, f.dst)

        # --- lifecycle of legitimate TCP connections
        for s in self.honest:
            if rng.random() < TCP_START_RATE:
                d = int(rng.choice(self.honest[self.honest != s]))
                f = Flow(int(s), d, TCP_RATE, "tcp")
                f.t_end = t + rng.exponential(TCP_MEAN_DUR)
                self.flows.append(f)
                self.m["tcp_attempt"] += 1
                newflow[s] += 1
        alive = []
        for f in self.flows:
            if f.kind == "tcp" and (t >= f.t_end or (f.syn_pending and f.syn_tries > 3)):
                continue
            alive.append(f)
        self.flows = alive

        # --- attack activation
        if attack_on:
            for f in self.flows:
                if f.attack and not f.on:
                    f.on = True

        # --- flow processing
        for f in self.flows:
            if f.kind == "cbr":
                if t >= f.t_next_toggle:
                    f.on = not f.on
                    f.t_next_toggle = t + rng.exponential(30.0 if f.on else 15.0)
                    if f.on:
                        newflow[f.src] += 1
                        if f.idle >= ACTIVE_ROUTE_TIMEOUT:
                            f.route = None
                if not f.on:
                    f.idle += 1
                    continue
                f.idle = 0
            if not f.on:
                continue
            # route check
            if f.route is not None:
                p = f.route
                brk = None
                for k in range(len(p) - 1):
                    u, v = p[k], p[k + 1]
                    if not adj[u, v] or convict[u, v]:
                        brk = k
                        break
                if brk is not None:
                    up = p[brk]
                    if up != f.src:
                        rerr[up] += 1
                        tx[up] += 1
                        for w in p[1:brk]:
                            rerr[w] += 1
                            tx[w] += 1
                    f.route = None
                    f.retry_at = t
            if f.route is None:
                if t < f.retry_at:
                    if f.kind == "tcp" and f.syn_pending:
                        pass
                    continue
                r = discover(f)
                if r is None:
                    rreq[f.src] += 1      # AODV retry inside the window
                    rreq_dests[f.src].append(f.dst)
                    flood_weight[f.src] += fwd_frac(f.src)
                    f.backoff = min(f.backoff + 1, 4)
                    f.retry_at = t + 2 ** f.backoff
                    if f.kind == "tcp" and f.syn_pending:
                        syn[f.src] += 1
                        f.syn_tries += 1
                    continue
                f.route, f.backoff = r, 0
            # handshake for legitimate TCP
            if f.kind == "tcp" and f.syn_pending:
                syn[f.src] += 1
                f.syn_tries += 1
                victim_full = (self.victim == f.dst and self.half_open > SYN_BACKLOG)
                if victim_full:
                    continue
                comp[f.src] += 1
                f.syn_pending = False
                f.established = True
                self.m["tcp_estab"] += 1
            # transmissions along the route
            p = f.route
            if f.kind == "synflood":
                k = rng.poisson(f.rate)
                syn[f.src] += k
                frac = 0.0 if (len(p) > 1 and convict[p[1], f.src]) else 1.0
                carried = k * frac
                self.half_open = 0.9 * self.half_open + carried
                for w in p[:-1]:
                    tx[w] += carried
                self.m["attack_tx"] += carried * (len(p) - 1)
                continue
            k = rng.poisson(f.rate)
            data[f.src] += k
            flow_dsts[f.src].add(id(f))
            frac = 0.0 if (len(p) > 1 and convict[p[1], f.src]) else 1.0
            carried = k * frac
            for w in p[:-1]:
                tx[w] += carried
            if f.attack:
                self.m["attack_tx"] += carried * (len(p) - 1)
            else:
                f._pending = (k, carried, p)
            if f.kind == "tcp":
                fb[f.src] += 0.5 * carried
                tx[f.dst] += 0.5 * carried

        # --- attack control traffic
        if attack_on and self.attack == "rreq":
            for a in self.attackers:
                k = rng.poisson(self.attack_rate)
                rreq[a] += k
                if self.stealth:
                    rreq_dests[a].extend(list(rng.choice(self.honest, size=k)))
                else:
                    rreq_unknown[a] += k
                    rreq_dests[a].extend(list(rng.integers(n, 10 * n, size=k)))
                flood_weight[a] += k * fwd_frac(a)
                self.m["attack_tx"] += k * fwd_frac(a) * comp_size[a]
        if attack_on and self.attack == "hello":
            for a in self.attackers:
                hello[a] = self.attack_rate

        # --- channel load
        tx_total = tx + hello + rerr
        hear = adj | np.eye(n, dtype=bool)
        load = hear.astype(float) @ tx_total
        # high-power hello reaches 2R
        for a in self.attackers:
            if hello_range[a] > R:
                load += (dist[a] <= 2 * R) * hello[a] * (dist[a] > R)
        # RREQ floods: every node of the component rebroadcasts once
        comp_flood = np.bincount(comp_lbl, weights=flood_weight, minlength=comp_n)
        load += comp_flood[comp_lbl] * (deg + 1)
        loss = np.clip(1.0 - CAPACITY / np.maximum(load, 1e-9), 0.0, 1.0)
        ctrl = (comp_flood[comp_lbl] > 0).sum()
        self.m["ctrl_tx"] += float((flood_weight * comp_size).sum() + rerr.sum() + hello.sum())

        # --- delivery of legitimate data
        for f in self.flows:
            pend = getattr(f, "_pending", None)
            if pend is None:
                continue
            k, carried, p = pend
            f._pending = None
            ok = carried
            for k2 in range(len(p) - 1):
                u, v = p[k2], p[k2 + 1]
                if not adj[u, v]:          # fake (hello-flood) or broken hop
                    ok = 0.0
                    break
                ok *= (1.0 - loss[v])
            self.m["data_sent"] += k
            self.m["data_recv"] += ok

        # ------------------------------------------------- observations X[i, j]
        X = np.zeros((n, n, NF), np.float32)
        p_rx = np.where(dist < 0.8 * R, 0.98, 0.98 - (dist - 0.8 * R) / (0.2 * R) * 0.4)
        p_rx = np.clip(p_rx, 0.0, 0.98) * (1.0 - 0.5 * loss[:, None])
        obs_mask = adj.copy()
        for a in self.attackers:
            if hello_range[a] > R:
                far = (dist[:, a] > R) & (dist[:, a] <= 2 * R)
                obs_mask[far, a] = True
                p_rx[far, a] = 0.85
        P = np.where(obs_mask, p_rx, 0.0)

        def thin(c):
            return rng.binomial(np.broadcast_to(np.round(c).astype(np.int64), (n, n)), P).astype(np.float32)

        X[..., F_RREQ] = thin(rreq)
        X[..., F_HELLO] = thin(hello)
        X[..., F_DATA] = thin(data)
        X[..., F_SYN] = thin(syn)
        X[..., F_RERR] = thin(rerr)
        X[..., F_HMISS] = np.where(obs_mask, np.maximum(0.0, HELLO_PER_WIN - np.minimum(X[..., F_HELLO], HELLO_PER_WIN)), 0)
        # j's advertised neighbour set (from its HELLO)
        nbrs = [set(np.where(adj[j])[0]) for j in range(n)]
        churn = np.array([len(nbrs[j] ^ self.prev_nbrs[j]) for j in range(n)], np.float32)
        X[..., F_DEGJ] = np.where(obs_mask, deg[None, :], 0)
        X[..., F_CHURN] = np.where(obs_mask, churn[None, :], 0)
        X[..., F_NEWFLOW] = thin(newflow)
        nflows = np.array([len(s) for s in flow_dsts], np.float32)
        X[..., F_ACTFLOW] = np.where(obs_mask & (X[..., F_DATA] > 0), nflows[None, :], 0)
        self.link_age = np.where(adj, self.link_age + 1, 0)
        X[..., F_LINKAGE] = np.where(obs_mask, np.minimum(self.link_age, 30) / 30.0, 0)
        if self.prev_dist is None:
            dd = np.zeros_like(dist)
        else:
            dd = np.abs(dist - self.prev_dist) / R
        X[..., F_DDIST] = np.where(obs_mask, dd, 0)
        X[..., F_DEGI] = np.where(obs_mask, deg[:, None], 0)
        X[..., F_COMP] = thin(comp)
        X[..., F_FB] = thin(fb)
        # symptoms
        unk = np.zeros(n, np.float32); ent = np.zeros(n, np.float32); dup = np.zeros(n, np.float32)
        for j in range(n):
            dl = rreq_dests[j]
            if dl:
                arr = np.asarray(dl)
                unk[j] = np.mean(arr >= n)
                _, cnt = np.unique(arr, return_counts=True)
                pr = cnt / cnt.sum()
                ent[j] = float(-(pr * np.log2(pr)).sum())
                dup[j] = 1.0 - len(cnt) / len(arr)
        has_rreq = X[..., F_RREQ] > 0
        X[..., F_UNKNOWN] = np.where(has_rreq, unk[None, :], 0)
        X[..., F_ENTROPY] = np.where(has_rreq, ent[None, :], 0)
        X[..., F_DUP] = np.where(has_rreq, dup[None, :], 0)
        # hello symmetry: does j's HELLO list i?
        X[..., F_HSYM] = np.where(obs_mask & (X[..., F_HELLO] > 0), adj.T.astype(np.float32), 0)
        X[..., F_PERFLOW] = np.where(X[..., F_DATA] > 0, X[..., F_DATA] / np.maximum(nflows[None, :], 1), 0)
        X[..., F_FBRATIO] = np.where(X[..., F_DATA] > 0, X[..., F_FB] / np.maximum(X[..., F_DATA], 1), 0)
        X[..., F_HALFOPEN] = np.where(X[..., F_SYN] > 0, np.maximum(0, X[..., F_SYN] - X[..., F_COMP]) / np.maximum(X[..., F_SYN], 1), 0)
        X[..., F_MASK] = obs_mask.astype(np.float32)

        self.prev_nbrs = nbrs
        self.prev_dist = dist
        self.t += 1
        self.last_obs_mask = obs_mask
        self.last_adj = adj
        return X

    def labels(self):
        """Per-node attack-type label for the current (last stepped) window."""
        lab = np.zeros(self.n, np.int64)
        if self.attack != "benign" and (self.t - 1) >= self.attack_start:
            lab[self.attackers] = ATTACK_TYPES[self.attack]
        return lab

    def pdr(self):
        return self.m["data_recv"] / max(self.m["data_sent"], 1.0)
