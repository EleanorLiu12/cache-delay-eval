"""Discrete-event routing simulator calibrated on the A30 runs.

Each instance follows vLLM v1's scheduler: a 2,048-token step budget with
chunked prefill, running requests first, FCFS admission while free blocks
last, recompute preemption, and a block pool whose free queue evicts cached
blocks in least-recently-freed order. The router keeps the same indicators as
``routing.py`` (batch size and queued prefill tokens, updated at dispatch,
first token and completion) and one of four prefix indexes:

``exact``    the engine's current cached blocks (what KV events give)
``dispatch`` every dispatched prefix, never removed (approximate tree
             between trims, as in the SGLang and vLLM routers)
``sglang``   ``dispatch`` plus the SGLang/vLLM router LRU trim: every
             ``trim_s`` seconds, drop least recently used blocks above
             ``trim_tokens`` per instance
``lru``      dispatched prefixes in an LRU bounded by the instance's KV
             capacity (an approximate tree sized to the cache)
``inflight`` ``exact`` plus the prefixes of requests dispatched to the
             instance whose prefill has not finished yet

Policy ``lmetric_detector`` is LMetric plus the two-phase KV hotspot detector
of the LMetric paper (OSDI '26, Section 5.2); see ``HotspotDetector``.

Policy ``pin`` sends a follow-up request (see ``load_trace``) to the instance
that served its parent and routes everything else by LMetric: session
stickiness with ideal session knowledge. Policy ``smetric`` is SMetric's
PD-colocated router (arXiv 2607.08565, Figure 27); see ``SMetric``.
"""

from __future__ import annotations

import heapq
import json
import math
import statistics
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field

BLOCK = 16
CLASS_BLOCKS = 32
INDEXES = ("exact", "dispatch", "sglang", "lru", "inflight")
POLICIES = ("load", "lmetric", "affinity", "sglang_ca", "lmetric_detector", "pin", "smetric")


@dataclass
class Cost:
    """Step time = base + per_token * tokens + prefill_pair * prefill attention
    pairs + decode_ctx * decode context tokens; TTFT adds ``overhead``."""
    base: float = 7.5
    per_token: float = 0.05
    prefill_pair: float = 8e-6
    decode_ctx: float = 6e-4
    overhead: float = 1.0
    overlap: float = 0.0     # 1: decode attention hides under prefill compute

    def step(self, chunks, decode_ctx_tokens, n_decode):
        tokens = n_decode + sum(n for n, _ in chunks)
        pairs = sum(n * (k + n / 2) for n, k in chunks)
        compute = self.per_token * tokens + self.prefill_pair * pairs
        memory = self.decode_ctx * decode_ctx_tokens
        return self.base + compute + memory - self.overlap * min(compute, memory)


@dataclass
class EngineConfig:
    num_blocks: int = 1733          # 27,744 tokens / 16, minus vLLM's null block
    token_budget: int = 2048
    max_seqs: int = 256


@dataclass
class SimRequest:
    rid: str
    arrival_ms: float
    keys: list            # trace block ids of the full prompt blocks
    input_len: int
    output_len: int
    cls: str = ""
    # engine state
    computed: int = 0
    generated: int = 0
    table: list = field(default_factory=list)
    out_keys: list = field(default_factory=list)
    first_ms: float | None = None
    done_ms: float | None = None
    engine_hit: int | None = None
    instance: int | None = None
    predicted_hit: int = 0
    preempted: int = 0
    # router view at dispatch: instances holding the class prefix (-1 if the
    # prompt is shorter than the class prefix), and detector decisions
    holders: int = -1
    alarm: bool = False
    filtered: bool = False
    # session view: rid of the earlier request this one extends, and the
    # tokens of this prompt that parent's prompt covers (SMetric's est_hit)
    parent: str | None = None
    est_hit: int = 0
    stuck: bool = False

    @property
    def length(self):
        return self.input_len + self.generated

    def key(self, i):
        """Cache key of full block i of the current sequence, or None."""
        if i < len(self.keys):
            return self.keys[i]
        j = i - len(self.keys)
        while len(self.out_keys) <= j:
            self.out_keys.append(("out", self.rid, len(self.out_keys)))
        # A block that mixes prompt tail and output tokens is unique too.
        return self.out_keys[j]


class Engine:
    def __init__(self, cfg: EngineConfig, cost: Cost):
        self.cfg, self.cost = cfg, cost
        self.free = OrderedDict((b, None) for b in range(cfg.num_blocks))
        self.block_key: list = [None] * cfg.num_blocks
        self.ref = [0] * cfg.num_blocks
        self.cached: dict = {}            # key -> block
        self.waiting: deque[SimRequest] = deque()
        self.running: list[SimRequest] = []
        self.busy = False

    # block pool -----------------------------------------------------------
    def _pop_free(self):
        block, _ = self.free.popitem(last=False)
        key = self.block_key[block]
        if key is not None:
            del self.cached[key]
            self.block_key[block] = None
        return block

    def _release(self, req):
        for block in reversed(req.table):
            self.ref[block] -= 1
            if self.ref[block] == 0:
                self.free[block] = None
        req.table = []

    def hit_blocks(self, keys):
        n = 0
        for key in keys:
            if key not in self.cached:
                break
            n += 1
        return n

    def _cache_full(self, req):
        for i in range(req.computed // BLOCK):
            block = req.table[i]
            if self.block_key[block] is None:
                key = req.key(i)
                if key not in self.cached:
                    self.cached[key] = block
                    self.block_key[block] = key

    def _grow(self, req, tokens):
        """Make the block table cover ``tokens`` tokens; False if out of blocks."""
        need = math.ceil(tokens / BLOCK) - len(req.table)
        if need > len(self.free):
            return False
        for _ in range(need):
            block = self._pop_free()
            self.ref[block] = 1
            req.table.append(block)
        return True

    # scheduler ------------------------------------------------------------
    def schedule(self):
        """Return (chunks, decodes) for one step; mutates allocation state."""
        budget = self.cfg.token_budget
        plan = []                      # (req, new_tokens)
        preempted = False
        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            new = min(req.length - req.computed, budget)
            while not self._grow(req, req.computed + new):
                victim = self.running.pop()
                self._release(victim)
                victim.computed = 0
                victim.preempted += 1
                self.waiting.appendleft(victim)
                preempted = True
                if victim is req:
                    break
            else:
                plan.append((req, new))
                budget -= new
                i += 1
                continue
            break
        if not preempted:
            while self.waiting and budget > 0 and len(self.running) < self.cfg.max_seqs:
                req = self.waiting[0]
                n_full = req.length // BLOCK
                keys = [req.key(j) for j in range(n_full)]
                hit = min(self.hit_blocks(keys), (req.length - 1) // BLOCK)
                new = min(req.length - hit * BLOCK, budget)
                hit_blocks = [self.cached[k] for k in keys[:hit]]
                evictable = sum(self.ref[b] == 0 for b in hit_blocks)
                need = math.ceil((hit * BLOCK + new) / BLOCK) - hit
                if need + evictable > len(self.free):
                    break
                for b in hit_blocks:
                    if self.ref[b] == 0:
                        del self.free[b]
                    self.ref[b] += 1
                req.table = hit_blocks
                req.computed = hit * BLOCK
                if req.engine_hit is None:
                    req.engine_hit = hit * BLOCK
                self._grow(req, req.computed + new)
                self.waiting.popleft()
                self.running.append(req)
                plan.append((req, new))
                budget -= new
        chunks, decode_ctx, n_decode = [], 0, 0
        for req, new in plan:
            if req.generated > 0 and req.length - req.computed == 1:
                decode_ctx += req.length
                n_decode += 1
            else:
                chunks.append((new, req.computed))
        return plan, self.cost.step(chunks, decode_ctx, n_decode) if plan else 0.0

    def finish_step(self, plan, now, on_first, on_done):
        for req, new in plan:
            req.computed += new
            if req.computed < req.length:      # prefill chunk, more to go
                self._cache_full(req)
                continue
            req.generated += 1                 # end of prefill or a decode step
            self._cache_full(req)
            if req.first_ms is None:
                req.first_ms = now
                on_first(req)
            if req.generated >= req.output_len:
                req.done_ms = now
                self.running.remove(req)
                self._release(req)
                on_done(req)


class RouterIndex:
    def __init__(self, mode, engine: Engine, trim_tokens=None):
        self.mode, self.engine = mode, engine
        self.lru: OrderedDict = OrderedDict()
        self.cap = engine.cfg.num_blocks if mode == "lru" else None
        self.trim_blocks = (trim_tokens // BLOCK) if trim_tokens else None
        self.pending: Counter = Counter()

    def hits(self, keys):
        if self.mode == "exact":
            return self.engine.hit_blocks(keys)
        if self.mode == "inflight":
            n = 0
            for key in keys:
                if key not in self.engine.cached and not self.pending[key]:
                    break
                n += 1
            return n
        n = 0
        for key in keys:
            if key not in self.lru:
                break
            n += 1
        return n

    def dispatched(self, keys):
        if self.mode == "exact":
            return
        if self.mode == "inflight":
            self.pending.update(keys)
            return
        for key in keys:
            self.lru[key] = None
            self.lru.move_to_end(key)
        if self.cap is not None:
            while len(self.lru) > self.cap:
                self.lru.popitem(last=False)

    def prefilled(self, keys):
        if self.mode == "inflight":
            self.pending.subtract(keys)

    def trim(self):
        if self.mode == "sglang" and self.trim_blocks is not None:
            while len(self.lru) > self.trim_blocks:
                self.lru.popitem(last=False)


def lmetric_score(i, bs, queued, hits, req):
    """P-token x BS: new prefill tokens (queued plus uncached input) times batch size."""
    return (queued[i] + req.input_len - hits[i] * BLOCK) * (bs[i] + 1)


class HotspotDetector:
    """Two-phase KV hotspot detector from the LMetric paper, Section 5.2.

    For class c (requests sharing the first ``CLASS_BLOCKS`` blocks), x is c's
    share of all arrivals in the trailing ``window_ms`` and M the instances
    whose cache holds c's prefix. Equation 2, x/(1-x) <= |M|/(N-|M|), is
    equivalent to x <= |M|/N.

    Phase 1 raises an alarm on a class request when Equation 2 fails (with
    0 < |M| < N and at least ``min_count`` class arrivals in the window).
    Phase 2 then follows the class's requests: a request counts toward the
    streak when LMetric's best score on M is no worse than its best score
    outside M; any other request resets it. Once 2|M| consecutive requests
    count, M is excluded for that class's requests until a class request
    passes Equation 2 again. The paper does not say how long filtering lasts;
    holding it while the alarm stands is our reading.
    """

    def __init__(self, n_instances, window_ms=60_000.0, min_count=5):
        self.n, self.window_ms, self.min_count = n_instances, window_ms, min_count
        self.arrivals: deque = deque()
        self.counts: Counter = Counter()
        self.streak: Counter = Counter()
        self.filtering: set = set()

    def observe(self, req, now):
        self.arrivals.append((now, req.cls))
        self.counts[req.cls] += 1
        while self.arrivals[0][0] <= now - self.window_ms:
            _, cls = self.arrivals.popleft()
            self.counts[cls] -= 1

    def allowed(self, req, hits, bs, queued):
        """Instances the request may go to, or None for all of them."""
        if req.holders < 0:
            return None
        cls, m, count = req.cls, req.holders, self.counts[req.cls]
        if not (0 < m < self.n and count >= self.min_count
                and count / len(self.arrivals) > m / self.n):
            self.streak[cls] = 0
            self.filtering.discard(cls)
            return None
        req.alarm = True
        if cls not in self.filtering:
            hot = min(lmetric_score(i, bs, queued, hits, req)
                      for i in range(self.n) if hits[i] >= CLASS_BLOCKS)
            cold = min(lmetric_score(i, bs, queued, hits, req)
                       for i in range(self.n) if hits[i] < CLASS_BLOCKS)
            self.streak[cls] = self.streak[cls] + 1 if hot <= cold else 0
            if self.streak[cls] < 2 * m:
                return None
            self.filtering.add(cls)
        req.filtered = True
        return [i for i in range(self.n) if hits[i] < CLASS_BLOCKS]


@dataclass
class SMetric:
    """SMetric's PD-colocated router (arXiv 2607.08565, Figure 27).

    A follow-up request sticks to the instance with the longest prefix hit if
    that hit exceeds ``hit_ratio`` of the hit its history implies (the session
    is still cached there) and the instance's estimated TTFT, queued prefill
    cost plus the request's own, is within ``slack`` times the TTFT SLO
    b + a * L. Otherwise the request goes to the instance minimizing
    estimated prefill cost times load. Costs use Eq. 1, c_lin * n +
    c_att * n * (L - n / 2), with the simulator's own per-token and
    attention coefficients, in ms. Two departures: the TPOT filter (Figure 27,
    lines 5-6) is omitted, since the simulator has no TPOT SLO and chunked
    prefill bounds a step's stall; the load factor is batch size + 1, as in
    the simulator's LMetric, so an idle instance does not score zero. SLO
    defaults are the paper's (b = 1 s, a = 62.5 ms per 1K tokens); slack and
    hit_ratio are its defaults.
    """
    slack: float = 1.0
    hit_ratio: float = 0.5
    slo_b_ms: float = 1000.0
    slo_a_ms: float = 0.0625

    def prefill_cost(self, cost: Cost, length, hit):
        n = length - hit
        return cost.per_token * n + cost.prefill_pair * n * (length - n / 2)

    def choose(self, cost, bs, qcost, hits, req, turn):
        n = len(bs)
        c = [h * BLOCK for h in hits]
        load = [qcost[i] + self.prefill_cost(cost, req.input_len, c[i]) for i in range(n)]
        prev = min(range(n), key=lambda i: (-c[i], load[i], i))
        if (req.parent is not None and c[prev] > self.hit_ratio * req.est_hit
                and load[prev] <= self.slack * (self.slo_b_ms + self.slo_a_ms * req.input_len)):
            req.stuck = True
            return prev
        scores = [load[i] * (bs[i] + 1) for i in range(n)]
        best = min(scores)
        tied = [i for i in range(n) if scores[i] == best]
        return tied[turn % len(tied)]


def choose(policy, bs, queued, hits, req, turn, tree_sizes=None, allowed=None):
    n = len(bs)
    if policy == "load":
        scores = [(bs[i], queued[i]) for i in range(n)]
    elif policy in ("lmetric", "lmetric_detector"):
        scores = [(lmetric_score(i, bs, queued, hits, req),) for i in range(n)]
    elif policy == "affinity":
        scores = [(-hits[i], bs[i]) for i in range(n)]
    elif policy == "sglang_ca":
        # SGLang/vLLM router cache_aware defaults: abs 64, rel 1.5, threshold 0.3.
        lo, hi = min(bs), max(bs)
        if hi - lo > 64 and hi > 1.5 * lo:
            scores = [(bs[i],) for i in range(n)]
        else:
            best = max(hits)
            rate = best * BLOCK / max(1, req.input_len)
            if rate > 0.3:
                scores = [(-hits[i], bs[i]) for i in range(n)]
            else:
                scores = [(tree_sizes[i], bs[i]) for i in range(n)]
    else:
        raise ValueError(policy)
    candidates = range(n) if allowed is None else allowed
    best = min(scores[i] for i in candidates)
    tied = [i for i in candidates if scores[i] == best]
    return tied[turn % len(tied)]


def simulate(requests, n_instances, policy, index, cost=None, cfg=None,
             trim_s=120.0, trim_tokens=2**26 // 4, detector_window_s=60.0, tie_offset=0,
             smetric=None):
    """Run one replay; returns the request list with timings filled in.
    ``tie_offset`` shifts the round-robin tie-break (a placebo perturbation)."""
    cost = cost or Cost()
    cfg = cfg or EngineConfig()
    engines = [Engine(cfg, cost) for _ in range(n_instances)]
    idx = [RouterIndex(index, e, trim_tokens) for e in engines]
    bs = [0] * n_instances
    queued = [0] * n_instances
    qcost = [0.0] * n_instances          # SMetric's queued prefill cost (ms)
    smetric = smetric or SMetric()
    placed: dict[str, int] = {}          # rid -> instance, for ``pin``
    new_tokens: dict[str, int] = {}
    new_cost: dict[str, float] = {}
    full_keys: dict[str, list] = {}
    events = []
    seq = 0

    def push(t, kind, payload):
        nonlocal seq
        heapq.heappush(events, (t, seq, kind, payload))
        seq += 1

    for req in requests:
        push(req.arrival_ms, 0, req)
    if index == "sglang":
        horizon = requests[-1].arrival_ms if requests else 0
        t = trim_s * 1000
        while t < horizon:
            push(t, 2, None)
            t += trim_s * 1000

    def start(i, now):
        plan, dur = engines[i].schedule()
        if plan:
            engines[i].busy = True
            push(now + dur, 1, (i, plan))
        else:
            engines[i].busy = False

    detector = (HotspotDetector(n_instances, detector_window_s * 1000)
                 if policy == "lmetric_detector" else None)
    turn = tie_offset
    while events:
        now, _, kind, payload = heapq.heappop(events)
        if kind == 0:
            req = payload
            full = req.keys[: req.input_len // BLOCK]
            hits = [x.hits(full) for x in idx]
            sizes = [len(x.lru) for x in idx]
            if len(full) >= CLASS_BLOCKS:
                req.holders = sum(h >= CLASS_BLOCKS for h in hits)
            allowed = None
            if detector:
                detector.observe(req, now)
                allowed = detector.allowed(req, hits, bs, queued)
            if policy == "smetric":
                i = smetric.choose(cost, bs, qcost, hits, req, turn)
            elif policy == "pin" and req.parent in placed:
                i = placed[req.parent]
                req.stuck = True
            else:
                i = choose("lmetric" if policy == "pin" else policy, bs, queued, hits, req, turn, sizes, allowed)
            turn += 1
            placed[req.rid] = i
            new_cost[req.rid] = smetric.prefill_cost(cost, req.input_len, hits[i] * BLOCK)
            qcost[i] += new_cost[req.rid]
            req.instance, req.predicted_hit = i, hits[i] * BLOCK
            new_tokens[req.rid] = req.input_len - hits[i] * BLOCK
            bs[i] += 1
            queued[i] += new_tokens[req.rid]
            idx[i].dispatched(full)
            full_keys[req.rid] = full
            engines[i].waiting.append(req)
            if not engines[i].busy:
                start(i, now)
        elif kind == 1:
            i, plan = payload

            def first(req, i=i):
                queued[i] -= new_tokens[req.rid]
                qcost[i] -= new_cost[req.rid]
                idx[i].prefilled(full_keys.pop(req.rid))

            def done(req, i=i):
                bs[i] -= 1

            engines[i].finish_step(plan, now, first, done)
            start(i, now)
        else:
            for x in idx:
                x.trim()
    return requests


def summarize(requests, cost: Cost):
    ttft = sorted(r.first_ms - r.arrival_ms + cost.overhead for r in requests)
    pct = lambda q: ttft[min(len(ttft) - 1, int(q * len(ttft)))]
    hit = [r.engine_hit or 0 for r in requests]
    over = [r for r in requests if r.predicted_hit > (r.engine_hit or 0) + BLOCK]
    tpot = [(r.done_ms - r.first_ms) / (r.output_len - 1) for r in requests if r.output_len > 1]
    return dict(n=len(requests), ttft_mean=statistics.fmean(ttft), p50=pct(.5), p90=pct(.9),
                p99=pct(.99), tpot_mean=statistics.fmean(tpot), cached_fraction=sum(hit) / max(1, sum(r.input_len for r in requests)),
                over_predicted=len(over),
                over_mean_tokens=statistics.fmean(r.predicted_hit - (r.engine_hit or 0) for r in over) if over else 0,
                preemptions=sum(r.preempted for r in requests),
                per_instance=[sum(r.instance == i for r in requests)
                              for i in range(max(r.instance for r in requests) + 1)])


def load_trace(path, start_s, duration_s, time_scale, max_input=8192, max_output=256,
               trace_block=BLOCK, timestamp_unit_s=1.0, parent_coverage=0.5):
    """Qwen-Bailian format by default. Mooncake traces hash 512-token blocks
    and stamp milliseconds: pass ``trace_block=512, timestamp_unit_s=1e-3``;
    each trace block is split into 16-token keys (h, 0), (h, 1), ...

    Sessions are rebuilt from prompts alone, as SMetric does offline
    (its Appendix A, without the user and turn fields): a request's parent is
    the latest earlier request in the window whose full-block prompt is the
    longest prefix of this prompt, kept only if it covers at least
    ``parent_coverage`` of this prompt. Without the threshold, a shared
    system prompt would make almost every request a "follow-up"."""
    import hashlib
    out = []
    split = trace_block // BLOCK
    prefixes: dict = {}     # running hash of a full-block prompt -> rid
    with open(path) as stream:
        for line_no, line in enumerate(stream):
            row = json.loads(line)
            t = row["timestamp"] * timestamp_unit_s - start_s
            if t < 0:
                continue
            if t >= duration_s:
                break
            n = min(row["input_length"], max_input)
            if split == 1:
                blocks = row["hash_ids"][: math.ceil(n / BLOCK)]
            else:
                blocks = [h * split + j for h in row["hash_ids"] for j in range(split)][: math.ceil(n / BLOCK)]
            cls = hashlib.sha1(",".join(map(str, blocks[:CLASS_BLOCKS])).encode()).hexdigest()[:12]
            rid = f"c{row.get('chat_id', line_no)}"
            hashes = row["hash_ids"]
            full = row["input_length"] // trace_block
            running, chain = 0, []
            for h in hashes[: full + 1]:
                running = hash((running, h))
                chain.append(running)
            parent, matched = None, 0
            for k in range(len(chain), 0, -1):
                if chain[k - 1] in prefixes:
                    parent, matched = prefixes[chain[k - 1]], k
                    break
            if matched * trace_block < parent_coverage * row["input_length"]:
                parent, matched = None, 0
            if full:
                prefixes[chain[full - 1]] = rid
            req = SimRequest(rid, t / time_scale * 1000, blocks[: n // BLOCK], n,
                             max(1, min(row["output_length"], max_output)), cls)
            req.parent, req.est_hit = parent, min(matched * trace_block, n // BLOCK * BLOCK)
            out.append(req)
    out.sort(key=lambda r: r.arrival_ms)
    return out
