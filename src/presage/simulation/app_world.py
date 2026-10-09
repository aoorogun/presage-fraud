"""APP-fraud / money-mule world simulator (discrete-event).

WHY: no credible open dataset carries APP labels together with device, session and
beneficiary-registration signals. This simulator generates a *synthetic* multi-bank
payment world. Everything produced here is SYNTHETIC and must be reported as such.

What is generated
  * customers (with bank, devices, usual payees, spending scale, activity rate)
  * legitimate payments, including hard negatives (large first-time transfers, new phones)
  * eight scam typologies producing authorised pushes to mule accounts
  * layered mule networks that span banks (collect -> layer -> cash-out), with
    pass-through forwarding, dormant "sleeper" mules and mule background activity
  * four drift types (sudden, gradual, recurring, feature) with a documented schedule
  * per-fraud label-report delays so label-delay experiments can be run later

Output: a transactions DataFrame plus an accounts DataFrame (see `simulate`).
"""
from __future__ import annotations
import heapq
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
from presage.utils.common import get_logger

log = get_logger("sim")
DAY = 86400.0

SCAM_TYPES = ["investment", "purchase", "romance", "impersonation", "bank_impersonation",
              "invoice_redirection", "ceo_fraud", "family_impersonation"]


@dataclass
class SimConfig:
    seed: int = 42
    n_customers: int = 4000
    n_payees: int = 10000
    n_banks: int = 5
    days: int = 120
    legit_rate_mean: float = 0.62          # mean legit payments per customer-day
    scam_rate_per_day: float = 3.0         # mean new scam campaigns (victims) per day, before drift multipliers
    n_mule_networks: int = 120
    # drift schedule (days). set to None to disable a drift type
    sudden_drift_day: int | None = 100     # fraudsters switch from one large transfer to several smaller ones
    gradual_drift: bool = True             # mean scam fraction of balance shrinks linearly across the horizon
    recurring_windows: tuple = ((20, 35), (90, 105))  # seasonal investment/impersonation campaigns
    feature_drift: bool = True             # fraud channel mix moves desktop -> mobile from day 50 to 110
    hard_negative_rate: float = 0.008      # legit large first-time transfers (house deposits, family)
    name: str = "presage_sim_v1"


def _zipf_weights(n: int, a: float, rng) -> np.ndarray:
    w = 1.0 / np.arange(1, n + 1) ** a
    w = w / w.sum()
    return w[rng.permutation(n)]


def simulate(cfg: SimConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    C, P = cfg.n_customers, cfg.n_payees
    horizon = cfg.days * DAY

    # ---------------- accounts ----------------
    # layout: [0,C) customers | [C,C+P) payees | mules afterwards
    net_sizes = []
    mule_tier, mule_net = [], []
    for n in range(cfg.n_mule_networks):
        t1, t2, t3 = rng.integers(2, 6), rng.integers(3, 7), rng.integers(2, 4)
        net_sizes.append((t1, t2, t3))
    M = int(sum(sum(s) for s in net_sizes))
    N = C + P + M
    bank = rng.integers(0, cfg.n_banks, size=N)
    bank[:C] = rng.choice(cfg.n_banks, size=C, p=np.array([.30, .25, .2, .15, .10])[:cfg.n_banks] / np.array([.30, .25, .2, .15, .10])[:cfg.n_banks].sum())
    acct_type = np.array(["customer"] * C + ["payee"] * P + ["mule"] * M, dtype=object)
    created_ts = np.zeros(N)
    created_ts[:C] = -rng.exponential(900 * DAY, C)                   # customers: long-standing
    created_ts[C:C + P] = -rng.exponential(700 * DAY, P)
    new_biz = rng.random(P) < 0.22                                    # 5% of legit payees are recently created businesses
    created_ts[C:C + P][new_biz] = -rng.exponential(70 * DAY, new_biz.sum())
    mule_tier = np.zeros(N, int); mule_net = -np.ones(N, int)
    net_nodes: list[list[np.ndarray]] = []
    idx = C + P
    net_start = rng.uniform(5, cfg.days - 20, cfg.n_mule_networks)
    net_life = rng.uniform(12, 35, cfg.n_mule_networks)
    for n, (t1, t2, t3) in enumerate(net_sizes):
        tiers = []
        for tier, sz in zip((1, 2, 3), (t1, t2, t3)):
            ids = np.arange(idx, idx + sz); idx += sz
            mule_tier[ids] = tier; mule_net[ids] = n
            created_ts[ids] = net_start[n] * DAY - rng.exponential(18 * DAY, sz)  # mules are young at activation
            tiers.append(ids)
        net_nodes.append(tiers)
    # networks assigned bank membership at random => they span institutions
    # Realism: (a) ~40% of tier-1 collection accounts are recruited GENUINE customer accounts (old, normal history);
    # (b) ~30% of deeper-tier mules are 'aged' accounts bought/dormant for months (not young).
    recruited = set()
    for n in range(cfg.n_mule_networks):
        t1 = net_nodes[n][0]
        k = int(np.ceil(0.6 * len(t1)))
        for j in range(k):
            cust = int(rng.integers(0, C))
            if cust in recruited:
                continue
            recruited.add(cust)
            old = t1[j]
            mule_tier[old] = 0; mule_net[old] = -1                       # orphaned slot becomes an inert account
            t1[j] = cust
            mule_tier[cust] = 1; mule_net[cust] = n
        for tier_ids in net_nodes[n]:
            for a in tier_ids:
                if rng.random() < 0.3:
                    created_ts[a] = -rng.exponential(300 * DAY)

    # ---------------- customers ----------------
    scale = np.exp(rng.normal(np.log(45), 0.6, C))                    # typical payment size
    bal0 = np.exp(rng.normal(8.4, 0.9, C))
    rate = rng.gamma(2.0, cfg.legit_rate_mean / 2.0, C)               # payments/day
    pref_hour = rng.normal(14, 3.5, C)
    n_dev = 1 + (rng.random(C) < 0.35)
    dev_id = [[c * 3 + k for k in range(n_dev[c])] for c in range(C)]
    next_dev = C * 3 + 10_000_000
    ip_home = rng.integers(0, C * 2, C)
    susceptibility = rng.beta(1.2, 6.0, C)                            # latent victim propensity
    pay_sets = []
    pw = _zipf_weights(P, 0.95, rng)
    for c in range(C):
        k = 2 + rng.poisson(5)
        pay_sets.append(set((C + rng.choice(P, k, replace=False, p=pw)).tolist()))
    # shared-device mule infrastructure: some mule networks share a device farm
    net_device = {n: 20_000_000 + n * 7 for n in range(cfg.n_mule_networks)}
    balance = bal0.copy()

    # ---------------- event heap ----------------
    ev: list = []
    cnt = 0

    def push(t, kind, payload):
        nonlocal cnt
        cnt += 1
        heapq.heappush(ev, (t, cnt, kind, payload))

    # legit events
    for c in range(C):
        n_ev = rng.poisson(rate[c] * cfg.days)
        ts = np.sort(rng.uniform(0, horizon, n_ev))
        hrs = (pref_hour[c] + rng.normal(0, 3.2, n_ev)) % 24
        ts = np.floor(ts / DAY) * DAY + hrs * 3600
        for t in ts:
            if 0 <= t < horizon:
                push(float(t), "legit", c)
    # salary credits
    for c in range(C):
        for d in range(0, cfg.days, 30):
            push((d + rng.integers(0, 3)) * DAY + 8 * 3600, "salary", c)

    # scam campaigns
    def day_multiplier(d: float) -> float:
        m = 1.0
        for a, b in cfg.recurring_windows:
            if a <= d <= b: m *= 2.2
        return m

    camp_id = 0
    for d in range(cfg.days):
        lam = cfg.scam_rate_per_day * day_multiplier(d)
        for _ in range(rng.poisson(lam)):
            v = int(rng.choice(C, p=susceptibility / susceptibility.sum()))
            kind_w = np.array([.18, .14, .10, .16, .16, .08, .08, .10])
            if any(a <= d <= b for a, b in cfg.recurring_windows):
                kind_w = kind_w * np.array([2.2, 1, 1, 2.0, 1, 1, 1, 1])
            st = SCAM_TYPES[int(rng.choice(8, p=kind_w / kind_w.sum()))]
            t0 = d * DAY + rng.uniform(7, 22) * 3600
            push(t0, "scam_start", (v, st, camp_id, d))
            camp_id += 1

    # ---------------- processing ----------------
    rows: list[tuple] = []
    known_dev = [set(dv) for dv in dev_id]
    last_in = {}      # mule/payee last receipt time
    mule_seen_victims = {}  # camp state for repeat payments
    txn = 0
    schedule_mule_net_usage = {n: 0 for n in range(cfg.n_mule_networks)}

    slow_net = rng.random(cfg.n_mule_networks) < 0.3           # 30% 'low-and-slow' networks: many small pieces, long holds

    def hold_time(n):
        if slow_net[n]:
            return float(rng.exponential(3 * DAY)) + 3600
        u = rng.random()
        if u < 0.5: return float(rng.exponential(3 * 3600)) + 120
        if u < 0.8: return float(rng.exponential(1.5 * DAY)) + 3600
        return float(rng.exponential(4 * DAY)) + 3600

    def active_nets(d):
        return [n for n in range(cfg.n_mule_networks) if net_start[n] <= d <= net_start[n] + net_life[n]]

    def emit(t, src, dst, amt, ttype, cust, dev, ip, sess_s, fail, chan, scam, camp, net, delay=np.nan, fraud=0, bb=np.nan):
        nonlocal txn
        rows.append((txn, t, src, dst, amt, ttype, cust, dev, ip, sess_s, fail, chan, scam, camp, net, delay, fraud, bb))
        txn += 1

    while ev:
        t, _, kind, pl = heapq.heappop(ev)
        d = t / DAY
        if kind == "salary":
            balance[pl] += float(np.exp(rng.normal(7.6, 0.5)))
            continue

        if kind == "legit":
            c = pl
            if rng.random() < cfg.hard_negative_rate:   # hard negative branch
                new_ben = True; hard = True
            else:
                hard = False
                new_ben = rng.random() < 0.09
            if new_ben:
                dst = int(C + rng.choice(P, p=pw))
                if hard:
                    amt = float(balance[c] * rng.uniform(0.15, 0.7))     # house deposit / family transfer
                else:
                    amt = float(scale[c] * np.exp(rng.normal(0.4, 0.8)))
                pay_sets[c].add(dst)
            else:
                dst = int(rng.choice(list(pay_sets[c])))
                amt = float(scale[c] * np.exp(rng.normal(0, 0.7)))
            amt = max(1.0, min(amt, max(balance[c], 0.0) * 0.95 + 5))
            newdev = rng.random() < 0.006
            if newdev:
                dv = next_dev; next_dev += 1; known_dev[c].add(dv); dev_id[c].append(dv)
            else:
                dv = int(rng.choice(dev_id[c]))
            chan = "mobile" if rng.random() < (0.55 if d < 50 else 0.55 + 0.1 * min(1, (d - 50) / 60)) else "desktop"
            sess = float(np.exp(rng.normal(np.log(75 if not new_ben else 130), 0.55)))
            fail = int(rng.poisson(0.22 if new_ben else 0.01))
            ip = int(ip_home[c]) if rng.random() < 0.93 else int(rng.integers(0, C * 2))
            bb = float(balance[c]); balance[c] -= amt
            emit(t, c, dst, amt, "legit", c, dv, ip, sess, fail, chan, "", -1, -1, bb=bb)
            continue

        if kind == "scam_start":
            v, st, cid, d0 = pl
            avail = active_nets(d0)
            if not avail:
                continue
            n = int(rng.choice(avail))
            tier1 = net_nodes[n][0]
            # ---- scam parameterisation (typology-specific) ----
            sudden = cfg.sudden_drift_day is not None and d0 >= cfg.sudden_drift_day
            grad = 1.0 - 0.45 * (d0 / cfg.days) if cfg.gradual_drift else 1.0
            if st == "investment":
                npay, frac, newb, gap = int(rng.integers(1, 5)), (0.25, 0.8), 0.97, (0.5, 5)
            elif st == "purchase":
                npay, frac, newb, gap = 1, (0.01, 0.12), 0.96, (0, 0)
            elif st == "romance":
                npay, frac, newb, gap = int(rng.integers(3, 9)), (0.04, 0.2), 0.95, (2, 9)
            elif st == "impersonation":
                npay, frac, newb, gap = int(rng.integers(1, 3)), (0.4, 0.9), 0.97, (0.02, 0.5)
            elif st == "bank_impersonation":
                npay, frac, newb, gap = 1, (0.6, 0.97), 0.98, (0, 0)
            elif st == "invoice_redirection":
                npay, frac, newb, gap = 1, (0.15, 0.5), 0.92, (0, 0)
            elif st == "ceo_fraud":
                npay, frac, newb, gap = int(rng.integers(1, 3)), (0.3, 0.7), 0.95, (0.1, 1)
            else:  # family_impersonation
                npay, frac, newb, gap = int(rng.integers(1, 3)), (0.05, 0.3), 0.93, (0.05, 1)
            lo, hi = frac
            f_total = rng.uniform(lo, hi) * grad
            if sudden and npay == 1 and st in ("impersonation", "bank_impersonation", "investment", "ceo_fraud"):
                npay = int(rng.integers(3, 7))                         # SUDDEN DRIFT: structuring into smaller transfers
                gap = (0.01, 0.4)
            if sudden:
                f_total = f_total * 0.9
            per = f_total / npay
            # channel mix (feature drift): fraud moves desktop -> mobile
            p_mobile = 0.2 if not cfg.feature_drift else 0.2 + 0.6 * np.clip((d0 - 50) / 60, 0, 1)
            chan = "mobile" if rng.random() < p_mobile else "desktop"
            remote_access = rng.random() < (0.35 if st in ("bank_impersonation", "impersonation") else 0.08)
            dv = int(rng.choice(dev_id[v]))
            if rng.random() < 0.12:                                    # attacker-induced new device / reset
                dv = next_dev; next_dev += 1; known_dev[v].add(dv); dev_id[v].append(dv)
            tt = t
            delay_report = float(rng.choice([3600, DAY, 3 * DAY, 7 * DAY, 30 * DAY], p=[.10, .35, .30, .18, .07]))
            mule = int(rng.choice(tier1))
            for k in range(npay):
                if k > 0:
                    tt += rng.uniform(*gap) * DAY
                if rng.random() < 0.35 and k > 0:
                    mule = int(rng.choice(tier1))
                push(tt, "scam_pay", (v, mule, per, st, cid, n, dv, chan, remote_access, newb, delay_report))
            continue

        if kind == "scam_pay":
            v, mule, per, st, cid, n, dv, chan, remote, newb, delay_report = pl
            amt = float(max(10.0, max(balance[v], 0.0) * per * rng.uniform(0.9, 1.1)))
            amt = max(5.0, min(amt, max(balance[v], 0.0) * 0.98))
            sess = float(np.exp(rng.normal(np.log(260 if st in ("bank_impersonation", "impersonation") else 150), 0.5)))
            fail = int(rng.poisson(0.5 if rng.random() < newb * 0.7 else 0.1))
            ip = int(ip_home[v]) if not remote else int(rng.integers(0, C * 2))
            bb = float(balance[v]); balance[v] -= amt
            emit(t, v, mule, amt, "scam", v, dv, ip, sess, fail, chan, st, cid, n, delay=delay_report, fraud=1, bb=bb)
            # mule receives -> forward (layering)
            last_in[mule] = t
            push(t + hold_time(n), "mule_forward", (mule, amt * float(rng.uniform(0.7, 0.99)), n, 1, cid))
            continue

        if kind == "mule_forward":
            src, amt, n, tier, cid = pl
            if tier >= 3:
                # cash-out to an external sink (payee-like) account
                dst = int(C + rng.integers(0, P))
                ttype = "cashout"
            else:
                dst = int(rng.choice(net_nodes[n][tier]))                # next layer
                ttype = "mule_forward"
            parts = int(rng.integers(3, 7)) if slow_net[n] else int(rng.integers(1, 3))
            for k in range(parts):
                a = amt / parts
                emit(t + k * 600, src, dst, float(a), ttype, src, net_device[n], int(rng.integers(0, C * 2)), 30.0, 0, "mobile", "", cid, n, delay=np.nan, fraud=0)
            if tier < 3:
                push(t + hold_time(n), "mule_forward", (dst, amt * float(rng.uniform(0.7, 0.99)), n, tier + 1, cid))
            continue

    df = pd.DataFrame(rows, columns=["txn_id", "ts", "src", "dst", "amount", "txn_type", "customer_id", "device_id", "ip_id",
                                     "session_s", "failed_ben_adds", "channel", "scam_type", "campaign_id", "net_id",
                                     "report_delay_s", "is_app_fraud", "balance_before"])
    df = df[df.ts < horizon].sort_values(["ts", "txn_id"]).reset_index(drop=True)
    df["txn_id"] = np.arange(len(df))

    # mule background activity (low-and-slow camouflage): small ordinary payments from random customers to mules
    n_bg = int(0.012 * len(df)) + 400
    bg_t = rng.uniform(0, horizon, n_bg)
    bg_m = np.where(mule_tier > 0)[0]
    bg_dst = rng.choice(bg_m, n_bg)
    ok = (bg_t / DAY > (net_start[mule_net[bg_dst]] - 15)) & (mule_net[bg_dst] >= 0) & (bg_dst >= C)   # camouflage inflow before/while active
    bg = pd.DataFrame(dict(ts=bg_t[ok], src=rng.integers(0, C, ok.sum()), dst=bg_dst[ok]))
    bg["amount"] = np.exp(rng.normal(3.2, 0.6, len(bg)))
    bg["txn_type"] = "legit"; bg["device_id"] = bg.src * 3; bg["ip_id"] = ip_home[bg.src.values]
    bg["customer_id"] = bg.src; bg["session_s"] = np.exp(rng.normal(np.log(120), 0.5, len(bg)))
    bg["failed_ben_adds"] = 0; bg["channel"] = "mobile"; bg["scam_type"] = ""; bg["campaign_id"] = -1; bg["net_id"] = -1
    bg["report_delay_s"] = np.nan; bg["is_app_fraud"] = 0; bg["balance_before"] = np.nan
    # mule outbound camouflage: ordinary small payments from mules to payees
    bo = pd.DataFrame(dict(ts=bg_t[ok][: len(bg) // 2], src=bg_dst[ok][: len(bg) // 2]))
    bo["dst"] = C + rng.integers(0, P, len(bo)); bo["amount"] = np.exp(rng.normal(3.0, 0.7, len(bo)))
    bo["txn_type"] = "legit"; bo["device_id"] = 30_000_000 + bo.src; bo["ip_id"] = rng.integers(0, C * 2, len(bo))
    bo["customer_id"] = bo.src; bo["session_s"] = np.exp(rng.normal(np.log(60), 0.4, len(bo))); bo["failed_ben_adds"] = 0
    bo["channel"] = "mobile"; bo["scam_type"] = ""; bo["campaign_id"] = -1; bo["net_id"] = -1; bo["report_delay_s"] = np.nan
    bo["is_app_fraud"] = 0; bo["balance_before"] = np.nan
    # legitimate payee/business activity: payees also send money (so 'has outflow' is not a trivial mule tell).
    # A subset behave like processors / marketplaces with high fan-in AND fan-out and fast pass-through.
    n_pp = int(0.18 * len(df))
    proc = C + rng.choice(P, int(0.04 * P), replace=False)           # processors / marketplaces
    n_proc = int(0.05 * len(df))
    pr_in = pd.DataFrame(dict(ts=rng.uniform(0, horizon, n_proc), src=rng.integers(0, C, n_proc), dst=rng.choice(proc, n_proc)))
    pr_in["amount"] = np.exp(rng.normal(3.8, 0.9, n_proc))
    pr_out = pd.DataFrame(dict(ts=pr_in.ts + rng.exponential(2 * 3600, n_proc) + 60, src=pr_in.dst.values, dst=C + rng.integers(0, P, n_proc)))
    pr_out["amount"] = pr_in.amount.values * rng.uniform(0.9, 0.99, n_proc)
    pr = pd.concat([pr_in, pr_out], ignore_index=True); pr = pr[pr.ts < horizon]
    for c_, v_ in dict(txn_type="legit", campaign_id=-1, net_id=-1, report_delay_s=np.nan, is_app_fraud=0, balance_before=np.nan, failed_ben_adds=0, channel="mobile", scam_type="").items(): pr[c_] = v_
    pr["device_id"] = 40_000_000 + pr.src; pr["ip_id"] = rng.integers(0, C * 2, len(pr)); pr["customer_id"] = pr.src
    pr["session_s"] = np.exp(rng.normal(np.log(50), 0.4, len(pr)))
    pw_act = _zipf_weights(P, 1.1, rng)
    pp = pd.DataFrame(dict(ts=rng.uniform(0, horizon, n_pp), src=C + rng.choice(P, n_pp, p=pw_act)))
    pp["dst"] = np.where(rng.random(n_pp) < 0.7, C + rng.choice(P, n_pp, p=pw), rng.integers(0, C, n_pp))
    pp["amount"] = np.exp(rng.normal(4.0, 1.0, n_pp)); pp["txn_type"] = "legit"
    pp["device_id"] = 40_000_000 + pp.src; pp["ip_id"] = rng.integers(0, C * 2, n_pp); pp["customer_id"] = pp.src
    pp["session_s"] = np.exp(rng.normal(np.log(50), 0.4, n_pp)); pp["failed_ben_adds"] = 0; pp["channel"] = "mobile"
    pp["scam_type"] = ""; pp["campaign_id"] = -1; pp["net_id"] = -1; pp["report_delay_s"] = np.nan
    pp["is_app_fraud"] = 0; pp["balance_before"] = np.nan
    bg = pd.concat([bg, bo, pp, pr], ignore_index=True)
    df = pd.concat([df.drop(columns="txn_id"), bg], ignore_index=True).sort_values("ts").reset_index(drop=True)
    df.insert(0, "txn_id", np.arange(len(df)))

    # accounts table (attributes known at transaction time: bank, created_ts. Labels kept in separate columns)
    acc = pd.DataFrame(dict(account=np.arange(N), account_type=acct_type, bank=bank, created_ts=created_ts,
                            is_mule=(mule_tier > 0).astype(int), mule_tier=mule_tier, mule_net=mule_net))
    df["bank_src"] = bank[df.src.values]; df["bank_dst"] = bank[df.dst.values]
    df["cross_bank"] = (df.bank_src != df.bank_dst).astype(int)
    df["dst_age_days"] = (df.ts.values - created_ts[df.dst.values]) / DAY
    df["src_age_days"] = (df.ts.values - created_ts[df.src.values]) / DAY
    df["day"] = (df.ts // DAY).astype(int)
    df["src_is_customer"] = (df.src.values < C).astype(int)   # task-A population: payments initiated by customers
    df.attrs["config"] = asdict(cfg)
    log.info("simulated %d txns (%d scam payments = %.3f%%), %d accounts, %d mule accounts",
             len(df), int(df.is_app_fraud.sum()), 100 * df.is_app_fraud.mean(), N, int((mule_tier > 0).sum()))
    return df, acc
