"""Cost-optimised intervention policy: approve | soft warning | strong warning + confirmation | hold / analyst review.

Thresholds (T1 <= T2 <= T3) are CHOSEN BY EXPECTED COST on validation data, not hard-coded.
All cost and effectiveness numbers are ASSUMPTIONS (simulated GBP) and must be re-estimated from an operator's own data:
  friction cost per intervention on a legitimate customer, review cost per held payment, and the probability each
  intervention stops a scam (soft warnings stop fewer scams than a hold).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import itertools
import numpy as np

ACTIONS = ["approve", "soft_warning", "strong_warning_confirm", "hold_review"]


@dataclass
class PolicyCosts:
    friction: tuple = (0.0, 0.5, 3.0, 12.0)      # cost imposed on a LEGITIMATE payment by each action (customer friction / delay)
    review: tuple = (0.0, 0.0, 0.0, 8.0)          # analyst cost per action
    stop_prob: tuple = (0.0, 0.30, 0.65, 0.95)    # probability the action prevents a scam (simulated assumption)
    fn_loss_share: float = 1.0                    # share of an unprevented scam's value that is lost


@dataclass
class Policy:
    t1: float; t2: float; t3: float
    costs: PolicyCosts = field(default_factory=PolicyCosts)

    def decide(self, p: np.ndarray) -> np.ndarray:
        return np.digitize(p, [self.t1, self.t2, self.t3])          # 0..3 index into ACTIONS

    def expected_cost(self, p, y, amount) -> dict:
        a = self.decide(p); c = self.costs
        fr = np.array(c.friction)[a] * (y == 0); rv = np.array(c.review)[a]
        loss = np.where(y == 1, amount * c.fn_loss_share * (1 - np.array(c.stop_prob)[a]), 0.0)
        prevented = np.where(y == 1, amount * np.array(c.stop_prob)[a], 0.0)
        return dict(total=float((fr + rv + loss).sum()), friction=float(fr.sum()), review=float(rv.sum()), loss=float(loss.sum()), prevented=float(prevented.sum()),
                    intervention_rate=float((a > 0).mean()), hold_rate=float((a == 3).mean()), counts={ACTIONS[i]: int((a == i).sum()) for i in range(4)})


def optimise_policy(p, y, amount, costs: PolicyCosts | None = None, grid: int = 40) -> Policy:
    """Exhaustive search over a quantile grid of (T1,T2,T3) on VALIDATION data."""
    costs = costs or PolicyCosts()
    qs = np.unique(np.quantile(p, np.concatenate([np.linspace(0.90, 0.999, grid), [0.9995, 0.9999]])))
    best, bp = np.inf, None
    for t1, t2, t3 in itertools.combinations_with_replacement(qs, 3):
        pol = Policy(t1, t2, t3, costs); c = pol.expected_cost(p, y, amount)["total"]
        if c < best: best, bp = c, pol
    return bp
