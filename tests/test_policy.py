import numpy as np
from presage.evaluation.policy import optimise_policy, Policy


def test_policy_beats_no_action_and_is_monotone():
    rng = np.random.default_rng(0); n = 20000
    y = (rng.random(n) < 0.01).astype(int); p = np.clip(0.02 + y * 0.6 + rng.normal(0, 0.1, n), 0, 1); amt = rng.uniform(20, 2000, n)
    pol = optimise_policy(p, y, amt, grid=12)
    assert pol.t1 <= pol.t2 <= pol.t3
    assert pol.expected_cost(p, y, amt)["total"] < Policy(2, 2, 2).expected_cost(p, y, amt)["total"]
