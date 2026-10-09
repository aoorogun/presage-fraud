import numpy as np, torch
from presage.federated.fl import secure_aggregate, rdp_epsilon, fed_train, predict


def test_secure_aggregation_masks_cancel():
    ups = [torch.randn(50) for _ in range(5)]
    assert torch.allclose(secure_aggregate(ups), torch.stack(ups).sum(0), atol=1e-3)


def test_epsilon_monotone_in_noise_and_rounds():
    assert rdp_epsilon(2.0, 10) < rdp_epsilon(1.0, 10) < rdp_epsilon(0.5, 10)
    assert rdp_epsilon(1.0, 5) < rdp_epsilon(1.0, 20)


def test_fedavg_learns_separable_signal():
    rng = np.random.default_rng(0)
    cl = []
    for _ in range(3):
        X = rng.normal(size=(1500, 6)).astype(np.float32); y = (X[:, 0] + 0.5 * X[:, 1] > 1.2).astype(np.float32); cl.append(dict(X=X, y=y))
    gm, _, info = fed_train(cl, 6, rounds=15, local_epochs=3)
    Xt = rng.normal(size=(2000, 6)).astype(np.float32); yt = (Xt[:, 0] + 0.5 * Xt[:, 1] > 1.2)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(yt, predict(gm, Xt)) > 0.9
