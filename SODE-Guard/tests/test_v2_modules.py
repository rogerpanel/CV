"""Unit tests for the v2 modules (imbalance, input encoding, robust training,
randomised-smoothing certificate, statistics)."""
import numpy as np
import torch

from src.certify.smoothing_sde import certify_randomized, radius_ceiling
from src.evaluation.stats import holm, mean_ci, paired_bootstrap
from src.models.pipeline import Pipeline, QuantileEncoder
from src.models.sode_guard import SODEGuard, SODEGuardConfig
from src.training.imbalance import ImbalanceLoss, fast_macro_f1, tune_tau
from src.training.robust import gaussian_consistency_loss, make_adversarial


def test_quantile_encoder_monotone_continuous_finite_slope():
    rng = np.random.default_rng(0)
    X = np.c_[rng.lognormal(size=2000), np.where(rng.random(2000) < 0.8, 0.0, rng.normal(size=2000)),
              np.full(2000, 3.0)]
    enc = QuantileEncoder(X)
    grid = torch.linspace(-20, 20, 4001).unsqueeze(1).repeat(1, 3)
    y = enc(grid)
    assert torch.isfinite(y).all()
    assert (y[1:] - y[:-1] >= -1e-6).all()                       # monotone in every feature
    assert (y[1:] - y[:-1]).abs().max() < 1.0                     # no jumps on a 0.01 grid
    assert enc.max_slope() < 1e3


def test_logit_adjusted_loss_and_macro_f1():
    counts = torch.tensor([900, 90, 10])
    z = torch.zeros(4, 3)
    y = torch.tensor([0, 1, 2, 2])
    la = ImbalanceLoss("la", counts)(z, y)
    ce = ImbalanceLoss("ce", counts)(z, y)
    assert la > ce                    # rare targets are penalised more under the prior shift
    yy, pp = np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2])
    from sklearn.metrics import f1_score
    assert abs(fast_macro_f1(yy, pp, 3) - f1_score(yy, pp, average="macro")) < 1e-9


def test_tune_tau_recovers_rare_class():
    prior = torch.tensor([0.9, 0.1])
    y = np.array([0] * 90 + [1] * 10)
    logits = torch.zeros(1, 100, 2)
    logits[0, :, 0] = 0.5                                         # a prior-biased scorer
    logits[0, :90, 1] = -1.0
    logits[0, 90:, 1] = 0.4                                       # rare class slightly below
    tau, _ = tune_tau(logits, y, prior)
    assert tau > 0


def test_pipeline_attack_and_consistency_loss_run():
    torch.manual_seed(0)
    m = Pipeline(SODEGuard(SODEGuardConfig(feature_dim=6, num_classes=3, hidden_dim=16, drift_hidden=16,
                                           diff_hidden=16, noise_dim=4)), None, 3)
    x, y = torch.randn(8, 6), torch.randint(0, 3, (8,))
    f = lambda z: m.forward_mean(z, n_paths=2)
    xa = make_adversarial(f, x, y, method="pgd", eps=0.03, steps=2)
    assert (xa - x).abs().max() <= 0.03 + 1e-6
    xn = make_adversarial(f, x, y, method="nfgsm", eps=0.03)
    assert (xn - x).abs().max() <= 3 * 0.03 + 1e-6
    loss = gaussian_consistency_loss(f, x, y, ImbalanceLoss("ce", torch.tensor([3, 3, 2])), sigma=0.25)
    loss.backward()
    assert m.sample_logits(x, [0, 1]).shape == (8, 2, 3)


def test_smoothing_certificate_on_constant_classifier():
    f = lambda z: torch.nn.functional.one_hot(torch.zeros(len(z), dtype=torch.long), 3).float()
    pred, r = certify_randomized(f, torch.zeros(4, 5), sigma=0.25, K=3, n0=20, n=500)
    assert (pred == 0).all()
    assert np.allclose(r, radius_ceiling(0.25, 500))


def test_stats():
    assert mean_ci([1.0, 1.0, 1.0])["mean"] == 1.0
    y = np.zeros(200, dtype=int)
    a, b = np.zeros(200), np.zeros(200)
    a[:50] = 1
    res = paired_bootstrap([(y, a, b, None)], lambda yy, pp, ww: float(pp.mean()), n_boot=200)
    assert res["diff"] > 0 and res["p"] < 0.05
    adj = holm({"x": 0.01, "y": 0.04, "z": 0.5})
    assert adj["x"] == 0.03 and adj["y"] == 0.08 and adj["z"] == 0.5


def test_flow_constraints_respect_mask_domain_integers_and_budget():
    from src.attacks.constrained import FlowConstraints, constrained_pgd
    rng = np.random.default_rng(0)
    raw = np.c_[rng.integers(0, 50, 500), rng.integers(0, 3, 500), rng.exponential(5.0, 500), rng.integers(1, 65535, 500)]
    names = ["Total Fwd Packet", "Bwd Packets/s", "Flow Duration", "Dst Port"]
    mu, sd = raw.mean(0), raw.std(0)
    C = FlowConstraints(names, mu, sd, raw.astype(float), eps=0.5)
    assert C.summary["n_mutable"] == 2 and C.summary["n_integer"] == 3
    x = torch.tensor((raw[:20] - mu) / sd, dtype=torch.float32)
    xa = C.finalize(x + 0.5 * torch.sign(torch.randn_like(x)), x)
    d = xa - x
    assert d.abs().max() <= 0.5 + 1e-5                      # budget
    assert torch.all(d[:, 1] == 0) and torch.all(d[:, 3] == 0)  # Bwd and port fixed
    raw_a = xa.numpy() * sd + mu
    assert np.allclose(raw_a[:, 0], np.round(raw_a[:, 0]), atol=1e-3)   # integer counts stay integers
    assert (raw_a[:, 2] >= raw[:, 2].min() - 1e-4).all()      # domain
    f = torch.nn.Linear(4, 3)
    xp = constrained_pgd(f, x, torch.zeros(20, dtype=torch.long), C, steps=3)
    assert torch.all((xp - x)[:, 3] == 0)


def test_conformal_coverage_and_alarm_far():
    from src.evaluation.operations import alarm_threshold, conformal_sets, fit_temperature
    rng = np.random.default_rng(1)
    K, n = 4, 4000
    y = rng.integers(0, K, n)
    z = rng.normal(size=(n, K)); z[np.arange(n), y] += 1.5
    p = torch.softmax(torch.tensor(z) / 2.0, 1).numpy()
    T = fit_temperature(torch.tensor(z, dtype=torch.float32) / 2.0, torch.tensor(y))
    assert 0.2 < T < 2.0
    r = conformal_sets(p[:2000], y[:2000], p[2000:], y[2000:], alpha=0.1, classwise=True)
    assert r["coverage"] >= 0.87 and r["min_class_coverage"] >= 0.84
    al = alarm_threshold(p[:2000], y[:2000], p[2000:], y[2000:], benign=0, far=0.05)
    assert al["test_far"] <= 0.08
