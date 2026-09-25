"""Theorem A (mean-predictor certificate) and Proposition B (path-wise flip bound)."""
import math

import pytest
import torch

from src.certify import certify_mean, clopper_pearson_upper, model_lipschitz, path_flip_bound
from src.certify.mean_certificate import project_ball
from src.models.sode_guard import SODEGuard, SODEGuardConfig


def certifiable_model(K=5, dt=0.05):
    torch.manual_seed(0)
    m = SODEGuard(SODEGuardConfig(num_classes=K, certifiable=True, dt=dt, hidden_dim=32,
                                  drift_hidden=64, diff_hidden=64, noise_dim=4))
    m.eval()
    with torch.no_grad():                       # settle the spectral-norm power iteration
        m.train(); [m(torch.randn(8, 83)) for _ in range(20)]; m.eval()
    return m


def test_lipschitz_requires_certifiable_encoder():
    with pytest.raises(ValueError):
        model_lipschitz(SODEGuard(SODEGuardConfig(num_classes=5)))


def test_lemma1_mean_square_bound_holds():
    """E||X_T(x+δ) − X_T(x)||² ≤ L_state² ||δ||² under the synchronous coupling."""
    m = certifiable_model()
    lip = model_lipschitz(m)
    x = torch.randn(16, 83)
    delta = torch.randn(16, 83)
    delta = 0.05 * delta / delta.norm(dim=1, keepdim=True)
    with torch.no_grad():
        h0, h1 = m.encode(x), m.encode(x + delta)
        ms = torch.stack([(m._integrate(h1, seed=s) - m._integrate(h0, seed=s)).pow(2).sum(-1)
                          for s in range(64)]).mean(0)
    assert (ms.sqrt() <= lip.L_state * 0.05 + 1e-6).all()


def test_theorem_a_lipschitz_of_mean_logits():
    m = certifiable_model()
    lip = model_lipschitz(m)
    x = torch.randn(16, 83)
    seeds = list(range(256))
    for scale in (1e-3, 1e-2, 1e-1):
        d = torch.randn(16, 83)
        d = scale * d / d.norm(dim=1, keepdim=True)
        with torch.no_grad():
            diff = (m.forward_mean(x + d, seeds=seeds) - m.forward_mean(x, seeds=seeds)).norm(dim=1)
        assert (diff <= lip.L * scale * 1.0001).all()


def test_ball_projection_is_nonexpansive():
    a, b = torch.randn(100, 7) * 5, torch.randn(100, 7) * 5
    assert ((project_ball(a, 2.0) - project_ball(b, 2.0)).norm(dim=1) <= (a - b).norm(dim=1) + 1e-6).all()


class _MockModel:
    """Paths with a clear margin for class 0: logits = [3, 0, ..., 0] + noise."""
    class cfg:
        num_classes = 5
        feature_dim = 83

    def eval(self):
        return self

    def sample_logits(self, x, seeds):
        g = torch.Generator().manual_seed(len(seeds))
        base = torch.zeros(x.shape[0], len(seeds), 5)
        base[..., 0] = 3.0
        return base + 0.1 * torch.randn(base.shape, generator=g)


@pytest.mark.parametrize("bound", ["hoeffding", "bernstein"])
def test_certify_mean_margin_and_radius(bound):
    cert = certify_mean(_MockModel(), torch.zeros(4, 83), L=2.0, B=5.0, n=2048, bound=bound)
    assert (cert.prediction == 0).all()
    assert (cert.margin_lcb > 0).all() and (cert.margin_lcb < 3.0).all()
    assert torch.allclose(cert.radius_l2, cert.margin_lcb / (math.sqrt(2) * 2.0))
    assert torch.allclose(cert.radius_linf, cert.radius_l2 / math.sqrt(83))


def test_bernstein_tighter_than_hoeffding_for_low_variance():
    h = certify_mean(_MockModel(), torch.zeros(2, 83), L=1.0, B=5.0, n=2048, bound="hoeffding")
    b = certify_mean(_MockModel(), torch.zeros(2, 83), L=1.0, B=5.0, n=2048, bound="bernstein")
    assert (b.margin_lcb > h.margin_lcb).all()


def test_clopper_pearson_upper():
    assert clopper_pearson_upper(0, 100, 0.05) == pytest.approx(1 - 0.05 ** (1 / 100), rel=1e-6)
    assert clopper_pearson_upper(100, 100, 0.05) == 1.0
    assert clopper_pearson_upper(10, 100, 0.05) > 0.1


def test_path_flip_bound_monotone_in_eps():
    m = certifiable_model()
    lip = model_lipschitz(m)
    x = torch.randn(6, 83)
    kstar = torch.zeros(6, dtype=torch.long)
    small = path_flip_bound(m, x, L=lip.L, eps_l2=1e-4, n=256, k_star=kstar)["flip_bound"]
    large = path_flip_bound(m, x, L=lip.L, eps_l2=1.0, n=256, k_star=kstar)["flip_bound"]
    assert (small <= large + 1e-9).all() and (large <= 1.0).all()
