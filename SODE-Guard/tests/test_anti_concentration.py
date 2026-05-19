"""Anti-concentration certificate behaviour."""
import torch
from src.regularizers.anti_concentration import (
    AntiConcentrationLoss,
    anti_concentration_certificate,
    certified_radius,
)


def test_ac_loss_decreases_with_larger_margin():
    loss = AntiConcentrationLoss(chaos_degree=4, beta_grid=(0.05,))
    small = torch.zeros(4, 8, 10); small[:, :, 0] = 0.1
    large = torch.zeros(4, 8, 10); large[:, :, 0] = 3.0
    l_small = loss(small).item()
    l_large = loss(large).item()
    assert l_large <= l_small


def test_certificate_monotone_in_eps():
    m = torch.ones(16) * 0.2
    b1 = anti_concentration_certificate(m, lipschitz=1.0, epsilon=0.01)
    b2 = anti_concentration_certificate(m, lipschitz=1.0, epsilon=0.05)
    assert (b1 >= b2).all()


def test_radius_positive_when_margin_positive():
    r = certified_radius(torch.tensor([0.4, 0.6]), lipschitz=1.0,
                         chaos_degree=4, confidence=0.95)
    assert (r > 0).all()
