"""Sanity checks for the DP / certificate stack."""
from __future__ import annotations

import math

import pytest
import torch

from fedloraguard.privacy.dp_sgd import ClipNoiseConfig, clip_and_noise
from fedloraguard.privacy.rdp_accountant import RDPAccountant
from fedloraguard.privacy.sensitivity import (
    SensitivityInputs, gaussian_noise_for_dp, gradient_sensitivity_bound,
)
from fedloraguard.privacy.certified_radius import certified_poisoning_radius


def test_clip_norm_bounds_l2():
    g = [torch.randn(50)]
    cfg = ClipNoiseConfig(clip_norm=1.0, noise_multiplier=0.0, enabled=True)
    out = clip_and_noise(g, cfg)
    assert torch.linalg.norm(torch.cat([t.reshape(-1) for t in out])) <= 1.0 + 1e-5


def test_rdp_compose_grows_monotone():
    acc = RDPAccountant()
    eps0 = acc.get_epsilon(1e-5)
    acc.step(q=0.2, sigma=1.1)
    eps1 = acc.get_epsilon(1e-5)
    acc.step(q=0.2, sigma=1.1)
    eps2 = acc.get_epsilon(1e-5)
    assert eps2 >= eps1 >= eps0


def test_sensitivity_matches_paper_eq11():
    s = gradient_sensitivity_bound(SensitivityInputs(
        clip_norm=1.0, lipschitz=1.0, weight_norm_bound=1.0,
        max_temporal_degree=1, num_layers=1, num_relations=4,
        local_minibatch_size=2,
    ))
    # 2*1*(1*1*1)^1 * sqrt(4) / 2 = 2.0
    assert math.isclose(s, 2.0, rel_tol=1e-6)


def test_certified_radius_monotone_in_margin():
    # Eq. (14) is monotone non-decreasing in the probability margin.
    k_low  = certified_poisoning_radius(0.55, 0.45, epsilon_T=1.0, num_clients=50)
    k_high = certified_poisoning_radius(0.95, 0.05, epsilon_T=1.0, num_clients=50)
    assert k_high >= k_low


def test_certified_radius_decreases_with_epsilon():
    k_tight = certified_poisoning_radius(0.95, 0.05, epsilon_T=0.5, num_clients=50)
    k_loose = certified_poisoning_radius(0.95, 0.05, epsilon_T=5.0, num_clients=50)
    assert k_tight >= k_loose


def test_gaussian_mechanism_is_positive():
    sigma = gaussian_noise_for_dp(sensitivity=1.0, epsilon=0.5, delta=1e-5)
    assert sigma > 0.0
