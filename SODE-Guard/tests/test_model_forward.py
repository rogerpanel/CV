"""End-to-end forward smoke tests for SODE-Guard and baselines."""
import torch
import pytest

from src.models.sode_guard import SODEGuard, SODEGuardConfig
from src.baselines import (
    EGraphSAGEBaseline, RTIDSTransformer, CNN_LSTM,
    IDSGraphMamba, SurrogateIDS7B, SDE_TGNN,
)


@pytest.mark.parametrize("ctor", [
    lambda: SODEGuard(SODEGuardConfig(num_classes=5)),
    lambda: EGraphSAGEBaseline(num_classes=5),
    lambda: RTIDSTransformer(num_classes=5),
    lambda: CNN_LSTM(num_classes=5),
    lambda: IDSGraphMamba(num_classes=5),
    lambda: SurrogateIDS7B(num_classes=5),
    lambda: SDE_TGNN(num_classes=5),
])
def test_forward_shape(ctor):
    model = ctor().eval()
    x = torch.randn(8, 83)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (8, 5)


def test_sode_guard_mc_softmax_sums_to_one():
    model = SODEGuard(SODEGuardConfig(num_classes=5, mc_paths_eval=4)).eval()
    x = torch.randn(4, 83)
    probs = model.forward_mc(x)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_sode_guard_pgd_step_runs():
    from src.attacks.pgd import PGD
    model = SODEGuard(SODEGuardConfig(num_classes=5)).eval()
    x = torch.rand(4, 83); y = torch.tensor([0, 1, 2, 3])
    x_adv = PGD(model, eps=0.05, steps=2)(x, y)
    assert x_adv.shape == x.shape
    assert (x_adv - x).abs().max().item() <= 0.05 + 1e-5
