"""Anti-concentration regulariser behaviour."""
import torch
from src.regularizers.anti_concentration import AntiConcentrationLoss


def test_ac_loss_decreases_with_larger_margin():
    loss = AntiConcentrationLoss(chaos_degree=4, beta_grid=(0.05,))
    small = torch.zeros(4, 8, 10); small[:, :, 0] = 0.1
    large = torch.zeros(4, 8, 10); large[:, :, 0] = 3.0
    assert loss(large).item() <= loss(small).item()


def test_ac_loss_is_differentiable():
    logits = torch.randn(4, 8, 10, requires_grad=True)
    AntiConcentrationLoss()(logits).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
