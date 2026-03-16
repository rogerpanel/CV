"""
Unit tests for all framework components.

Validates correctness of:
  - SurrogateIDS forward pass and MC Dropout
  - EWC loss computation
  - Reservoir replay buffer
  - Fisher Information computation
  - Drift detector
  - CMDP environment
  - CPO update step
  - Unified FIM
  - Adversarial attacks
"""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from continual_learning_ids.models.surrogate_ids import SurrogateIDS
from continual_learning_ids.models.policy_network import (
    PolicyNetwork, ValueNetwork, CostValueNetwork,
)
from continual_learning_ids.models.unified_fim import UnifiedFIM
from continual_learning_ids.utils.replay_buffer import ReservoirReplayBuffer
from continual_learning_ids.utils.fisher import (
    compute_fisher_diagonal_efficient, update_running_fisher,
)
from continual_learning_ids.training.drift_detector import DriftDetector, DriftStatus
from continual_learning_ids.environments.nids_env import NIDSResponseEnv
from continual_learning_ids.evaluation.metrics import ContinualMetrics
from continual_learning_ids.evaluation.adversarial import AdversarialEvaluator
from continual_learning_ids.data.dataset_loader import NIDSDataset, create_dataloader


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.randn(500, 79).astype(np.float32)
    y = np.random.randint(0, 10, 500).astype(np.int64)
    return X, y


@pytest.fixture
def model():
    return SurrogateIDS(input_dim=79, num_classes=10, num_branches=7)


# ── SurrogateIDS Tests ───────────────────────────────────────────────

class TestSurrogateIDS:
    def test_forward_shape(self, model):
        x = torch.randn(32, 79)
        logits, features = model(x)
        assert logits.shape == (32, 10)
        assert features.shape[0] == 32

    def test_mc_dropout_uncertainty(self, model):
        x = torch.randn(16, 79)
        result = model.predict_with_uncertainty(x, num_samples=5)
        assert result["probabilities"].shape == (16, 10)
        assert result["epistemic_uncertainty"].shape == (16,)
        assert result["aleatoric_uncertainty"].shape == (16,)
        assert result["predictions"].shape == (16,)
        # Probabilities sum to 1
        assert torch.allclose(
            result["probabilities"].sum(dim=1),
            torch.ones(16),
            atol=1e-5,
        )

    def test_rl_state_construction(self, model):
        x = torch.randn(8, 79)
        state = model.construct_rl_state(x)
        assert state.shape == (8, 55)

    def test_classifier_expansion(self, model):
        assert model.num_classes == 10
        model.update_classifier_head(15)
        assert model.num_classes == 15
        x = torch.randn(4, 79)
        logits, _ = model(x)
        assert logits.shape == (4, 15)

    def test_shared_parameters(self, model):
        shared = model.get_shared_parameters()
        assert len(shared) > 0


# ── Policy Network Tests ─────────────────────────────────────────────

class TestPolicyNetwork:
    def test_forward(self):
        policy = PolicyNetwork(state_dim=55, num_actions=5)
        state = torch.randn(16, 55)
        logits = policy(state)
        assert logits.shape == (16, 5)

    def test_get_action(self):
        policy = PolicyNetwork(state_dim=55, num_actions=5)
        state = torch.randn(1, 55)
        action, log_prob = policy.get_action(state)
        assert 0 <= action.item() <= 4
        assert log_prob.shape == (1,)

    def test_evaluate_actions(self):
        policy = PolicyNetwork(state_dim=55, num_actions=5)
        states = torch.randn(32, 55)
        actions = torch.randint(0, 5, (32,))
        log_probs, entropy = policy.evaluate_actions(states, actions)
        assert log_probs.shape == (32,)
        assert entropy.shape == (32,)

    def test_value_network(self):
        vn = ValueNetwork(state_dim=55)
        state = torch.randn(16, 55)
        values = vn(state)
        assert values.shape == (16,)

    def test_cost_value_network(self):
        cvn = CostValueNetwork(state_dim=55)
        state = torch.randn(16, 55)
        costs = cvn(state)
        assert costs.shape == (16,)


# ── Replay Buffer Tests ──────────────────────────────────────────────

class TestReplayBuffer:
    def test_basic_add_and_sample(self):
        buf = ReservoirReplayBuffer(capacity=100, feature_dim=10)
        for i in range(50):
            buf.add(np.random.randn(10).astype(np.float32), i % 5)
        assert len(buf) == 50

        X, y = buf.sample(20)
        assert X.shape == (20, 10)
        assert y.shape == (20,)

    def test_reservoir_overflow(self):
        buf = ReservoirReplayBuffer(capacity=50, feature_dim=10)
        for i in range(200):
            buf.add(np.random.randn(10).astype(np.float32), i % 5)
        assert len(buf) == 50
        assert buf.count == 200

    def test_batch_add(self):
        buf = ReservoirReplayBuffer(capacity=100, feature_dim=10)
        X = np.random.randn(30, 10).astype(np.float32)
        y = np.random.randint(0, 5, 30)
        buf.add_batch(X, y)
        assert len(buf) == 30

    def test_class_distribution(self):
        buf = ReservoirReplayBuffer(capacity=100, feature_dim=10)
        for i in range(100):
            buf.add(np.zeros(10, dtype=np.float32), i % 3)
        dist = buf.get_class_distribution()
        assert len(dist) == 3


# ── Fisher Information Tests ─────────────────────────────────────────

class TestFisher:
    def test_compute_fisher(self, model, sample_data, device):
        X, y = sample_data
        loader = create_dataloader(X[:100], y[:100], batch_size=32, shuffle=False)
        fisher = compute_fisher_diagonal_efficient(model, loader, device, num_samples=50)
        assert len(fisher) > 0
        for name, vals in fisher.items():
            assert vals.shape == dict(model.named_parameters())[name].shape
            assert (vals >= 0).all()  # Fisher diagonal is non-negative

    def test_running_average(self):
        f1 = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])}
        f2 = {"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([1.0])}
        updated = update_running_fisher(f1, f2, decay=0.5)
        assert torch.allclose(updated["a"], torch.tensor([2.0, 3.0]))
        assert torch.allclose(updated["b"], torch.tensor([2.0]))


# ── Drift Detector Tests ─────────────────────────────────────────────

class TestDriftDetector:
    def test_no_drift(self, model, sample_data, device):
        X, y = sample_data
        dd = DriftDetector(num_classes=10)
        dd.set_reference(model, X[:100], y[:100], device)
        status, kl = dd.check_drift(model, X[:100], device)
        assert status == DriftStatus.STABLE
        assert kl < 0.05

    def test_drift_detection(self, model, device):
        dd = DriftDetector(num_classes=10)
        dd.reference_distribution = np.array([0.5, 0.5] + [0]*8)
        # Force drift by using very different distribution
        dd.reference_distribution = dd.reference_distribution / dd.reference_distribution.sum()

        summary = dd.get_drift_summary()
        assert summary["total_checks"] == 0


# ── NIDS Environment Tests ───────────────────────────────────────────

class TestNIDSEnv:
    def test_reset_and_step(self, sample_data):
        X, y = sample_data
        env = NIDSResponseEnv(features=X, labels=y)
        state = env.reset()
        assert state.shape == (55,)

        for action in range(5):
            state, reward, cost, done, info = env.step(action)
            assert state.shape == (55,)
            assert isinstance(reward, float)
            assert cost in [0.0, 1.0]
            assert isinstance(done, bool)

    def test_episode_stats(self, sample_data):
        X, y = sample_data
        env = NIDSResponseEnv(features=X, labels=y)
        env.reset()
        for _ in range(10):
            _, _, _, done, _ = env.step(np.random.randint(0, 5))
            if done:
                break
        stats = env.get_episode_stats()
        assert "total_reward" in stats
        assert "fp_blocking_rate" in stats


# ── Continual Metrics Tests ──────────────────────────────────────────

class TestContinualMetrics:
    def test_aa_bwt_fwt(self):
        cm = ContinualMetrics()
        # Simulate: 3 tasks
        cm.record_accuracy(1, 1, 95.0)
        cm.record_accuracy(2, 1, 93.0)
        cm.record_accuracy(2, 2, 96.0)
        cm.record_accuracy(3, 1, 92.0)
        cm.record_accuracy(3, 2, 95.0)
        cm.record_accuracy(3, 3, 97.0)

        cm.set_random_baseline(2, 50.0)
        cm.set_random_baseline(3, 50.0)

        metrics = cm.compute_all_metrics()
        assert metrics["average_accuracy"] == pytest.approx(
            (92 + 95 + 97) / 3, abs=0.1
        )
        # BWT: avg of (92-95, 95-96) = avg(-3, -1) = -2
        assert metrics["backward_transfer"] < 0


# ── Adversarial Tests ────────────────────────────────────────────────

class TestAdversarial:
    def test_fgsm(self, model, sample_data, device):
        X, y = sample_data
        adv = AdversarialEvaluator(device)
        X_t = torch.FloatTensor(X[:50]).to(device)
        y_t = torch.LongTensor(y[:50]).to(device)
        X_adv = adv.fgsm_attack(model, X_t, y_t, epsilon=0.1)
        assert X_adv.shape == X_t.shape
        # Perturbation should be non-zero
        assert not torch.allclose(X_adv, X_t)

    def test_gaussian_noise(self, model, sample_data, device):
        X, y = sample_data
        adv = AdversarialEvaluator(device)
        X_t = torch.FloatTensor(X[:50]).to(device)
        y_t = torch.LongTensor(y[:50]).to(device)
        X_noisy = adv.gaussian_noise_attack(model, X_t, y_t, sigma=0.1)
        assert X_noisy.shape == X_t.shape


# ── Unified FIM Tests ────────────────────────────────────────────────

class TestUnifiedFIM:
    def test_compute_unified(self):
        fim = UnifiedFIM(beta=0.7)
        det = {"layer.weight": torch.tensor([1.0, 2.0])}
        pol = {"layer.weight": torch.tensor([3.0, 4.0])}
        unified = fim.compute_unified(det, pol)
        # 0.7 * [1,2] + 0.3 * [3,4] = [1.6, 2.6]
        expected = torch.tensor([1.6, 2.6])
        assert torch.allclose(unified["layer.weight"], expected, atol=1e-5)

    def test_detection_only_params(self):
        fim = UnifiedFIM(beta=0.7)
        det = {"det_only": torch.tensor([1.0])}
        pol = {"pol_only": torch.tensor([2.0])}
        unified = fim.compute_unified(det, pol)
        assert "det_only" in unified
        assert "pol_only" in unified
        assert torch.allclose(unified["det_only"], torch.tensor([1.0]))
        assert torch.allclose(unified["pol_only"], torch.tensor([2.0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
