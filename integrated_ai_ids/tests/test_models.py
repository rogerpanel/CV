"""
Comprehensive Model Testing Suite
==================================

Tests for all dissertation models with real data scenarios.

Author: Roger Nick Anaedevha
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.neural_ode import TemporalAdaptiveNeuralODE, PointProcessNeuralODE, BayesianNeuralODE
from models.optimal_transport import PPFOTDetector, SinkhornDistance
from models.encrypted_traffic import EncryptedTrafficAnalyzer, StreamingEncryptedAnalyzer
from models.federated_graph import FedGTDModel, GraphTemporalODE
from models.heterogeneous_graph import HGPModel
from models.bayesian_inference import BayesianUncertaintyNet
from core.unified_model import UnifiedIDS


class TestNeuralODE:
    """Test Neural ODE models"""

    def test_temporal_adaptive_neural_ode(self):
        """Test TA-BN-ODE model"""
        model = TemporalAdaptiveNeuralODE(
            input_dim=64,
            hidden_dims=[128, 256, 256, 128],
            num_classes=13
        )

        # Sample input
        batch_size = 16
        x = torch.randn(batch_size, 64)

        # Forward pass
        binary_logits, multiclass_logits = model(x)

        # Check shapes
        assert binary_logits.shape == (batch_size, 1)
        assert multiclass_logits.shape == (batch_size, 13)

        # Check values are finite
        assert torch.isfinite(binary_logits).all()
        assert torch.isfinite(multiclass_logits).all()

        print("✓ Temporal Adaptive Neural ODE test passed")

    def test_point_process_neural_ode(self):
        """Test Point Process Neural ODE"""
        model = PointProcessNeuralODE(
            input_dim=64,
            hidden_dim=256,
            num_event_types=13
        )

        # Sample event sequence
        batch_size = 8
        seq_len = 20
        event_sequence = torch.randn(batch_size, seq_len, 64)
        inter_event_times = torch.rand(batch_size, seq_len) * 2.0

        # Forward pass
        intensity, mark_logits, hidden_states = model(event_sequence, inter_event_times)

        # Check shapes
        assert intensity.shape == (batch_size, seq_len, 1)
        assert mark_logits.shape == (batch_size, seq_len, 13)

        # Intensity should be positive
        assert (intensity > 0).all()

        print("✓ Point Process Neural ODE test passed")

    def test_bayesian_neural_ode(self):
        """Test Bayesian Neural ODE"""
        model = BayesianNeuralODE(
            input_dim=64,
            hidden_dim=256,
            num_classes=13,
            num_mc_samples=5
        )

        model.train()
        x = torch.randn(16, 64)

        # Forward with uncertainty
        binary_mean, multiclass_mean, binary_unc, multiclass_unc = model(
            x,
            return_uncertainty=True
        )

        # Check shapes
        assert binary_mean.shape == (16, 1)
        assert multiclass_mean.shape == (16, 13)
        assert binary_unc.shape == (16, 1)
        assert multiclass_unc.shape == (16, 13)

        # Uncertainty should be non-negative
        assert (binary_unc >= 0).all()
        assert (multiclass_unc >= 0).all()

        print("✓ Bayesian Neural ODE test passed")


class TestOptimalTransport:
    """Test Optimal Transport models"""

    def test_sinkhorn_distance(self):
        """Test Sinkhorn algorithm"""
        sinkhorn = SinkhornDistance(reg=0.1, max_iter=50)

        batch_size = 8
        n_source = 30
        n_target = 25
        dim = 64

        source = torch.randn(batch_size, n_source, dim)
        target = torch.randn(batch_size, n_target, dim)

        # Compute distance
        distance, transport_plan = sinkhorn(source, target)

        # Check shapes
        assert distance.shape == (batch_size,)
        assert transport_plan.shape == (batch_size, n_source, n_target)

        # Distance should be non-negative
        assert (distance >= 0).all()

        # Transport plan should sum to marginals
        marginal_sum = transport_plan.sum(dim=2).sum(dim=1)
        assert torch.allclose(marginal_sum, torch.ones(batch_size), atol=1e-3)

        print("✓ Sinkhorn distance test passed")

    def test_ppfot_detector(self):
        """Test PPFOT-IDS model"""
        model = PPFOTDetector(
            input_dim=64,
            hidden_dim=256,
            num_classes=13,
            epsilon=0.85,
            enable_privacy=True,
            enable_byzantine=True
        )

        batch_size = 16
        x = torch.randn(batch_size, 64)

        # Forward pass
        binary_logits, multiclass_logits = model(x)

        # Check shapes
        assert binary_logits.shape == (batch_size, 1)
        assert multiclass_logits.shape == (batch_size, 13)

        print("✓ PPFOT detector test passed")

    def test_domain_adaptation(self):
        """Test domain adaptation"""
        model = PPFOTDetector(
            input_dim=64,
            hidden_dim=256,
            num_classes=13
        )

        batch_size = 16
        source = torch.randn(batch_size, 64)
        target = torch.randn(batch_size, 64)

        # Forward with adaptation
        binary_logits, multiclass_logits = model(source, target)

        assert binary_logits.shape == (batch_size, 1)
        assert multiclass_logits.shape == (batch_size, 13)

        print("✓ Domain adaptation test passed")


class TestEncryptedTraffic:
    """Test Encrypted Traffic Analyzer"""

    def test_encrypted_traffic_analyzer(self):
        """Test hybrid CNN-LSTM-Transformer"""
        model = EncryptedTrafficAnalyzer(
            input_dim=64,
            packet_seq_len=100,
            cnn_filters=[64, 128, 256],
            lstm_hidden=128,
            transformer_dim=256,
            num_classes=13
        )

        batch_size = 8
        seq_len = 100
        packet_sequence = torch.randn(batch_size, seq_len, 64)
        tls_features = torch.randn(batch_size, 32)

        # Forward pass
        binary_logits, multiclass_logits, attention_weights = model(
            packet_sequence,
            tls_features
        )

        # Check shapes
        assert binary_logits.shape == (batch_size, 1)
        assert multiclass_logits.shape == (batch_size, 13)

        print("✓ Encrypted traffic analyzer test passed")

    def test_streaming_analyzer(self):
        """Test streaming encrypted traffic analysis"""
        base_model = EncryptedTrafficAnalyzer(
            input_dim=64,
            packet_seq_len=100,
            num_classes=13
        )

        streaming_model = StreamingEncryptedAnalyzer(
            base_model=base_model,
            window_size=50,
            stride=25
        )

        # Add packets one by one
        results = []
        for i in range(60):
            packet = torch.randn(64)
            result = streaming_model.add_packet(packet)
            if result is not None:
                results.append(result)

        # Should get at least one result
        assert len(results) > 0

        print("✓ Streaming encrypted traffic test passed")


class TestGraphModels:
    """Test Graph-based models"""

    def test_fedgtd_model(self):
        """Test Federated Graph Temporal Dynamics"""
        model = FedGTDModel(
            node_dim=64,
            hidden_dim=256,
            num_classes=13
        )

        num_nodes = 50
        node_features = torch.randn(num_nodes, 64)
        adj_matrix = torch.rand(num_nodes, num_nodes)
        adj_matrix = (adj_matrix + adj_matrix.t()) / 2
        adj_matrix = adj_matrix / (adj_matrix.sum(1, keepdim=True) + 1e-8)

        # Forward pass
        binary_logits, multiclass_logits = model(node_features, adj_matrix)

        assert binary_logits.shape == (1, 1)
        assert multiclass_logits.shape == (1, 13)

        print("✓ FedGTD model test passed")

    def test_hgp_model(self):
        """Test Heterogeneous Graph Pooling"""
        model = HGPModel(
            node_types=['host', 'switch', 'router'],
            edge_types=['connection', 'flow'],
            node_feature_dim=64,
            hidden_dim=256,
            num_classes=13
        )

        # Sample heterogeneous graph
        node_features = {
            'host': torch.randn(20, 64),
            'switch': torch.randn(10, 64),
            'router': torch.randn(5, 64)
        }

        num_edges = 80
        edge_index = torch.randint(0, 35, (2, num_edges))
        edge_types = torch.randint(0, 2, (num_edges,))

        # Forward pass
        binary_logits, multiclass_logits = model(node_features, edge_index, edge_types)

        assert binary_logits.shape == (1, 1)
        assert multiclass_logits.shape == (1, 13)

        print("✓ HGP model test passed")


class TestBayesianInference:
    """Test Bayesian uncertainty quantification"""

    def test_bayesian_uncertainty_net(self):
        """Test Bayesian uncertainty network"""
        model = BayesianUncertaintyNet(
            input_dim=64,
            hidden_dims=[256, 128, 64],
            num_classes=13,
            num_mc_samples=10
        )

        model.train()
        x = torch.randn(16, 64)

        # Forward with uncertainty
        output = model(x, return_uncertainty=True)

        assert 'binary_mean' in output
        assert 'multiclass_mean' in output
        assert 'binary_uncertainty' in output
        assert 'multiclass_uncertainty' in output

        # Check shapes
        assert output['binary_mean'].shape == (16, 1)
        assert output['multiclass_mean'].shape == (16, 13)

        # Uncertainty should be non-negative
        assert (output['binary_uncertainty'] >= 0).all()
        assert (output['multiclass_uncertainty'] >= 0).all()

        print("✓ Bayesian uncertainty net test passed")

    def test_kl_divergence(self):
        """Test KL divergence computation"""
        model = BayesianUncertaintyNet(
            input_dim=64,
            hidden_dims=[128, 64],
            num_classes=13,
            kl_weight=0.01
        )

        kl_loss = model.compute_kl_loss()

        # KL should be non-negative
        assert kl_loss >= 0

        print("✓ KL divergence test passed")

    def test_pac_bayes_bound(self):
        """Test PAC-Bayes generalization bound"""
        model = BayesianUncertaintyNet(
            input_dim=64,
            hidden_dims=[128, 64],
            num_classes=13
        )

        train_loss = torch.tensor(0.5)
        bound = model.pac_bayes_bound(train_loss, num_samples=10000)

        # Bound should be greater than training loss
        assert bound >= train_loss

        print("✓ PAC-Bayes bound test passed")


class TestUnifiedModel:
    """Test integrated unified IDS"""

    def test_unified_ids(self):
        """Test complete unified IDS system"""
        model = UnifiedIDS(
            models=['neural_ode', 'optimal_transport', 'encrypted_traffic'],
            confidence_threshold=0.85
        )

        batch_size = 8
        x = torch.randn(batch_size, 64)

        # Forward pass
        result = model(x)

        # Check result attributes
        assert hasattr(result, 'is_malicious')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'attack_type')
        assert hasattr(result, 'severity')
        assert hasattr(result, 'model_predictions')

        # Check types
        assert isinstance(result.is_malicious, bool)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

        print("✓ Unified IDS test passed")

    def test_batch_detection(self):
        """Test batch detection"""
        model = UnifiedIDS(
            models=['neural_ode', 'optimal_transport'],
            confidence_threshold=0.85
        )

        batch_size = 16
        x = torch.randn(batch_size, 64)

        # Process batch
        results = []
        for i in range(batch_size):
            result = model(x[i:i+1])
            results.append(result)

        assert len(results) == batch_size

        print("✓ Batch detection test passed")


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*60)
    print("INTEGRATED AI-IDS MODEL TESTING SUITE")
    print("="*60 + "\n")

    # Neural ODE tests
    print("[ Neural ODE Tests ]")
    test_node = TestNeuralODE()
    test_node.test_temporal_adaptive_neural_ode()
    test_node.test_point_process_neural_ode()
    test_node.test_bayesian_neural_ode()
    print()

    # Optimal Transport tests
    print("[ Optimal Transport Tests ]")
    test_ot = TestOptimalTransport()
    test_ot.test_sinkhorn_distance()
    test_ot.test_ppfot_detector()
    test_ot.test_domain_adaptation()
    print()

    # Encrypted Traffic tests
    print("[ Encrypted Traffic Tests ]")
    test_et = TestEncryptedTraffic()
    test_et.test_encrypted_traffic_analyzer()
    test_et.test_streaming_analyzer()
    print()

    # Graph model tests
    print("[ Graph Model Tests ]")
    test_graph = TestGraphModels()
    test_graph.test_fedgtd_model()
    test_graph.test_hgp_model()
    print()

    # Bayesian inference tests
    print("[ Bayesian Inference Tests ]")
    test_bayes = TestBayesianInference()
    test_bayes.test_bayesian_uncertainty_net()
    test_bayes.test_kl_divergence()
    test_bayes.test_pac_bayes_bound()
    print()

    # Unified model tests
    print("[ Unified Model Tests ]")
    test_unified = TestUnifiedModel()
    test_unified.test_unified_ids()
    test_unified.test_batch_detection()
    print()

    print("="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
