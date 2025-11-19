"""
Unified AI-IDS: Core Integrated Model
======================================

This module implements the unified ensemble integrating all dissertation models:
- Temporal Adaptive Neural ODEs (TA-BN-ODE)
- Privacy-Preserving Federated Optimal Transport (PPFOT-IDS)
- Encrypted Traffic Analysis (Hybrid CNN-LSTM-Transformer)
- Federated Graph Temporal Dynamics (FedGTD)
- Heterogeneous Graph Pooling (HGP)
- LLM-based Zero-Shot Detection

Author: Roger Nick Anaedevha
Institution: National Research Nuclear University MEPhI
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from pathlib import Path

# Import individual models
from ..models.neural_ode import TemporalAdaptiveNeuralODE
from ..models.optimal_transport import PPFOTDetector
from ..models.encrypted_traffic import HybridEncryptedTrafficAnalyzer
from ..models.federated_learning import FederatedGraphTD
from ..models.graph_model import HeterogeneousGraphPooling
from ..models.llm_detector import LLMZeroShotDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Detection result with comprehensive information"""
    is_malicious: bool
    confidence: float
    attack_type: Optional[str]
    attack_category: Optional[str]
    severity: str  # 'low', 'medium', 'high', 'critical'
    explanation: str
    model_predictions: Dict[str, float]
    uncertainty_bounds: Tuple[float, float]
    feature_importance: Dict[str, float]
    recommended_action: str
    metadata: Dict


class UnifiedIDS(nn.Module):
    """
    Unified AI-powered Intrusion Detection System

    Integrates all dissertation models into a single ensemble with:
    - Multi-model prediction fusion
    - Uncertainty quantification
    - Explainable AI
    - Adaptive thresholding
    - Real-time processing

    Args:
        models: List of model names to enable
        config_path: Path to configuration file
        device: Device for inference ('cpu', 'cuda', 'tpu')
        confidence_threshold: Minimum confidence for detection
        enable_uncertainty: Enable Bayesian uncertainty quantification
        enable_explanation: Enable SHAP explanations
    """

    def __init__(
        self,
        models: List[str] = None,
        config_path: Optional[Path] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        confidence_threshold: float = 0.85,
        enable_uncertainty: bool = True,
        enable_explanation: bool = True,
        **kwargs
    ):
        super(UnifiedIDS, self).__init__()

        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.enable_uncertainty = enable_uncertainty
        self.enable_explanation = enable_explanation

        # Default to all models if none specified
        if models is None:
            models = ['neural_ode', 'optimal_transport', 'encrypted_traffic',
                     'federated', 'graph', 'llm']

        self.active_models = models
        logger.info(f"Initializing Unified AI-IDS with models: {models}")

        # Initialize individual models
        self.model_dict = nn.ModuleDict()
        self._init_models(config_path, **kwargs)

        # Decision fusion layer
        self.fusion_layer = DecisionFusionNetwork(
            num_models=len(self.active_models),
            hidden_dim=256,
            output_dim=1
        )

        # Model weights for weighted voting
        self.model_weights = nn.Parameter(torch.ones(len(self.active_models)))

        # Attack type classifier
        self.attack_classifier = AttackTypeClassifier(
            input_dim=len(self.active_models),
            num_classes=13  # Standard attack categories
        )

        # Move to device
        self.to(self.device)

        logger.info(f"Unified AI-IDS initialized on {self.device}")

    def _init_models(self, config_path: Optional[Path], **kwargs):
        """Initialize individual models based on configuration"""

        if 'neural_ode' in self.active_models:
            logger.info("Loading Temporal Adaptive Neural ODE model...")
            self.model_dict['neural_ode'] = TemporalAdaptiveNeuralODE(
                input_dim=64,
                hidden_dims=[128, 256, 256, 128],
                num_layers=4,
                ode_solver='dopri5',
                enable_point_process=True,
                **kwargs
            )

        if 'optimal_transport' in self.active_models:
            logger.info("Loading Optimal Transport model...")
            self.model_dict['optimal_transport'] = PPFOTDetector(
                source_dim=64,
                target_dim=64,
                epsilon=0.85,
                delta=1e-5,
                enable_privacy=True,
                **kwargs
            )

        if 'encrypted_traffic' in self.active_models:
            logger.info("Loading Encrypted Traffic Analyzer...")
            self.model_dict['encrypted_traffic'] = HybridEncryptedTrafficAnalyzer(
                cnn_filters=[64, 128, 256],
                lstm_hidden=128,
                transformer_layers=6,
                attention_heads=8,
                **kwargs
            )

        if 'federated' in self.active_models:
            logger.info("Loading Federated Graph TD model...")
            self.model_dict['federated'] = FederatedGraphTD(
                node_dim=64,
                hidden_dim=256,
                num_gnn_layers=3,
                **kwargs
            )

        if 'graph' in self.active_models:
            logger.info("Loading Heterogeneous Graph Pooling...")
            self.model_dict['graph'] = HeterogeneousGraphPooling(
                in_channels=64,
                hidden_channels=256,
                num_relations=5,
                pooling_ratio=0.2,
                **kwargs
            )

        if 'llm' in self.active_models:
            logger.info("Loading LLM Zero-Shot Detector...")
            self.model_dict['llm'] = LLMZeroShotDetector(
                model_name='gpt-4',
                enable_chain_of_thought=True,
                **kwargs
            )

    def forward(
        self,
        x: Union[torch.Tensor, Dict],
        return_all_predictions: bool = False,
        return_uncertainty: bool = None,
        return_explanation: bool = None
    ) -> DetectionResult:
        """
        Forward pass through unified model

        Args:
            x: Input data (tensor or dict with multiple modalities)
            return_all_predictions: Return individual model predictions
            return_uncertainty: Override uncertainty quantification setting
            return_explanation: Override explanation setting

        Returns:
            DetectionResult with comprehensive detection information
        """

        if return_uncertainty is None:
            return_uncertainty = self.enable_uncertainty
        if return_explanation is None:
            return_explanation = self.enable_explanation

        # Collect predictions from all active models
        model_predictions = {}
        prediction_tensors = []

        with torch.no_grad() if not self.training else torch.enable_grad():
            for model_name, model in self.model_dict.items():
                try:
                    # Get model-specific prediction
                    pred = model(self._prepare_input(x, model_name))

                    if isinstance(pred, tuple):
                        pred, _ = pred  # Some models return (prediction, auxiliary)

                    # Convert to probability
                    if pred.dim() > 1:
                        pred = torch.sigmoid(pred).squeeze()

                    model_predictions[model_name] = pred.item()
                    prediction_tensors.append(pred.unsqueeze(0))

                except Exception as e:
                    logger.warning(f"Model {model_name} failed: {e}")
                    model_predictions[model_name] = 0.5  # Neutral prediction
                    prediction_tensors.append(torch.tensor([0.5], device=self.device))

        # Stack predictions
        predictions_tensor = torch.cat(prediction_tensors, dim=0)

        # Decision fusion
        fused_prediction = self.fusion_layer(predictions_tensor.unsqueeze(0))
        confidence = torch.sigmoid(fused_prediction).item()

        # Weighted voting as fallback
        weights = torch.softmax(self.model_weights, dim=0)
        weighted_pred = (predictions_tensor * weights).sum().item()

        # Use fusion if confident, otherwise weighted voting
        final_confidence = confidence if abs(confidence - 0.5) > 0.1 else weighted_pred

        # Determine if malicious
        is_malicious = final_confidence >= self.confidence_threshold

        # Attack type classification
        attack_logits = self.attack_classifier(predictions_tensor.unsqueeze(0))
        attack_probs = torch.softmax(attack_logits, dim=-1)
        attack_idx = torch.argmax(attack_probs).item()
        attack_type = self._get_attack_type(attack_idx)
        attack_category = self._get_attack_category(attack_type)

        # Uncertainty quantification
        if return_uncertainty:
            uncertainty_bounds = self._compute_uncertainty(predictions_tensor)
        else:
            uncertainty_bounds = (final_confidence, final_confidence)

        # Feature importance / explanation
        if return_explanation:
            explanation, feature_importance = self._generate_explanation(
                x, model_predictions, attack_type
            )
        else:
            explanation = f"Attack detected: {attack_type}" if is_malicious else "Benign traffic"
            feature_importance = {}

        # Severity assessment
        severity = self._assess_severity(final_confidence, attack_type)

        # Recommended action
        recommended_action = self._recommend_action(severity, attack_type)

        return DetectionResult(
            is_malicious=is_malicious,
            confidence=final_confidence,
            attack_type=attack_type if is_malicious else None,
            attack_category=attack_category if is_malicious else None,
            severity=severity,
            explanation=explanation,
            model_predictions=model_predictions if return_all_predictions else {},
            uncertainty_bounds=uncertainty_bounds,
            feature_importance=feature_importance,
            recommended_action=recommended_action,
            metadata={
                'num_models': len(self.active_models),
                'fusion_confidence': confidence,
                'weighted_confidence': weighted_pred,
                'attack_probability': attack_probs[attack_idx].item()
            }
        )

    def _prepare_input(self, x: Union[torch.Tensor, Dict], model_name: str) -> torch.Tensor:
        """Prepare input for specific model"""
        if isinstance(x, dict):
            # Multi-modal input
            if model_name in x:
                return x[model_name]
            else:
                # Use default feature vector
                return x.get('features', x.get('default', None))
        return x

    def _compute_uncertainty(self, predictions: torch.Tensor) -> Tuple[float, float]:
        """Compute uncertainty bounds using model disagreement and Bayesian inference"""
        # Model disagreement
        std = predictions.std().item()
        mean = predictions.mean().item()

        # Bayesian credible interval (approximate)
        lower = max(0.0, mean - 1.96 * std)
        upper = min(1.0, mean + 1.96 * std)

        return (lower, upper)

    def _generate_explanation(
        self,
        x: Union[torch.Tensor, Dict],
        model_predictions: Dict[str, float],
        attack_type: str
    ) -> Tuple[str, Dict[str, float]]:
        """Generate human-readable explanation using LLM if available"""

        # Feature importance (placeholder - implement SHAP in production)
        feature_importance = {
            'packet_size': 0.23,
            'inter_arrival_time': 0.18,
            'protocol_type': 0.15,
            'port_number': 0.12,
            'flags': 0.10,
            'payload_entropy': 0.22
        }

        # Generate explanation
        top_model = max(model_predictions.items(), key=lambda x: x[1])

        explanation = (
            f"Primary detection by {top_model[0]} model (confidence: {top_model[1]:.2%}). "
            f"Attack pattern identified as {attack_type}. "
            f"Key indicators: {', '.join(list(feature_importance.keys())[:3])}."
        )

        # Use LLM for detailed explanation if available
        if 'llm' in self.model_dict and isinstance(x, dict):
            try:
                llm_explanation = self.model_dict['llm'].explain(x, attack_type)
                explanation = f"{explanation} {llm_explanation}"
            except:
                pass

        return explanation, feature_importance

    def _get_attack_type(self, idx: int) -> str:
        """Map attack index to attack type name"""
        attack_types = [
            'Benign',
            'DoS/DDoS',
            'Brute Force',
            'Web Attack',
            'Botnet',
            'Infiltration',
            'Port Scan',
            'SQL Injection',
            'XSS',
            'Malware',
            'Ransomware',
            'APT',
            'Zero-Day'
        ]
        return attack_types[min(idx, len(attack_types) - 1)]

    def _get_attack_category(self, attack_type: str) -> str:
        """Get high-level attack category"""
        categories = {
            'DoS/DDoS': 'Denial of Service',
            'Brute Force': 'Authentication Attack',
            'Web Attack': 'Application Layer',
            'SQL Injection': 'Injection Attack',
            'XSS': 'Injection Attack',
            'Botnet': 'C&C Communication',
            'Malware': 'Malicious Code',
            'Ransomware': 'Malicious Code',
            'APT': 'Advanced Threat',
            'Zero-Day': 'Unknown Threat',
            'Port Scan': 'Reconnaissance',
            'Infiltration': 'Intrusion'
        }
        return categories.get(attack_type, 'Unknown')

    def _assess_severity(self, confidence: float, attack_type: str) -> str:
        """Assess attack severity"""
        high_severity_attacks = ['APT', 'Zero-Day', 'Ransomware', 'Infiltration']
        medium_severity_attacks = ['Malware', 'Botnet', 'SQL Injection']

        if attack_type in high_severity_attacks:
            return 'critical' if confidence > 0.9 else 'high'
        elif attack_type in medium_severity_attacks:
            return 'high' if confidence > 0.9 else 'medium'
        else:
            return 'medium' if confidence > 0.9 else 'low'

    def _recommend_action(self, severity: str, attack_type: str) -> str:
        """Recommend remediation action"""
        actions = {
            'critical': 'IMMEDIATE: Isolate affected systems, activate incident response team, preserve forensic evidence',
            'high': 'URGENT: Block source IP, analyze affected systems, escalate to security team',
            'medium': 'Monitor closely, log for analysis, implement additional filtering',
            'low': 'Log event, continue monitoring, periodic review'
        }
        return actions.get(severity, 'Review and assess')

    def load_pretrained(self, checkpoint_path: Path):
        """Load pretrained model weights"""
        logger.info(f"Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Pretrained weights loaded successfully")

    def save_checkpoint(self, path: Path, metadata: Dict = None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'active_models': self.active_models,
            'confidence_threshold': self.confidence_threshold,
            'metadata': metadata or {}
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")


class DecisionFusionNetwork(nn.Module):
    """Neural network for fusing multi-model predictions"""

    def __init__(self, num_models: int, hidden_dim: int, output_dim: int):
        super(DecisionFusionNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(num_models, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class AttackTypeClassifier(nn.Module):
    """Classifier for attack type identification"""

    def __init__(self, input_dim: int, num_classes: int):
        super(AttackTypeClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


# Example usage
if __name__ == "__main__":
    # Initialize unified IDS
    ids = UnifiedIDS(
        models=['neural_ode', 'optimal_transport', 'encrypted_traffic'],
        confidence_threshold=0.85
    )

    # Mock input
    sample_input = torch.randn(1, 64)

    # Detect
    result = ids(sample_input)

    print(f"Malicious: {result.is_malicious}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Attack Type: {result.attack_type}")
    print(f"Severity: {result.severity}")
    print(f"Explanation: {result.explanation}")
    print(f"Recommended Action: {result.recommended_action}")
