"""
Real-time Detection Engine
===========================

Integrates AI models for live traffic analysis and threat detection.

Author: Roger Nick Anaedevha
"""

import asyncio
import logging
import torch
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import sys

# Add path to integrated_ai_ids models
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "integrated_ai_ids"))

from integrated_ai_ids.core.unified_model import UnifiedIDS, DetectionResult
from app.core.config import settings
from app.models.detection import (
    Detection, DetectionStatus, AttackType, SeverityLevel
)

logger = logging.getLogger(__name__)


class DetectionEngine:
    """
    Real-time threat detection engine

    Processes network traffic through AI models and generates
    detection results and alerts.
    """

    def __init__(self):
        """Initialize detection engine"""
        self.model: Optional[UnifiedIDS] = None
        self.device = "cuda" if (settings.ENABLE_GPU and torch.cuda.is_available()) else "cpu"
        self.is_initialized = False

        # Performance metrics
        self.total_detections = 0
        self.total_malicious = 0
        self.total_benign = 0
        self.avg_latency_ms = 0.0

        logger.info(f"Detection engine initialized on device: {self.device}")

    async def initialize(self):
        """Load and initialize AI models"""
        if self.is_initialized:
            logger.warning("Detection engine already initialized")
            return

        try:
            logger.info("Loading AI models...")

            # Initialize unified IDS model
            self.model = UnifiedIDS(
                models=settings.ENABLED_MODELS_LIST,
                confidence_threshold=settings.CONFIDENCE_THRESHOLD,
                device=self.device
            )

            # Load checkpoint if available
            checkpoint_path = Path(settings.MODEL_CHECKPOINT_PATH)
            if checkpoint_path.exists():
                logger.info(f"Loading model checkpoint from: {checkpoint_path}")
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                logger.info("✅ Model checkpoint loaded")
            else:
                logger.warning(f"No checkpoint found at: {checkpoint_path}. Using random initialization.")

            # Set to evaluation mode
            self.model.eval()

            # Compile with TorchScript if enabled
            if settings.ENABLE_TORCHSCRIPT:
                logger.info("Compiling model with TorchScript...")
                sample_input = torch.randn(1, 64).to(self.device)
                self.model = torch.jit.trace(self.model, sample_input)
                logger.info("✅ Model compiled with TorchScript")

            self.is_initialized = True
            logger.info("✅ Detection engine ready")

        except Exception as e:
            logger.error(f"Failed to initialize detection engine: {e}", exc_info=True)
            raise

    async def detect(self, traffic_id: int, db) -> Optional[Detection]:
        """
        Run detection on network traffic

        Args:
            traffic_id: ID of NetworkTraffic record
            db: Database session

        Returns:
            Detection object
        """
        if not self.is_initialized:
            await self.initialize()

        start_time = datetime.now()

        try:
            # Get traffic record
            from app.models.detection import NetworkTraffic

            traffic = db.query(NetworkTraffic).filter(NetworkTraffic.id == traffic_id).first()
            if not traffic:
                logger.error(f"Traffic record not found: {traffic_id}")
                return None

            # Extract features
            features = self._extract_features(traffic)

            # Convert to tensor
            feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                result: DetectionResult = self.model(feature_tensor)

            # Calculate processing time
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Map attack type
            attack_type = self._map_attack_type(result.attack_type)

            # Determine severity
            severity = self._determine_severity(result.confidence, attack_type)

            # Create detection record
            detection = Detection(
                traffic_id=traffic_id,
                is_malicious=result.is_malicious,
                attack_type=attack_type,
                confidence=result.confidence,
                severity=severity,
                model_name="UnifiedIDS",
                model_version=settings.API_VERSION,
                model_predictions=result.model_predictions,
                epistemic_uncertainty=result.uncertainty.get('epistemic'),
                aleatoric_uncertainty=result.uncertainty.get('aleatoric'),
                feature_importance=result.explanation.get('feature_importance'),
                explanation=result.explanation.get('text'),
                status=DetectionStatus.COMPLETED,
                processing_time_ms=processing_time_ms
            )

            # Save to database
            db.add(detection)
            db.commit()
            db.refresh(detection)

            # Update metrics
            self.total_detections += 1
            if result.is_malicious:
                self.total_malicious += 1
            else:
                self.total_benign += 1

            # Update avg latency (exponential moving average)
            alpha = 0.1
            self.avg_latency_ms = alpha * processing_time_ms + (1 - alpha) * self.avg_latency_ms

            logger.info(
                f"Detection completed: traffic_id={traffic_id}, "
                f"malicious={result.is_malicious}, "
                f"attack_type={attack_type}, "
                f"confidence={result.confidence:.3f}, "
                f"latency={processing_time_ms:.1f}ms"
            )

            # Trigger alerts if malicious
            if result.is_malicious and result.confidence >= settings.ALERT_MIN_CONFIDENCE:
                await self._create_alert(detection, db)

            # Trigger prevention if enabled
            if settings.AUTO_BLOCK and result.is_malicious and result.confidence >= 0.95:
                await self._trigger_prevention(detection, traffic, db)

            return detection

        except Exception as e:
            logger.error(f"Error in detection: {e}", exc_info=True)

            # Create failed detection record
            detection = Detection(
                traffic_id=traffic_id,
                is_malicious=False,
                attack_type=AttackType.UNKNOWN,
                confidence=0.0,
                severity=SeverityLevel.INFO,
                model_name="UnifiedIDS",
                status=DetectionStatus.FAILED,
                explanation=f"Error: {str(e)}"
            )
            db.add(detection)
            db.commit()

            return detection

    def _extract_features(self, traffic) -> np.ndarray:
        """
        Extract ML features from traffic record

        Args:
            traffic: NetworkTraffic object

        Returns:
            Feature vector (64 dimensions)
        """
        # Use stored features if available
        if traffic.features:
            feature_list = [
                traffic.features.get('packet_length', 0),
                traffic.features.get('header_length', 0),
                traffic.features.get('payload_size', 0),
                traffic.features.get('payload_entropy', 0.0),
                traffic.features.get('protocol_number', 0),
                traffic.features.get('ttl', 0),
                traffic.features.get('tcp_window', 0),
                traffic.features.get('tcp_flags_int', 0),
                traffic.features.get('is_encrypted', 0),
            ]

            # Pad to 64 dimensions
            while len(feature_list) < 64:
                feature_list.append(0.0)

            return np.array(feature_list[:64], dtype=np.float32)

        # Fallback: extract basic features
        features = np.zeros(64, dtype=np.float32)

        features[0] = traffic.bytes_transferred
        features[1] = traffic.packets
        features[2] = traffic.duration
        features[3] = traffic.payload_size
        features[4] = traffic.payload_entropy or 0.0
        features[5] = float(traffic.is_encrypted)
        features[6] = traffic.src_port or 0
        features[7] = traffic.dst_port or 0

        # Protocol encoding
        protocol_map = {'tcp': 1, 'udp': 2, 'icmp': 3}
        features[8] = protocol_map.get(traffic.protocol, 0)

        return features

    @staticmethod
    def _map_attack_type(attack_type_str: str) -> AttackType:
        """Map model output to AttackType enum"""
        mapping = {
            'ddos': AttackType.DDOS,
            'port_scan': AttackType.PORT_SCAN,
            'brute_force': AttackType.BRUTE_FORCE,
            'sql_injection': AttackType.SQL_INJECTION,
            'xss': AttackType.XSS,
            'malware': AttackType.MALWARE,
            'ransomware': AttackType.RANSOMWARE,
            'backdoor': AttackType.BACKDOOR,
            'botnet': AttackType.BOTNET,
            'data_exfiltration': AttackType.DATA_EXFILTRATION,
            'privilege_escalation': AttackType.PRIVILEGE_ESCALATION,
            'lateral_movement': AttackType.LATERAL_MOVEMENT,
            'zero_day': AttackType.ZERO_DAY,
            'benign': AttackType.BENIGN,
        }
        return mapping.get(attack_type_str.lower(), AttackType.UNKNOWN)

    @staticmethod
    def _determine_severity(confidence: float, attack_type: AttackType) -> SeverityLevel:
        """Determine alert severity based on confidence and attack type"""
        if attack_type == AttackType.BENIGN:
            return SeverityLevel.INFO

        # Critical attacks
        critical_attacks = {
            AttackType.RANSOMWARE,
            AttackType.BACKDOOR,
            AttackType.DATA_EXFILTRATION,
            AttackType.ZERO_DAY
        }
        if attack_type in critical_attacks:
            return SeverityLevel.CRITICAL

        # High severity based on confidence
        if confidence >= 0.95:
            return SeverityLevel.HIGH
        elif confidence >= 0.85:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW

    async def _create_alert(self, detection: Detection, db):
        """
        Create alert from detection

        Args:
            detection: Detection object
            db: Database session
        """
        from app.models.detection import Alert
        from app.services.alert_service import AlertService

        try:
            # Create alert record
            alert = Alert(
                detection_id=detection.id,
                title=f"{detection.attack_type.value.upper()} Detected",
                description=detection.explanation or f"Detected {detection.attack_type.value} attack",
                severity=detection.severity,
                is_acknowledged=False,
                is_resolved=False
            )

            db.add(alert)
            db.commit()
            db.refresh(alert)

            # Send notifications
            alert_service = AlertService()
            await alert_service.send_alert(alert, detection)

            logger.info(f"Alert created: {alert.id} for detection {detection.id}")

        except Exception as e:
            logger.error(f"Error creating alert: {e}", exc_info=True)

    async def _trigger_prevention(self, detection: Detection, traffic, db):
        """
        Trigger automated prevention action

        Args:
            detection: Detection object
            traffic: NetworkTraffic object
            db: Database session
        """
        from app.models.detection import PreventionAction, ActionStatus
        from app.services.firewall import FirewallService

        try:
            # Create prevention action
            action = PreventionAction(
                detection_id=detection.id,
                action_type="block_ip",
                target=traffic.src_ip,
                parameters={
                    "duration": settings.BLOCK_DURATION,
                    "reason": f"{detection.attack_type.value} detected"
                },
                status=ActionStatus.PENDING,
                auto_rollback=True
            )

            db.add(action)
            db.commit()
            db.refresh(action)

            # Execute firewall rule
            firewall = FirewallService()
            success = await firewall.block_ip(traffic.src_ip, settings.BLOCK_DURATION)

            if success:
                action.status = ActionStatus.EXECUTED
                action.executed_at = datetime.now()
                logger.info(f"✅ Blocked IP: {traffic.src_ip}")
            else:
                action.status = ActionStatus.FAILED
                action.error_message = "Failed to execute firewall rule"
                logger.error(f"❌ Failed to block IP: {traffic.src_ip}")

            db.commit()

        except Exception as e:
            logger.error(f"Error triggering prevention: {e}", exc_info=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get detection engine statistics"""
        return {
            "total_detections": self.total_detections,
            "malicious_count": self.total_malicious,
            "benign_count": self.total_benign,
            "detection_rate": (
                self.total_malicious / self.total_detections
                if self.total_detections > 0 else 0.0
            ),
            "avg_latency_ms": self.avg_latency_ms,
            "is_initialized": self.is_initialized,
            "device": self.device
        }
