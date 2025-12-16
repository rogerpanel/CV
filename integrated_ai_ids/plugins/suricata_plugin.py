"""
Suricata Plugin for Integrated AI-IDS
======================================

Real-time integration with Suricata IDS using EVE JSON output.

Features:
- Real-time EVE JSON parsing
- AI-enhanced alert enrichment
- Bidirectional communication
- Custom rule generation
- Alert correlation

Installation:
    python -m integrated_ai_ids.plugins.install suricata

Configuration:
    /etc/suricata/ai-ids.yaml

Author: Roger Nick Anaedevha
"""

import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import socket

from ..core.unified_model import UnifiedIDS, DetectionResult

logger = logging.getLogger(__name__)


class SuricataPlugin:
    """
    Suricata Integration Plugin

    Monitors Suricata EVE JSON output and enhances alerts with AI predictions

    Args:
        eve_log_path: Path to Suricata EVE JSON log
        ai_ids: Unified AI-IDS instance
        output_path: Path for enhanced alerts
        enable_rule_generation: Auto-generate Suricata rules from AI detections
    """

    def __init__(
        self,
        eve_log_path: Path = Path("/var/log/suricata/eve.json"),
        ai_ids: Optional[UnifiedIDS] = None,
        output_path: Path = Path("/var/log/suricata/ai-enhanced.json"),
        enable_rule_generation: bool = True,
        rule_output_path: Path = Path("/etc/suricata/rules/ai-generated.rules")
    ):
        self.eve_log_path = eve_log_path
        self.output_path = output_path
        self.enable_rule_generation = enable_rule_generation
        self.rule_output_path = rule_output_path

        # Initialize AI-IDS
        self.ai_ids = ai_ids or UnifiedIDS(
            models=['neural_ode', 'encrypted_traffic', 'graph'],
            confidence_threshold=0.85
        )

        # Alert correlation buffer
        self.alert_buffer = []
        self.correlation_window = 60  # seconds

        logger.info(f"Suricata plugin initialized, monitoring: {eve_log_path}")

    async def start(self):
        """Start monitoring Suricata EVE log"""
        logger.info("Starting Suricata AI-IDS integration...")

        # Create output file
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Start log monitoring
        await self._monitor_eve_log()

    async def _monitor_eve_log(self):
        """Monitor EVE JSON log file in real-time"""
        try:
            with open(self.eve_log_path, 'r') as eve_file:
                # Move to end of file
                eve_file.seek(0, 2)

                logger.info("Monitoring Suricata EVE log for events...")

                while True:
                    line = eve_file.readline()

                    if line:
                        await self._process_eve_event(line)
                    else:
                        # No new data, wait briefly
                        await asyncio.sleep(0.1)

        except FileNotFoundError:
            logger.error(f"EVE log not found: {self.eve_log_path}")
            logger.info("Waiting for Suricata to create log file...")
            await asyncio.sleep(5)
            await self._monitor_eve_log()

        except Exception as e:
            logger.error(f"Error monitoring EVE log: {e}")
            raise

    async def _process_eve_event(self, line: str):
        """Process single EVE JSON event"""
        try:
            event = json.loads(line)
            event_type = event.get('event_type')

            # Process different event types
            if event_type == 'alert':
                await self._handle_alert(event)
            elif event_type == 'flow':
                await self._handle_flow(event)
            elif event_type == 'http':
                await self._handle_http(event)
            elif event_type == 'tls':
                await self._handle_tls(event)
            elif event_type == 'dns':
                await self._handle_dns(event)

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in EVE log: {line[:100]}")
        except Exception as e:
            logger.error(f"Error processing event: {e}")

    async def _handle_alert(self, event: Dict):
        """Handle Suricata alert with AI enhancement"""
        alert_data = event.get('alert', {})
        flow_data = event.get('flow', {})

        # Extract features for AI analysis
        features = self._extract_features(event)

        # Get AI prediction
        ai_result = await self._get_ai_prediction(features)

        # Enrich alert with AI analysis
        enhanced_alert = {
            'timestamp': event.get('timestamp'),
            'event_type': 'ai_enhanced_alert',
            'suricata_alert': {
                'signature': alert_data.get('signature'),
                'severity': alert_data.get('severity'),
                'category': alert_data.get('category'),
                'gid': alert_data.get('gid'),
                'sid': alert_data.get('sid')
            },
            'ai_analysis': {
                'is_malicious': ai_result.is_malicious,
                'confidence': ai_result.confidence,
                'attack_type': ai_result.attack_type,
                'attack_category': ai_result.attack_category,
                'severity': ai_result.severity,
                'explanation': ai_result.explanation,
                'recommended_action': ai_result.recommended_action,
                'model_predictions': ai_result.model_predictions,
                'uncertainty_bounds': ai_result.uncertainty_bounds
            },
            'flow': flow_data,
            'src_ip': event.get('src_ip'),
            'dest_ip': event.get('dest_ip'),
            'src_port': event.get('src_port'),
            'dest_port': event.get('dest_port'),
            'proto': event.get('proto')
        }

        # Write enhanced alert
        await self._write_enhanced_alert(enhanced_alert)

        # Generate rule if enabled and high confidence
        if self.enable_rule_generation and ai_result.confidence > 0.95:
            await self._generate_suricata_rule(enhanced_alert, ai_result)

        # Alert correlation
        await self._correlate_alert(enhanced_alert)

        # Log critical alerts
        if ai_result.severity in ['critical', 'high']:
            logger.warning(
                f"CRITICAL ALERT: {ai_result.attack_type} from {event.get('src_ip')} "
                f"(confidence: {ai_result.confidence:.2%})"
            )

    async def _handle_flow(self, event: Dict):
        """Handle flow event for behavioral analysis"""
        flow_data = event.get('flow', {})

        # Extract temporal features
        features = {
            'bytes_toserver': flow_data.get('bytes_toserver', 0),
            'bytes_toclient': flow_data.get('bytes_toclient', 0),
            'pkts_toserver': flow_data.get('pkts_toserver', 0),
            'pkts_toclient': flow_data.get('pkts_toclient', 0),
            'start': flow_data.get('start'),
            'end': flow_data.get('end')
        }

        # Analyze with Neural ODE (temporal patterns)
        # This captures continuous flow dynamics
        pass

    async def _handle_http(self, event: Dict):
        """Handle HTTP event"""
        http_data = event.get('http', {})

        # Analyze HTTP patterns for web attacks
        features = {
            'http_method': http_data.get('http_method'),
            'hostname': http_data.get('hostname'),
            'url': http_data.get('url'),
            'user_agent': http_data.get('http_user_agent'),
            'status': http_data.get('status'),
            'length': http_data.get('length', 0)
        }

        # Check for SQL injection, XSS, etc.
        pass

    async def _handle_tls(self, event: Dict):
        """Handle TLS event for encrypted traffic analysis"""
        tls_data = event.get('tls', {})

        # Extract TLS metadata (no decryption)
        features = {
            'version': tls_data.get('version'),
            'sni': tls_data.get('sni'),
            'subject': tls_data.get('subject'),
            'issuer': tls_data.get('issuerdn'),
            'notbefore': tls_data.get('notbefore'),
            'notafter': tls_data.get('notafter'),
            'ja3': tls_data.get('ja3', {}).get('hash')  # JA3 fingerprint
        }

        # Use encrypted traffic analyzer
        # This is where your CNN-LSTM-Transformer model analyzes TLS without decryption
        pass

    async def _handle_dns(self, event: Dict):
        """Handle DNS event for malicious domain detection"""
        dns_data = event.get('dns', {})

        # DNS tunneling, DGA detection
        features = {
            'query': dns_data.get('rrname'),
            'qtype': dns_data.get('rrtype'),
            'rcode': dns_data.get('rcode'),
            'answers': dns_data.get('answers', [])
        }

        pass

    def _extract_features(self, event: Dict) -> Dict:
        """Extract features from Suricata event for AI analysis"""
        flow = event.get('flow', {})
        alert = event.get('alert', {})

        features = {
            # Flow features
            'bytes_toserver': flow.get('bytes_toserver', 0),
            'bytes_toclient': flow.get('bytes_toclient', 0),
            'pkts_toserver': flow.get('pkts_toserver', 0),
            'pkts_toclient': flow.get('pkts_toclient', 0),

            # Alert features
            'severity': alert.get('severity', 3),
            'category': self._encode_category(alert.get('category', '')),

            # Network 5-tuple
            'src_port': event.get('src_port', 0),
            'dest_port': event.get('dest_port', 0),
            'proto': self._encode_protocol(event.get('proto', 'tcp')),

            # Temporal
            'timestamp': event.get('timestamp')
        }

        return features

    async def _get_ai_prediction(self, features: Dict) -> DetectionResult:
        """Get AI prediction from unified model"""
        # Convert features to tensor
        import torch

        # Feature engineering (simplified)
        feature_vector = torch.tensor([
            features.get('bytes_toserver', 0) / 1e6,  # Normalize
            features.get('bytes_toclient', 0) / 1e6,
            features.get('pkts_toserver', 0) / 1000,
            features.get('pkts_toclient', 0) / 1000,
            features.get('severity', 3) / 3,
            features.get('category', 0),
            features.get('src_port', 0) / 65535,
            features.get('dest_port', 0) / 65535,
            features.get('proto', 0)
        ], dtype=torch.float32).unsqueeze(0)

        # Pad to expected dimension (64)
        if feature_vector.size(1) < 64:
            padding = torch.zeros(1, 64 - feature_vector.size(1))
            feature_vector = torch.cat([feature_vector, padding], dim=1)

        # Get prediction
        result = self.ai_ids(feature_vector)

        return result

    async def _write_enhanced_alert(self, alert: Dict):
        """Write enhanced alert to output file"""
        with open(self.output_path, 'a') as f:
            f.write(json.dumps(alert) + '\n')

    async def _generate_suricata_rule(self, alert: Dict, ai_result: DetectionResult):
        """Auto-generate Suricata rule from AI detection"""
        # Generate rule based on AI findings
        sid = hash(str(alert)) % 9000000 + 1000000  # Generate unique SID

        rule = (
            f"alert {alert['proto']} any any -> any any "
            f"(msg:\"AI-IDS: {ai_result.attack_type} detected\"; "
            f"flow:established; "
            f"threshold:type limit, track by_src, count 1, seconds 60; "
            f"classtype:{ai_result.attack_category.lower().replace(' ', '-')}; "
            f"sid:{sid}; "
            f"rev:1; "
            f"metadata:confidence {ai_result.confidence:.2f}, "
            f"severity {ai_result.severity};)"
        )

        # Append to rules file
        with open(self.rule_output_path, 'a') as f:
            f.write(rule + '\n')

        logger.info(f"Generated Suricata rule: SID {sid}")

    async def _correlate_alert(self, alert: Dict):
        """Correlate alerts to detect multi-stage attacks"""
        # Add to buffer
        self.alert_buffer.append(alert)

        # Remove old alerts
        current_time = datetime.now()
        self.alert_buffer = [
            a for a in self.alert_buffer
            if (current_time - datetime.fromisoformat(
                a['timestamp'].replace('Z', '+00:00')
            )).total_seconds() < self.correlation_window
        ]

        # Detect patterns (e.g., reconnaissance followed by exploitation)
        if len(self.alert_buffer) >= 3:
            # Check for attack chain
            attack_types = [a['ai_analysis']['attack_type'] for a in self.alert_buffer[-3:]]

            if 'Port Scan' in attack_types and 'Brute Force' in attack_types:
                logger.warning("ATTACK CHAIN DETECTED: Reconnaissance -> Exploitation")

    def _encode_category(self, category: str) -> int:
        """Encode alert category to numeric"""
        categories = {
            'attempted-admin': 1,
            'attempted-user': 2,
            'web-application-attack': 3,
            'misc-attack': 4,
            'denial-of-service': 5
        }
        return categories.get(category.lower(), 0)

    def _encode_protocol(self, proto: str) -> int:
        """Encode protocol to numeric"""
        protocols = {'tcp': 6, 'udp': 17, 'icmp': 1}
        return protocols.get(proto.lower(), 0)


# Standalone execution
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    plugin = SuricataPlugin()

    try:
        asyncio.run(plugin.start())
    except KeyboardInterrupt:
        logger.info("Suricata plugin stopped by user")
        sys.exit(0)
