"""
Traffic Capture Service
========================

Captures network packets and forwards them to the detection engine.

Supports multiple deployment modes:
- Passive monitoring (SPAN port)
- Inline prevention (gateway mode)
- Agent-based (distributed capture)

Author: Roger Nick Anaedevha
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import base64
import json

try:
    from scapy.all import (
        sniff, IP, TCP, UDP, ICMP, Raw,
        wrpcap, rdpcap, AsyncSniffer
    )
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logging.warning("Scapy not available. Install with: pip install scapy")

from app.core.config import settings
from app.services.detection_engine import DetectionEngine
from app.db.session import get_db
from app.models.detection import NetworkTraffic
from app.services.geoip import get_geoip_info


logger = logging.getLogger(__name__)


@dataclass
class PacketMetadata:
    """Metadata extracted from network packet"""
    timestamp: datetime
    src_ip: str
    dst_ip: str
    src_port: Optional[int]
    dst_port: Optional[int]
    protocol: str
    packets: int
    bytes: int
    duration: float
    tcp_flags: Optional[str]
    payload_size: int
    payload_entropy: float
    is_encrypted: bool
    tls_version: Optional[str]
    ja3_fingerprint: Optional[str]
    raw_packet: Optional[str]
    features: Dict[str, Any]


class TrafficCapture:
    """
    Network traffic capture service

    Captures packets from network interface and processes them
    for threat detection.
    """

    def __init__(
        self,
        interface: str = None,
        filter_expr: str = None,
        buffer_size: int = None,
        detection_engine: DetectionEngine = None
    ):
        """
        Initialize traffic capture

        Args:
            interface: Network interface to capture from
            filter_expr: BPF filter expression
            buffer_size: Capture buffer size
            detection_engine: Detection engine instance
        """
        self.interface = interface or settings.CAPTURE_INTERFACE
        self.filter_expr = filter_expr or settings.PCAP_FILTER
        self.buffer_size = buffer_size or settings.CAPTURE_BUFFER_SIZE

        self.detection_engine = detection_engine or DetectionEngine()

        self.is_running = False
        self.sniffer = None
        self.packet_count = 0
        self.bytes_captured = 0

        # Packet buffer for batch processing
        self.packet_buffer: List[PacketMetadata] = []
        self.buffer_max_size = settings.STREAMING_BATCH_SIZE

        logger.info(
            f"Traffic capture initialized: "
            f"interface={self.interface}, filter={self.filter_expr}"
        )

    async def start(self):
        """Start packet capture"""
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy not installed. Cannot capture traffic.")

        self.is_running = True

        logger.info(f"Starting packet capture on interface: {self.interface}")

        try:
            # Start async sniffer
            self.sniffer = AsyncSniffer(
                iface=self.interface,
                filter=self.filter_expr,
                prn=self._packet_callback,
                store=False
            )
            self.sniffer.start()

            # Start batch processor
            asyncio.create_task(self._batch_processor())

            logger.info("✅ Packet capture started successfully")

        except Exception as e:
            logger.error(f"Failed to start packet capture: {e}")
            self.is_running = False
            raise

    async def stop(self):
        """Stop packet capture"""
        self.is_running = False

        if self.sniffer:
            self.sniffer.stop()

        logger.info(
            f"Packet capture stopped. "
            f"Captured {self.packet_count} packets, {self.bytes_captured} bytes"
        )

    def _packet_callback(self, packet):
        """
        Callback for each captured packet

        Args:
            packet: Scapy packet object
        """
        try:
            # Extract metadata
            metadata = self._extract_packet_metadata(packet)

            # Add to buffer
            self.packet_buffer.append(metadata)

            # Update counters
            self.packet_count += 1
            self.bytes_captured += metadata.bytes

            # Log every 1000 packets
            if self.packet_count % 1000 == 0:
                logger.info(
                    f"Captured {self.packet_count} packets "
                    f"({self.bytes_captured / 1024 / 1024:.2f} MB)"
                )

        except Exception as e:
            logger.error(f"Error processing packet: {e}", exc_info=True)

    def _extract_packet_metadata(self, packet) -> PacketMetadata:
        """
        Extract metadata from packet

        Args:
            packet: Scapy packet

        Returns:
            PacketMetadata object
        """
        # Basic info
        timestamp = datetime.now()
        src_ip = None
        dst_ip = None
        src_port = None
        dst_port = None
        protocol = "unknown"
        tcp_flags = None
        payload_size = 0
        payload_entropy = 0.0
        is_encrypted = False
        tls_version = None

        # Extract IP layer
        if packet.haslayer(IP):
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst

        # Extract transport layer
        if packet.haslayer(TCP):
            protocol = "tcp"
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
            tcp_flags = self._get_tcp_flags(packet[TCP])

            # Check for TLS
            if dst_port in [443, 8443] or src_port in [443, 8443]:
                is_encrypted = True
                # Try to extract TLS version (simplified)
                if packet.haslayer(Raw):
                    raw_data = bytes(packet[Raw].load)
                    if len(raw_data) > 5 and raw_data[0] == 0x16:
                        tls_version = f"TLS 1.{raw_data[2]}"

        elif packet.haslayer(UDP):
            protocol = "udp"
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport

        elif packet.haslayer(ICMP):
            protocol = "icmp"

        # Extract payload
        if packet.haslayer(Raw):
            payload = bytes(packet[Raw].load)
            payload_size = len(payload)
            payload_entropy = self._calculate_entropy(payload)

        # Encode raw packet
        raw_packet_b64 = base64.b64encode(bytes(packet)).decode('utf-8')

        # Extract features for ML
        features = {
            "packet_length": len(packet),
            "header_length": len(packet) - payload_size,
            "payload_size": payload_size,
            "payload_entropy": payload_entropy,
            "protocol_number": packet[IP].proto if packet.haslayer(IP) else 0,
            "ttl": packet[IP].ttl if packet.haslayer(IP) else 0,
            "tcp_window": packet[TCP].window if packet.haslayer(TCP) else 0,
            "tcp_flags_int": self._tcp_flags_to_int(tcp_flags) if tcp_flags else 0,
            "is_encrypted": int(is_encrypted),
        }

        return PacketMetadata(
            timestamp=timestamp,
            src_ip=src_ip or "0.0.0.0",
            dst_ip=dst_ip or "0.0.0.0",
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            packets=1,
            bytes=len(packet),
            duration=0.0,
            tcp_flags=tcp_flags,
            payload_size=payload_size,
            payload_entropy=payload_entropy,
            is_encrypted=is_encrypted,
            tls_version=tls_version,
            ja3_fingerprint=None,  # TODO: Implement JA3 extraction
            raw_packet=raw_packet_b64,
            features=features
        )

    async def _batch_processor(self):
        """
        Process packets in batches for efficiency
        """
        logger.info("Batch processor started")

        while self.is_running:
            try:
                # Wait for buffer to fill or timeout
                await asyncio.sleep(1.0)

                if len(self.packet_buffer) >= self.buffer_max_size:
                    # Extract batch
                    batch = self.packet_buffer[:self.buffer_max_size]
                    self.packet_buffer = self.packet_buffer[self.buffer_max_size:]

                    # Process batch
                    await self._process_batch(batch)

            except Exception as e:
                logger.error(f"Error in batch processor: {e}", exc_info=True)

        logger.info("Batch processor stopped")

    async def _process_batch(self, batch: List[PacketMetadata]):
        """
        Process batch of packets through detection engine

        Args:
            batch: List of PacketMetadata
        """
        try:
            logger.debug(f"Processing batch of {len(batch)} packets")

            # Store in database
            async for db in get_db():
                # Create NetworkTraffic records
                traffic_records = []
                for metadata in batch:
                    # Get GeoIP info
                    src_geo = get_geoip_info(metadata.src_ip)
                    dst_geo = get_geoip_info(metadata.dst_ip)

                    traffic = NetworkTraffic(
                        timestamp=metadata.timestamp,
                        src_ip=metadata.src_ip,
                        dst_ip=metadata.dst_ip,
                        src_port=metadata.src_port,
                        dst_port=metadata.dst_port,
                        protocol=metadata.protocol,
                        packets=metadata.packets,
                        bytes_transferred=metadata.bytes,
                        duration=metadata.duration,
                        tcp_flags=metadata.tcp_flags,
                        payload_size=metadata.payload_size,
                        payload_entropy=metadata.payload_entropy,
                        is_encrypted=metadata.is_encrypted,
                        tls_version=metadata.tls_version,
                        ja3_fingerprint=metadata.ja3_fingerprint,
                        src_country=src_geo.get('country_code'),
                        src_city=src_geo.get('city'),
                        dst_country=dst_geo.get('country_code'),
                        dst_city=dst_geo.get('city'),
                        raw_packet=metadata.raw_packet[:10000],  # Limit size
                        features=metadata.features
                    )
                    traffic_records.append(traffic)

                # Bulk insert
                db.bulk_save_objects(traffic_records)
                await db.commit()

                # Get IDs for detection
                for traffic in traffic_records:
                    await db.refresh(traffic)

                logger.debug(f"Stored {len(traffic_records)} traffic records")

                # Run detection
                for traffic in traffic_records:
                    await self.detection_engine.detect(traffic.id, db)

        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)

    @staticmethod
    def _get_tcp_flags(tcp_layer) -> str:
        """Extract TCP flags as string"""
        flags = []
        if tcp_layer.flags.F:
            flags.append('FIN')
        if tcp_layer.flags.S:
            flags.append('SYN')
        if tcp_layer.flags.R:
            flags.append('RST')
        if tcp_layer.flags.P:
            flags.append('PSH')
        if tcp_layer.flags.A:
            flags.append('ACK')
        if tcp_layer.flags.U:
            flags.append('URG')
        return ','.join(flags)

    @staticmethod
    def _tcp_flags_to_int(flags_str: Optional[str]) -> int:
        """Convert TCP flags string to integer"""
        if not flags_str:
            return 0

        flags_map = {
            'FIN': 0x01,
            'SYN': 0x02,
            'RST': 0x04,
            'PSH': 0x08,
            'ACK': 0x10,
            'URG': 0x20
        }

        value = 0
        for flag in flags_str.split(','):
            value |= flags_map.get(flag, 0)

        return value

    @staticmethod
    def _calculate_entropy(data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        if len(data) == 0:
            return 0.0

        import math

        # Count byte frequencies
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1

        # Calculate entropy
        entropy = 0.0
        data_len = len(data)

        for count in byte_counts:
            if count > 0:
                probability = count / data_len
                entropy -= probability * math.log2(probability)

        return entropy


class CaptureAgent:
    """
    Standalone capture agent for distributed deployment

    Can be deployed on network segments to capture and forward
    traffic to central RobustIDPS.ai server via API.
    """

    def __init__(
        self,
        server_url: str,
        api_token: str,
        interface: str = "eth0",
        batch_size: int = 100
    ):
        """
        Initialize capture agent

        Args:
            server_url: RobustIDPS.ai server URL
            api_token: API authentication token
            interface: Network interface to monitor
            batch_size: Number of packets to batch before sending
        """
        self.server_url = server_url.rstrip('/')
        self.api_token = api_token
        self.interface = interface
        self.batch_size = batch_size

        self.capture = TrafficCapture(
            interface=interface,
            detection_engine=None  # Agent doesn't run detection locally
        )

        logger.info(
            f"Capture agent initialized: "
            f"server={server_url}, interface={interface}"
        )

    async def start(self):
        """Start capture agent"""
        logger.info("Starting capture agent...")

        # Override batch processor to send to API
        self.capture._process_batch = self._forward_to_api

        await self.capture.start()

    async def _forward_to_api(self, batch: List[PacketMetadata]):
        """
        Forward captured packets to API

        Args:
            batch: List of PacketMetadata
        """
        import aiohttp

        try:
            # Convert to JSON
            payload = {
                "packets": [
                    {
                        "timestamp": metadata.timestamp.isoformat(),
                        "src_ip": metadata.src_ip,
                        "dst_ip": metadata.dst_ip,
                        "src_port": metadata.src_port,
                        "dst_port": metadata.dst_port,
                        "protocol": metadata.protocol,
                        "bytes": metadata.bytes,
                        "features": metadata.features
                    }
                    for metadata in batch
                ]
            }

            # Send to API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.server_url}/api/v1/traffic/ingest",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_token}"}
                ) as response:
                    if response.status == 200:
                        logger.debug(f"Successfully forwarded {len(batch)} packets")
                    else:
                        logger.error(
                            f"Failed to forward packets: "
                            f"{response.status} - {await response.text()}"
                        )

        except Exception as e:
            logger.error(f"Error forwarding packets to API: {e}", exc_info=True)


# Example usage
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check mode
    if len(sys.argv) > 1 and sys.argv[1] == "agent":
        # Agent mode: forward to API
        agent = CaptureAgent(
            server_url="https://robustidps.ai",
            api_token="your-api-token",
            interface="eth0"
        )
        asyncio.run(agent.start())
    else:
        # Local mode: capture and detect
        capture = TrafficCapture()
        asyncio.run(capture.start())
