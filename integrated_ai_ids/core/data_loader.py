"""
Multi-Dataset Data Loader for Integrated AI-IDS
================================================

Unified data loader supporting multiple IDS datasets:
- Cloud security logs (AWS, Azure, GCP)
- Network traffic (PCAP, NetFlow, CSV)
- Encrypted traffic (TLS metadata)
- API logs (REST, GraphQL)
- Container/Kubernetes logs
- IoT/IIoT data

Author: Roger Nick Anaedevha
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import logging
from scapy.all import rdpcap, IP, TCP, UDP
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


class UnifiedDataLoader:
    """
    Unified data loader for multiple IDS dataset formats

    Supports:
    - CIC-IDS2018, UNSW-NB15, CIC-IoT-2023 (CSV)
    - ICS3D (Container, IoT, Enterprise)
    - PCAP files
    - Real-time network streams
    - Cloud logs (JSON)
    - API logs
    """

    def __init__(
        self,
        dataset_type: str = 'csv',
        normalize: bool = True,
        categorical_encoding: str = 'label'
    ):
        self.dataset_type = dataset_type
        self.normalize = normalize
        self.categorical_encoding = categorical_encoding

        # Initialize scalers
        self.scaler = StandardScaler()
        self.label_encoders = {}

        # Feature mappings for different datasets
        self.feature_maps = self._initialize_feature_maps()

    def _initialize_feature_maps(self) -> Dict:
        """Initialize feature mappings for different datasets"""
        return {
            'cic-ids2018': {
                'flow_duration': 'Flow Duration',
                'total_fwd_packets': 'Total Fwd Packets',
                'total_bwd_packets': 'Total Backward Packets',
                'flow_bytes_per_sec': 'Flow Bytes/s',
                'flow_packets_per_sec': 'Flow Packets/s',
                'label': 'Label'
            },
            'unsw-nb15': {
                'dur': 'dur',
                'sbytes': 'sbytes',
                'dbytes': 'dbytes',
                'sttl': 'sttl',
                'dttl': 'dttl',
                'label': 'attack_cat'
            },
            'ics3d-container': {
                'duration': 'duration',
                'protocol': 'protocol',
                'src_port': 'src_port',
                'dst_port': 'dst_port',
                'bytes': 'tot_bytes',
                'packets': 'tot_pkts',
                'label': 'label'
            }
        }

    def load_csv_dataset(
        self,
        file_path: Path,
        dataset_name: str = 'cic-ids2018'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load CSV-based IDS dataset

        Args:
            file_path: Path to CSV file
            dataset_name: Dataset identifier for feature mapping

        Returns:
            (features, labels) as torch tensors
        """
        logger.info(f"Loading CSV dataset from {file_path}")

        try:
            # Read CSV
            df = pd.read_csv(file_path)

            # Get feature map
            feature_map = self.feature_maps.get(dataset_name, {})

            # Extract label
            label_col = feature_map.get('label', 'Label')
            if label_col in df.columns:
                labels = df[label_col].values
                df = df.drop(columns=[label_col])
            else:
                labels = None

            # Handle categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))

            # Handle missing values
            df = df.fillna(0)

            # Replace inf values
            df = df.replace([np.inf, -np.inf], 0)

            # Extract features
            features = df.values.astype(np.float32)

            # Normalize
            if self.normalize:
                features = self.scaler.fit_transform(features)

            # Convert to tensors
            features_tensor = torch.tensor(features, dtype=torch.float32)

            if labels is not None:
                # Encode labels
                if not isinstance(labels[0], (int, np.integer)):
                    le = LabelEncoder()
                    labels = le.fit_transform(labels)
                labels_tensor = torch.tensor(labels, dtype=torch.long)
            else:
                labels_tensor = None

            logger.info(f"Loaded {len(features_tensor)} samples with {features_tensor.shape[1]} features")

            return features_tensor, labels_tensor

        except Exception as e:
            logger.error(f"Error loading CSV dataset: {e}")
            raise

    def load_pcap_file(self, file_path: Path) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Load PCAP file and extract features

        Args:
            file_path: Path to PCAP file

        Returns:
            (features, metadata) where metadata contains raw packet info
        """
        logger.info(f"Loading PCAP file from {file_path}")

        try:
            packets = rdpcap(str(file_path))

            features_list = []
            metadata_list = []

            for pkt in packets:
                if IP in pkt:
                    features, metadata = self._extract_packet_features(pkt)
                    if features is not None:
                        features_list.append(features)
                        metadata_list.append(metadata)

            if not features_list:
                raise ValueError("No valid packets found in PCAP")

            # Stack features
            features_array = np.array(features_list, dtype=np.float32)

            # Normalize
            if self.normalize:
                features_array = self.scaler.fit_transform(features_array)

            features_tensor = torch.tensor(features_array, dtype=torch.float32)

            logger.info(f"Extracted features from {len(features_tensor)} packets")

            return features_tensor, metadata_list

        except Exception as e:
            logger.error(f"Error loading PCAP file: {e}")
            raise

    def _extract_packet_features(self, pkt) -> Tuple[Optional[np.ndarray], Dict]:
        """Extract features from a single packet"""
        try:
            features = []
            metadata = {}

            # IP layer features
            if IP in pkt:
                ip = pkt[IP]
                features.extend([
                    len(pkt),  # Packet size
                    ip.len,     # IP length
                    ip.ttl,     # TTL
                    ip.proto,   # Protocol
                ])
                metadata.update({
                    'src_ip': ip.src,
                    'dst_ip': ip.dst,
                    'protocol': ip.proto
                })

            # TCP layer features
            if TCP in pkt:
                tcp = pkt[TCP]
                features.extend([
                    tcp.sport,      # Source port
                    tcp.dport,      # Destination port
                    tcp.flags,      # TCP flags
                    tcp.window,     # Window size
                    len(tcp.payload) if tcp.payload else 0  # Payload size
                ])
                metadata.update({
                    'src_port': tcp.sport,
                    'dst_port': tcp.dport,
                    'tcp_flags': tcp.flags
                })
            elif UDP in pkt:
                udp = pkt[UDP]
                features.extend([
                    udp.sport,      # Source port
                    udp.dport,      # Destination port
                    0, 0,           # Placeholder for TCP-specific
                    len(udp.payload) if udp.payload else 0
                ])
                metadata.update({
                    'src_port': udp.sport,
                    'dst_port': udp.dport
                })
            else:
                features.extend([0, 0, 0, 0, 0])  # No transport layer

            # Pad to fixed size (64 features)
            while len(features) < 64:
                features.append(0.0)

            return np.array(features[:64], dtype=np.float32), metadata

        except Exception as e:
            logger.warning(f"Error extracting packet features: {e}")
            return None, {}

    def load_cloud_logs(self, file_path: Path, cloud_provider: str = 'aws') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load cloud security logs (AWS CloudTrail, Azure Activity, GCP Audit)

        Args:
            file_path: Path to JSON log file
            cloud_provider: 'aws', 'azure', or 'gcp'

        Returns:
            (features, labels) as torch tensors
        """
        logger.info(f"Loading {cloud_provider.upper()} cloud logs from {file_path}")

        try:
            with open(file_path, 'r') as f:
                logs = json.load(f)

            features_list = []
            labels_list = []

            for log_entry in logs:
                features, label = self._extract_cloud_log_features(log_entry, cloud_provider)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(label)

            features_array = np.array(features_list, dtype=np.float32)
            labels_array = np.array(labels_list, dtype=np.int64)

            if self.normalize:
                features_array = self.scaler.fit_transform(features_array)

            return torch.tensor(features_array), torch.tensor(labels_array)

        except Exception as e:
            logger.error(f"Error loading cloud logs: {e}")
            raise

    def _extract_cloud_log_features(self, log_entry: Dict, provider: str) -> Tuple[Optional[np.ndarray], int]:
        """Extract features from cloud log entry"""
        features = []

        if provider == 'aws':
            # AWS CloudTrail features
            features.extend([
                hash(log_entry.get('eventName', '')) % 1000,
                hash(log_entry.get('userIdentity', {}).get('type', '')) % 100,
                1 if log_entry.get('errorCode') else 0,
                len(log_entry.get('resources', [])),
            ])
        elif provider == 'azure':
            # Azure Activity Log features
            features.extend([
                hash(log_entry.get('operationName', '')) % 1000,
                hash(log_entry.get('category', '')) % 100,
                1 if log_entry.get('resultType') == 'Failed' else 0,
                len(log_entry.get('properties', {})),
            ])
        elif provider == 'gcp':
            # GCP Audit Log features
            features.extend([
                hash(log_entry.get('protoPayload', {}).get('methodName', '')) % 1000,
                hash(log_entry.get('protoPayload', {}).get('serviceName', '')) % 100,
                1 if log_entry.get('severity') == 'ERROR' else 0,
                len(log_entry.get('protoPayload', {}).get('request', {})),
            ])

        # Pad to 64 features
        while len(features) < 64:
            features.append(0.0)

        # Simple labeling (in practice, use actual threat intelligence)
        label = 1 if any(suspicious in str(log_entry).lower() for suspicious in ['unauthorized', 'denied', 'error', 'failed']) else 0

        return np.array(features[:64], dtype=np.float32), label

    def load_encrypted_traffic(self, file_path: Path) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Load encrypted traffic metadata (TLS handshakes, cert info)

        Args:
            file_path: Path to PCAP or JSON with TLS metadata

        Returns:
            (features, metadata)
        """
        logger.info(f"Loading encrypted traffic from {file_path}")

        # For encrypted traffic, we extract metadata without decryption
        # Features: packet sizes, inter-arrival times, TLS version, cipher suites, etc.

        if file_path.suffix == '.pcap':
            return self.load_pcap_file(file_path)
        elif file_path.suffix == '.json':
            # Load TLS metadata from Suricata/Zeek logs
            with open(file_path, 'r') as f:
                tls_data = [json.loads(line) for line in f if line.strip()]

            features_list = []
            metadata_list = []

            for entry in tls_data:
                if 'tls' in entry:
                    features, metadata = self._extract_tls_features(entry['tls'])
                    features_list.append(features)
                    metadata_list.append(metadata)

            features_array = np.array(features_list, dtype=np.float32)
            if self.normalize:
                features_array = self.scaler.fit_transform(features_array)

            return torch.tensor(features_array), metadata_list

    def _extract_tls_features(self, tls_data: Dict) -> Tuple[np.ndarray, Dict]:
        """Extract features from TLS metadata"""
        features = []

        # TLS version
        version_map = {'TLS 1.0': 10, 'TLS 1.1': 11, 'TLS 1.2': 12, 'TLS 1.3': 13}
        features.append(version_map.get(tls_data.get('version'), 0))

        # Cipher suite (hash to numeric)
        features.append(hash(tls_data.get('cipher', '')) % 1000)

        # Certificate info
        features.append(len(tls_data.get('subject', '')))
        features.append(len(tls_data.get('issuer', '')))

        # JA3 fingerprint (if available)
        features.append(hash(tls_data.get('ja3', {}).get('hash', '')) % 10000)

        # SNI length
        features.append(len(tls_data.get('sni', '')))

        # Pad to 64
        while len(features) < 64:
            features.append(0.0)

        metadata = {
            'sni': tls_data.get('sni'),
            'ja3': tls_data.get('ja3', {}).get('hash'),
            'version': tls_data.get('version')
        }

        return np.array(features[:64], dtype=np.float32), metadata

    def create_dataloader(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        """
        Create PyTorch DataLoader

        Args:
            features: Feature tensor
            labels: Label tensor (optional)
            batch_size: Batch size
            shuffle: Whether to shuffle data

        Returns:
            DataLoader instance
        """
        if labels is not None:
            dataset = torch.utils.data.TensorDataset(features, labels)
        else:
            dataset = torch.utils.data.TensorDataset(features)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )


# Example usage
if __name__ == "__main__":
    loader = UnifiedDataLoader()

    # Load CIC-IDS2018
    # features, labels = loader.load_csv_dataset(
    #     Path("/data/CIC-IDS2018/Friday-WorkingHours.pcap_ISCX.csv"),
    #     dataset_name='cic-ids2018'
    # )

    # Load PCAP
    # features, metadata = loader.load_pcap_file(Path("/data/capture.pcap"))

    # Load cloud logs
    # features, labels = loader.load_cloud_logs(
    #     Path("/data/cloudtrail.json"),
    #     cloud_provider='aws'
    # )

    print("UnifiedDataLoader ready for use")
