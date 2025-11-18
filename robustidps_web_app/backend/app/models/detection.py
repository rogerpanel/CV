"""
Database Models for Detections and Alerts
==========================================

SQLAlchemy ORM models for storing network traffic, detections, and alerts.
"""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Index, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

from app.db.base_class import Base


class SeverityLevel(str, enum.Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AttackType(str, enum.Enum):
    """Attack classification types"""
    DDOS = "ddos"
    PORT_SCAN = "port_scan"
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    MALWARE = "malware"
    RANSOMWARE = "ransomware"
    BACKDOOR = "backdoor"
    BOTNET = "botnet"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    ZERO_DAY = "zero_day"
    BENIGN = "benign"
    UNKNOWN = "unknown"


class DetectionStatus(str, enum.Enum):
    """Detection processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ActionStatus(str, enum.Enum):
    """Prevention action status"""
    PENDING = "pending"
    EXECUTED = "executed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class NetworkTraffic(Base):
    """Raw network traffic data"""

    __tablename__ = "network_traffic"

    id = Column(Integer, primary_key=True, index=True)

    # Timestamps
    timestamp = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now())

    # Network Information
    src_ip = Column(String(45), nullable=False, index=True)  # IPv4/IPv6
    dst_ip = Column(String(45), nullable=False, index=True)
    src_port = Column(Integer, nullable=True)
    dst_port = Column(Integer, nullable=True)
    protocol = Column(String(20), nullable=False)  # tcp, udp, icmp, etc.

    # Traffic Metrics
    packets = Column(Integer, default=0)
    bytes_transferred = Column(Integer, default=0)
    duration = Column(Float, default=0.0)  # seconds

    # Packet Details
    tcp_flags = Column(String(50), nullable=True)
    payload_size = Column(Integer, default=0)
    payload_entropy = Column(Float, nullable=True)

    # TLS/Encryption
    is_encrypted = Column(Boolean, default=False)
    tls_version = Column(String(20), nullable=True)
    ja3_fingerprint = Column(String(64), nullable=True)
    ja3s_fingerprint = Column(String(64), nullable=True)

    # Geographic
    src_country = Column(String(2), nullable=True)
    src_city = Column(String(100), nullable=True)
    dst_country = Column(String(2), nullable=True)
    dst_city = Column(String(100), nullable=True)

    # Raw Data
    raw_packet = Column(Text, nullable=True)  # Base64 encoded PCAP
    features = Column(JSON, nullable=True)  # Extracted ML features

    # Relationships
    detections = relationship("Detection", back_populates="traffic")

    # Indexes
    __table_args__ = (
        Index('idx_traffic_timestamp', 'timestamp'),
        Index('idx_traffic_src_ip', 'src_ip'),
        Index('idx_traffic_dst_ip', 'dst_ip'),
        Index('idx_traffic_protocol', 'protocol'),
        Index('idx_traffic_created', 'created_at'),
    )


class Detection(Base):
    """AI detection results"""

    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)

    # Traffic Reference
    traffic_id = Column(Integer, ForeignKey("network_traffic.id"), nullable=False)

    # Timestamps
    detected_at = Column(DateTime, server_default=func.now(), index=True)
    processed_at = Column(DateTime, nullable=True)

    # Detection Results
    is_malicious = Column(Boolean, nullable=False, index=True)
    attack_type = Column(SQLEnum(AttackType), nullable=False, index=True)
    confidence = Column(Float, nullable=False)  # 0.0 to 1.0
    severity = Column(SQLEnum(SeverityLevel), nullable=False, index=True)

    # Model Information
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=True)

    # Multi-Model Predictions
    model_predictions = Column(JSON, nullable=True)  # Individual model results
    ensemble_weights = Column(JSON, nullable=True)

    # Uncertainty Quantification
    epistemic_uncertainty = Column(Float, nullable=True)
    aleatoric_uncertainty = Column(Float, nullable=True)
    uncertainty_threshold = Column(Float, default=0.1)

    # Explainability
    feature_importance = Column(JSON, nullable=True)  # SHAP values
    explanation = Column(Text, nullable=True)

    # Processing Status
    status = Column(SQLEnum(DetectionStatus), default=DetectionStatus.PENDING)
    processing_time_ms = Column(Float, nullable=True)

    # False Positive Feedback
    is_false_positive = Column(Boolean, nullable=True)
    user_feedback = Column(Text, nullable=True)
    feedback_timestamp = Column(DateTime, nullable=True)

    # Relationships
    traffic = relationship("NetworkTraffic", back_populates="detections")
    alerts = relationship("Alert", back_populates="detection")
    prevention_actions = relationship("PreventionAction", back_populates="detection")

    # Indexes
    __table_args__ = (
        Index('idx_detection_timestamp', 'detected_at'),
        Index('idx_detection_malicious', 'is_malicious'),
        Index('idx_detection_severity', 'severity'),
        Index('idx_detection_attack_type', 'attack_type'),
    )


class Alert(Base):
    """Security alerts generated from detections"""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)

    # Detection Reference
    detection_id = Column(Integer, ForeignKey("detections.id"), nullable=False)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), index=True)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    # Alert Information
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    severity = Column(SQLEnum(SeverityLevel), nullable=False, index=True)

    # Status
    is_acknowledged = Column(Boolean, default=False, index=True)
    is_resolved = Column(Boolean, default=False, index=True)
    acknowledged_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    resolved_by = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Notification Status
    email_sent = Column(Boolean, default=False)
    slack_sent = Column(Boolean, default=False)
    sms_sent = Column(Boolean, default=False)
    webhook_sent = Column(Boolean, default=False)

    # Deduplication
    fingerprint = Column(String(64), nullable=True, index=True)
    duplicate_count = Column(Integer, default=1)
    last_occurrence = Column(DateTime, nullable=True)

    # Investigation
    investigation_notes = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)
    assigned_to = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Relationships
    detection = relationship("Detection", back_populates="alerts")

    # Indexes
    __table_args__ = (
        Index('idx_alert_created', 'created_at'),
        Index('idx_alert_severity', 'severity'),
        Index('idx_alert_acknowledged', 'is_acknowledged'),
        Index('idx_alert_resolved', 'is_resolved'),
    )


class PreventionAction(Base):
    """Automated prevention actions taken"""

    __tablename__ = "prevention_actions"

    id = Column(Integer, primary_key=True, index=True)

    # Detection Reference
    detection_id = Column(Integer, ForeignKey("detections.id"), nullable=False)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    executed_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)

    # Action Details
    action_type = Column(String(50), nullable=False)  # block_ip, rate_limit, quarantine
    target = Column(String(255), nullable=False)  # IP, subnet, user, etc.
    parameters = Column(JSON, nullable=True)

    # Status
    status = Column(SQLEnum(ActionStatus), default=ActionStatus.PENDING)
    error_message = Column(Text, nullable=True)

    # Firewall Details
    firewall_rule_id = Column(String(100), nullable=True)
    firewall_backend = Column(String(50), nullable=True)

    # Auto-Rollback
    auto_rollback = Column(Boolean, default=True)
    rollback_at = Column(DateTime, nullable=True)
    rolled_back_at = Column(DateTime, nullable=True)

    # Audit
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    approved_by = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Relationships
    detection = relationship("Detection", back_populates="prevention_actions")

    # Indexes
    __table_args__ = (
        Index('idx_action_created', 'created_at'),
        Index('idx_action_status', 'status'),
        Index('idx_action_type', 'action_type'),
    )


class User(Base):
    """User accounts for authentication"""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    # Basic Info
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)

    # Profile
    full_name = Column(String(255), nullable=True)
    organization = Column(String(255), nullable=True)

    # Status
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)

    # Authentication
    last_login = Column(DateTime, nullable=True)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)

    # MFA
    mfa_secret = Column(String(255), nullable=True)
    mfa_enabled = Column(Boolean, default=False)

    # API Keys
    api_key = Column(String(255), unique=True, nullable=True, index=True)
    api_key_created = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())


class SystemMetric(Base):
    """System performance and health metrics"""

    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, index=True)

    # Timestamp
    timestamp = Column(DateTime, server_default=func.now(), index=True)

    # Performance Metrics
    detections_per_second = Column(Float, default=0.0)
    avg_detection_latency_ms = Column(Float, default=0.0)
    queue_size = Column(Integer, default=0)

    # Resource Usage
    cpu_usage_percent = Column(Float, default=0.0)
    memory_usage_percent = Column(Float, default=0.0)
    gpu_usage_percent = Column(Float, nullable=True)

    # Detection Statistics
    total_detections = Column(Integer, default=0)
    malicious_count = Column(Integer, default=0)
    benign_count = Column(Integer, default=0)

    # Model Performance
    accuracy = Column(Float, nullable=True)
    false_positive_rate = Column(Float, nullable=True)
    false_negative_rate = Column(Float, nullable=True)

    # System Health
    database_connections = Column(Integer, default=0)
    redis_connections = Column(Integer, default=0)
    active_workers = Column(Integer, default=0)

    # Indexes
    __table_args__ = (
        Index('idx_metrics_timestamp', 'timestamp'),
    )
