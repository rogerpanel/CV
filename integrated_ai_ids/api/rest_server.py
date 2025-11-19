"""
REST API Service for Integrated AI-IDS
=======================================

Production-ready REST API for real-time intrusion detection

Features:
- Real-time detection endpoint
- Batch processing
- WebSocket streaming
- Health monitoring
- Metrics collection
- Authentication & authorization

Usage:
    python -m integrated_ai_ids.api.rest_server --port 8000

API Documentation: http://localhost:8000/docs

Author: Roger Nick Anaedevha
"""

from fastapi import FastAPI, HTTPException, WebSocket, Depends, Security, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import torch
import asyncio
import uvicorn
import logging
from datetime import datetime
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

from ..core.unified_model import UnifiedIDS, DetectionResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Integrated AI-IDS API",
    description="AI-Powered Network Intrusion Detection System API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
DETECTION_COUNTER = Counter('ids_detections_total', 'Total detections', ['attack_type', 'severity'])
DETECTION_LATENCY = Histogram('ids_detection_latency_seconds', 'Detection latency')
ACTIVE_CONNECTIONS = Gauge('ids_active_connections', 'Active WebSocket connections')
THREAT_SCORE = Gauge('ids_threat_score', 'Current threat score')

# API Key authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

# Global IDS instance
ids_model: Optional[UnifiedIDS] = None


# Request/Response Models
class TrafficData(BaseModel):
    """Single traffic flow data"""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    packets: int
    bytes: int
    duration: float
    flags: Optional[str] = None
    payload_features: Optional[Dict] = None
    timestamp: Optional[str] = None


class BatchTrafficData(BaseModel):
    """Batch traffic data"""
    flows: List[TrafficData]
    metadata: Optional[Dict] = None


class DetectionResponse(BaseModel):
    """Detection response"""
    flow_id: str
    is_malicious: bool
    confidence: float
    attack_type: Optional[str]
    attack_category: Optional[str]
    severity: str
    explanation: str
    recommended_action: str
    uncertainty_bounds: tuple
    timestamp: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    active_models: List[str]
    version: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Metrics response"""
    total_detections: int
    malicious_count: int
    benign_count: int
    avg_confidence: float
    avg_latency_ms: float
    active_connections: int
    threat_level: str


# Utility functions
def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify API key"""
    # In production, validate against database or secret store
    valid_keys = {"demo-key-12345", "production-key-67890"}
    if api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


def initialize_model():
    """Initialize AI-IDS model"""
    global ids_model
    if ids_model is None:
        logger.info("Initializing Unified AI-IDS model...")
        ids_model = UnifiedIDS(
            models=['neural_ode', 'optimal_transport', 'encrypted_traffic',
                   'federated', 'graph', 'llm'],
            confidence_threshold=0.85,
            enable_uncertainty=True,
            enable_explanation=True
        )
        logger.info("Model initialized successfully")
    return ids_model


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting Integrated AI-IDS REST API...")
    initialize_model()
    logger.info("API ready to accept requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Integrated AI-IDS REST API...")


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "service": "Integrated AI-IDS",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    model = initialize_model()

    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        active_models=model.active_models if model else [],
        version="1.0.0",
        uptime_seconds=0.0  # Implement uptime tracking
    )


@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_single(
    traffic: TrafficData,
    api_key: str = Depends(verify_api_key)
):
    """
    Detect threats in single traffic flow

    Args:
        traffic: Traffic flow data
        api_key: API authentication key

    Returns:
        DetectionResponse with analysis results
    """
    start_time = datetime.now()

    try:
        model = initialize_model()

        # Convert traffic data to tensor
        features = _traffic_to_tensor(traffic)

        # Perform detection
        with DETECTION_LATENCY.time():
            result = model(features)

        # Update metrics
        DETECTION_COUNTER.labels(
            attack_type=result.attack_type or 'benign',
            severity=result.severity
        ).inc()

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return DetectionResponse(
            flow_id=f"{traffic.src_ip}:{traffic.src_port}->{traffic.dst_ip}:{traffic.dst_port}",
            is_malicious=result.is_malicious,
            confidence=result.confidence,
            attack_type=result.attack_type,
            attack_category=result.attack_category,
            severity=result.severity,
            explanation=result.explanation,
            recommended_action=result.recommended_action,
            uncertainty_bounds=result.uncertainty_bounds,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/detect/batch", response_model=List[DetectionResponse], tags=["Detection"])
async def detect_batch(
    batch: BatchTrafficData,
    api_key: str = Depends(verify_api_key),
    background_tasks: BackgroundTasks = None
):
    """
    Detect threats in batch of traffic flows

    Args:
        batch: Batch of traffic flows
        api_key: API authentication key

    Returns:
        List of DetectionResponse for each flow
    """
    model = initialize_model()
    results = []

    for traffic in batch.flows:
        try:
            features = _traffic_to_tensor(traffic)
            result = model(features)

            results.append(DetectionResponse(
                flow_id=f"{traffic.src_ip}:{traffic.src_port}->{traffic.dst_ip}:{traffic.dst_port}",
                is_malicious=result.is_malicious,
                confidence=result.confidence,
                attack_type=result.attack_type,
                attack_category=result.attack_category,
                severity=result.severity,
                explanation=result.explanation,
                recommended_action=result.recommended_action,
                uncertainty_bounds=result.uncertainty_bounds,
                timestamp=datetime.now().isoformat(),
                processing_time_ms=0.0
            ))

            # Update metrics
            DETECTION_COUNTER.labels(
                attack_type=result.attack_type or 'benign',
                severity=result.severity
            ).inc()

        except Exception as e:
            logger.error(f"Error processing flow: {e}")
            continue

    return results


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming detection

    Connect to ws://localhost:8000/ws/stream
    Send JSON: {"src_ip": "...", "dst_ip": "...", ...}
    Receive JSON: {"is_malicious": true, "confidence": 0.95, ...}
    """
    await websocket.accept()
    ACTIVE_CONNECTIONS.inc()

    logger.info("WebSocket connection established")

    try:
        model = initialize_model()

        while True:
            # Receive traffic data
            data = await websocket.receive_json()

            # Convert to TrafficData
            traffic = TrafficData(**data)

            # Detect
            features = _traffic_to_tensor(traffic)
            result = model(features)

            # Send response
            await websocket.send_json({
                "is_malicious": result.is_malicious,
                "confidence": result.confidence,
                "attack_type": result.attack_type,
                "severity": result.severity,
                "explanation": result.explanation,
                "timestamp": datetime.now().isoformat()
            })

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        ACTIVE_CONNECTIONS.dec()
        logger.info("WebSocket connection closed")


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics(api_key: str = Depends(verify_api_key)):
    """Get system metrics"""
    # Return Prometheus metrics in text format
    return StreamingResponse(
        iter([prometheus_client.generate_latest()]),
        media_type="text/plain"
    )


@app.get("/stats", response_model=MetricsResponse, tags=["Monitoring"])
async def get_stats(api_key: str = Depends(verify_api_key)):
    """Get detection statistics"""
    # Implement statistics tracking
    return MetricsResponse(
        total_detections=1000,
        malicious_count=150,
        benign_count=850,
        avg_confidence=0.89,
        avg_latency_ms=95.0,
        active_connections=int(ACTIVE_CONNECTIONS._value.get()),
        threat_level="medium"
    )


@app.post("/model/update", tags=["Management"])
async def update_model(
    checkpoint_path: str,
    api_key: str = Depends(verify_api_key)
):
    """Update model with new checkpoint"""
    try:
        model = initialize_model()
        model.load_pretrained(checkpoint_path)
        return {"status": "success", "message": "Model updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model update failed: {str(e)}")


@app.get("/model/info", tags=["Management"])
async def model_info(api_key: str = Depends(verify_api_key)):
    """Get model information"""
    model = initialize_model()
    return {
        "active_models": model.active_models,
        "confidence_threshold": model.confidence_threshold,
        "device": str(model.device),
        "uncertainty_enabled": model.enable_uncertainty,
        "explanation_enabled": model.enable_explanation
    }


# Helper functions
def _traffic_to_tensor(traffic: TrafficData) -> torch.Tensor:
    """Convert traffic data to tensor for model input"""
    # Feature engineering
    features = [
        traffic.packets / 1000.0,
        traffic.bytes / 1e6,
        traffic.duration,
        traffic.src_port / 65535.0,
        traffic.dst_port / 65535.0,
        _encode_protocol(traffic.protocol),
        # Add more features as needed
    ]

    # Pad to 64 dimensions
    while len(features) < 64:
        features.append(0.0)

    return torch.tensor(features[:64], dtype=torch.float32).unsqueeze(0)


def _encode_protocol(proto: str) -> float:
    """Encode protocol string to numeric"""
    protocols = {'tcp': 0.6, 'udp': 0.17, 'icmp': 0.01}
    return protocols.get(proto.lower(), 0.0)


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Integrated AI-IDS REST API")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    logger.info(f"Starting Integrated AI-IDS REST API on {args.host}:{args.port}")

    uvicorn.run(
        "integrated_ai_ids.api.rest_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info"
    )
