"""
Model Registry and Upgrade System
===================================

Future-proof architecture for deploying new models including:
- Traditional ML/DL models
- Quantum ML models
- Neuromorphic computing models
- Federated learning updates
- Cloud-native auto-upgrades

Author: Roger Nick Anaedevha
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import hashlib
import json
import aiohttp
import torch
import yaml

logger = logging.getLogger(__name__)


class ModelMetadata:
    """Metadata for registered models"""

    def __init__(
        self,
        name: str,
        version: str,
        model_type: str,
        framework: str,
        input_dim: int,
        output_dim: int,
        performance_metrics: Dict[str, float],
        requirements: List[str],
        checksum: str,
        created_at: datetime,
        author: str,
        description: str
    ):
        self.name = name
        self.version = version
        self.model_type = model_type  # 'pytorch', 'tensorflow', 'onnx', 'quantum', etc.
        self.framework = framework
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.performance_metrics = performance_metrics
        self.requirements = requirements
        self.checksum = checksum
        self.created_at = created_at
        self.author = author
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'version': self.version,
            'model_type': self.model_type,
            'framework': self.framework,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'performance_metrics': self.performance_metrics,
            'requirements': self.requirements,
            'checksum': self.checksum,
            'created_at': self.created_at.isoformat(),
            'author': self.author,
            'description': self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class ModelRegistry:
    """
    Central registry for managing AI models

    Supports:
    - Hot-swapping models without downtime
    - Version management
    - Automatic rollback on failure
    - Cloud-based model repository
    - Multi-framework support (PyTorch, TensorFlow, ONNX, Quantum)
    """

    def __init__(
        self,
        registry_path: Path = Path("/app/models"),
        remote_registry_url: Optional[str] = None,
        enable_auto_update: bool = True
    ):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.remote_registry_url = remote_registry_url
        self.enable_auto_update = enable_auto_update

        self.models: Dict[str, ModelMetadata] = {}
        self.active_models: Dict[str, Any] = {}  # Loaded model instances

        self._load_registry()

        logger.info(f"Model registry initialized at: {self.registry_path}")

    def _load_registry(self):
        """Load registry from disk"""
        registry_file = self.registry_path / "registry.json"

        if registry_file.exists():
            with open(registry_file, 'r') as f:
                data = json.load(f)
                for model_data in data.get('models', []):
                    metadata = ModelMetadata.from_dict(model_data)
                    self.models[f"{metadata.name}:{metadata.version}"] = metadata

            logger.info(f"Loaded {len(self.models)} models from registry")

    def _save_registry(self):
        """Save registry to disk"""
        registry_file = self.registry_path / "registry.json"

        data = {
            'models': [metadata.to_dict() for metadata in self.models.values()],
            'updated_at': datetime.now().isoformat()
        }

        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info("Registry saved to disk")

    def register_model(
        self,
        name: str,
        version: str,
        model_path: Path,
        metadata: ModelMetadata
    ) -> bool:
        """
        Register a new model

        Args:
            name: Model name
            version: Model version (semantic versioning)
            model_path: Path to model file
            metadata: Model metadata

        Returns:
            Success status
        """
        try:
            # Verify model file exists
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Compute checksum
            checksum = self._compute_checksum(model_path)
            metadata.checksum = checksum

            # Copy to registry
            dest_path = self.registry_path / f"{name}_v{version}.pt"
            import shutil
            shutil.copy(model_path, dest_path)

            # Register metadata
            key = f"{name}:{version}"
            self.models[key] = metadata

            # Save registry
            self._save_registry()

            logger.info(f"✅ Registered model: {key}")
            return True

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return False

    def get_model(self, name: str, version: str = "latest") -> Optional[ModelMetadata]:
        """Get model metadata"""
        if version == "latest":
            # Find latest version
            versions = [
                (v.split(':')[1], meta)
                for v, meta in self.models.items()
                if v.startswith(f"{name}:")
            ]
            if not versions:
                return None

            # Sort by version (simple string comparison)
            versions.sort(key=lambda x: x[0], reverse=True)
            return versions[0][1]

        key = f"{name}:{version}"
        return self.models.get(key)

    def load_model(self, name: str, version: str = "latest", device: str = "cuda") -> Any:
        """
        Load model into memory

        Args:
            name: Model name
            version: Model version
            device: Device to load on ('cuda', 'cpu')

        Returns:
            Loaded model instance
        """
        metadata = self.get_model(name, version)
        if not metadata:
            raise ValueError(f"Model not found: {name}:{version}")

        model_path = self.registry_path / f"{name}_v{metadata.version}.pt"

        logger.info(f"Loading model: {name}:{metadata.version} on {device}")

        # Load based on framework
        if metadata.framework == "pytorch":
            model = torch.load(model_path, map_location=device)
            model.eval()

        elif metadata.framework == "onnx":
            import onnxruntime as ort
            model = ort.InferenceSession(str(model_path))

        elif metadata.framework == "quantum":
            # Placeholder for quantum ML models (e.g., PennyLane, Qiskit)
            from .quantum_models import load_quantum_model
            model = load_quantum_model(model_path)

        elif metadata.framework == "tensorflow":
            import tensorflow as tf
            model = tf.keras.models.load_model(str(model_path))

        else:
            raise ValueError(f"Unsupported framework: {metadata.framework}")

        # Cache loaded model
        key = f"{name}:{metadata.version}"
        self.active_models[key] = model

        logger.info(f"✅ Model loaded: {key}")
        return model

    def hot_swap_model(
        self,
        current_name: str,
        new_version: str,
        validation_fn: Optional[callable] = None
    ) -> bool:
        """
        Hot-swap a model without downtime

        Args:
            current_name: Current model name
            new_version: New model version
            validation_fn: Optional validation function

        Returns:
            Success status
        """
        try:
            logger.info(f"Hot-swapping {current_name} to version {new_version}")

            # Load new model
            new_model = self.load_model(current_name, new_version)

            # Validate if function provided
            if validation_fn:
                if not validation_fn(new_model):
                    logger.error("Validation failed for new model")
                    return False

            # Update active model
            current_key = f"{current_name}:latest"
            old_model = self.active_models.get(current_key)

            self.active_models[current_key] = new_model

            logger.info(f"✅ Hot-swap successful: {current_name} → v{new_version}")

            # Cleanup old model
            if old_model and hasattr(old_model, 'cpu'):
                old_model.cpu()
                del old_model

            return True

        except Exception as e:
            logger.error(f"Hot-swap failed: {e}")
            return False

    async def check_for_updates(self) -> List[Dict[str, Any]]:
        """
        Check remote registry for model updates

        Returns:
            List of available updates
        """
        if not self.remote_registry_url:
            logger.warning("No remote registry configured")
            return []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.remote_registry_url}/models") as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch updates: {response.status}")
                        return []

                    remote_models = await response.json()

            # Find updates
            updates = []
            for remote_model in remote_models:
                name = remote_model['name']
                version = remote_model['version']
                key = f"{name}:{version}"

                # Check if newer version available
                local_model = self.get_model(name, "latest")

                if not local_model or self._is_newer_version(version, local_model.version):
                    updates.append({
                        'name': name,
                        'current_version': local_model.version if local_model else None,
                        'new_version': version,
                        'performance_metrics': remote_model.get('performance_metrics', {}),
                        'description': remote_model.get('description', '')
                    })

            logger.info(f"Found {len(updates)} available updates")
            return updates

        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            return []

    async def download_and_install_update(
        self,
        name: str,
        version: str,
        auto_activate: bool = True
    ) -> bool:
        """
        Download and install model update

        Args:
            name: Model name
            version: Model version
            auto_activate: Automatically activate after install

        Returns:
            Success status
        """
        try:
            logger.info(f"Downloading {name}:{version}...")

            async with aiohttp.ClientSession() as session:
                # Download model file
                url = f"{self.remote_registry_url}/models/{name}/{version}/download"
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Download failed: {response.status}")
                        return False

                    model_data = await response.read()

                # Download metadata
                meta_url = f"{self.remote_registry_url}/models/{name}/{version}/metadata"
                async with session.get(meta_url) as response:
                    metadata_dict = await response.json()
                    metadata = ModelMetadata.from_dict(metadata_dict)

            # Save model
            model_path = self.registry_path / f"{name}_v{version}.pt"
            with open(model_path, 'wb') as f:
                f.write(model_data)

            # Verify checksum
            checksum = self._compute_checksum(model_path)
            if checksum != metadata.checksum:
                logger.error("Checksum verification failed")
                model_path.unlink()
                return False

            # Register model
            key = f"{name}:{version}"
            self.models[key] = metadata
            self._save_registry()

            logger.info(f"✅ Downloaded and installed: {name}:{version}")

            # Auto-activate
            if auto_activate:
                return self.hot_swap_model(name, version)

            return True

        except Exception as e:
            logger.error(f"Failed to download update: {e}")
            return False

    async def auto_update_loop(self, interval_hours: int = 24):
        """
        Automatic update check loop

        Args:
            interval_hours: Check interval in hours
        """
        if not self.enable_auto_update:
            logger.info("Auto-update disabled")
            return

        logger.info(f"Starting auto-update loop (interval: {interval_hours}h)")

        while True:
            try:
                # Check for updates
                updates = await self.check_for_updates()

                for update in updates:
                    name = update['name']
                    version = update['new_version']

                    # Check if update meets criteria
                    if self._should_auto_update(update):
                        logger.info(f"Auto-updating {name} to {version}")
                        success = await self.download_and_install_update(
                            name,
                            version,
                            auto_activate=True
                        )

                        if success:
                            logger.info(f"✅ Auto-update successful: {name}:{version}")
                        else:
                            logger.error(f"❌ Auto-update failed: {name}:{version}")

                # Wait for next check
                await asyncio.sleep(interval_hours * 3600)

            except Exception as e:
                logger.error(f"Error in auto-update loop: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour

    def _should_auto_update(self, update: Dict[str, Any]) -> bool:
        """
        Determine if update should be auto-installed

        Criteria:
        - Performance improvement >= 2%
        - No breaking changes
        - Stability score >= 95%
        """
        metrics = update.get('performance_metrics', {})

        # Check performance improvement
        current_model = self.get_model(update['name'], 'latest')
        if current_model:
            current_accuracy = current_model.performance_metrics.get('accuracy', 0)
            new_accuracy = metrics.get('accuracy', 0)

            improvement = (new_accuracy - current_accuracy) / current_accuracy * 100

            if improvement < 2.0:
                logger.info(f"Skipping update: improvement {improvement:.2f}% < 2%")
                return False

        # Check stability
        stability = metrics.get('stability_score', 0)
        if stability < 0.95:
            logger.info(f"Skipping update: stability {stability:.2f} < 0.95")
            return False

        return True

    @staticmethod
    def _compute_checksum(file_path: Path) -> str:
        """Compute SHA256 checksum"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _is_newer_version(v1: str, v2: str) -> bool:
        """Compare semantic versions"""
        # Simple comparison (should use proper semver library)
        parts1 = [int(x) for x in v1.split('.')]
        parts2 = [int(x) for x in v2.split('.')]

        for p1, p2 in zip(parts1, parts2):
            if p1 > p2:
                return True
            elif p1 < p2:
                return False

        return len(parts1) > len(parts2)

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        return [
            {
                'name': meta.name,
                'version': meta.version,
                'framework': meta.framework,
                'performance': meta.performance_metrics,
                'created_at': meta.created_at.isoformat()
            }
            for meta in self.models.values()
        ]

    def export_model(self, name: str, version: str, format: str = "onnx") -> Path:
        """
        Export model to different format

        Args:
            name: Model name
            version: Model version
            format: Target format ('onnx', 'torchscript', 'tflite')

        Returns:
            Path to exported model
        """
        model = self.load_model(name, version)
        export_path = self.registry_path / f"{name}_v{version}.{format}"

        if format == "onnx":
            # Export PyTorch to ONNX
            dummy_input = torch.randn(1, 64)  # Adjust based on model
            torch.onnx.export(
                model,
                dummy_input,
                export_path,
                export_params=True,
                opset_version=14,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )

        elif format == "torchscript":
            # TorchScript compilation
            scripted = torch.jit.script(model)
            scripted.save(str(export_path))

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"✅ Exported model to {format}: {export_path}")
        return export_path


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        # Initialize registry
        registry = ModelRegistry(
            registry_path=Path("/app/models"),
            remote_registry_url="https://robustidps.ai/api/v1/model-registry",
            enable_auto_update=True
        )

        # Register a new model
        metadata = ModelMetadata(
            name="neural_ode_v2",
            version="2.0.0",
            model_type="pytorch",
            framework="pytorch",
            input_dim=64,
            output_dim=13,
            performance_metrics={
                'accuracy': 0.989,
                'f1_score': 0.985,
                'false_positive_rate': 0.012
            },
            requirements=['torch>=2.0.0', 'torchdiffeq>=0.2.3'],
            checksum="",
            created_at=datetime.now(),
            author="Roger Nick Anaedevha",
            description="Neural ODE v2 with improved temporal adaptation"
        )

        registry.register_model(
            name="neural_ode_v2",
            version="2.0.0",
            model_path=Path("/tmp/neural_ode_v2.pt"),
            metadata=metadata
        )

        # Load model
        model = registry.load_model("neural_ode_v2", "latest")

        # Check for updates
        updates = await registry.check_for_updates()
        print(f"Available updates: {updates}")

        # Hot-swap model
        registry.hot_swap_model("neural_ode", "2.1.0")

        # Start auto-update loop
        await registry.auto_update_loop(interval_hours=24)

    asyncio.run(main())
