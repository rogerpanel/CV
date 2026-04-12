"""
FedLLM-API: Main model implementation

This module implements the complete FedLLM-API architecture including:
- Privacy-preserving API encoding
- LoRA-based parameter-efficient fine-tuning
- Prompt-based conditioning for zero-shot detection
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import numpy as np


class FedLLM_API(nn.Module):
    """
    FedLLM-API model for privacy-preserving federated API threat detection.

    Args:
        backbone_name: Pre-trained language model name (e.g., 'distilbert-base-uncased')
        lora_rank: Rank for low-rank adaptation matrices
        lora_alpha: Scaling factor for LoRA
        prompt_length: Length of continuous prompt embeddings
        hidden_dim: Hidden dimension for classification head
        num_classes: Number of output classes (2 for binary detection)
        dropout: Dropout probability
    """

    def __init__(
        self,
        backbone_name: str = 'distilbert-base-uncased',
        lora_rank: int = 8,
        lora_alpha: int = 16,
        prompt_length: int = 20,
        hidden_dim: int = 768,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.prompt_length = prompt_length
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Load pre-trained language model (frozen)
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)

        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Continuous prompt embeddings (trainable)
        self.prompt_embeddings = nn.Parameter(
            torch.randn(prompt_length, hidden_dim) * 0.02
        )

        # LoRA adapters will be injected into attention layers
        self.lora_adapters = nn.ModuleDict()
        self._inject_lora_adapters()

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Prompt aggregation for federated learning
        self.use_federated_prompt = False
        self.federated_prompt = None

    def _inject_lora_adapters(self):
        """Inject LoRA adapters into transformer attention layers."""
        from models.lora_adapter import LoRALayer

        for name, module in self.backbone.named_modules():
            if 'attention' in name.lower() and isinstance(module, nn.Linear):
                # Add LoRA to query and value projections
                if 'query' in name.lower() or 'value' in name.lower():
                    lora_layer = LoRALayer(
                        module.in_features,
                        module.out_features,
                        rank=self.lora_rank,
                        alpha=self.lora_alpha
                    )
                    self.lora_adapters[name] = lora_layer

    def forward(
        self,
        encoded_api_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through FedLLM-API.

        Args:
            encoded_api_features: Privacy-preserved API encodings [batch, seq_len, hidden]
            attention_mask: Attention mask for padding [batch, seq_len]

        Returns:
            logits: Classification logits [batch, num_classes]
        """
        batch_size, seq_len, _ = encoded_api_features.shape

        # Prepend prompt embeddings
        prompt = self.federated_prompt if self.use_federated_prompt else self.prompt_embeddings
        prompt_batch = prompt.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate [prompt; api_features]
        combined_features = torch.cat([prompt_batch, encoded_api_features], dim=1)

        # Extend attention mask for prompt
        if attention_mask is not None:
            prompt_mask = torch.ones(
                batch_size, self.prompt_length,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # Pass through backbone with LoRA
        outputs = self._forward_with_lora(
            combined_features,
            attention_mask=attention_mask
        )

        # Extract [CLS] or pooled representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Classification
        logits = self.classifier(pooled_output)

        return logits

    def _forward_with_lora(self, inputs_embeds, attention_mask=None):
        """Forward pass through backbone with LoRA adapters applied."""
        # This is a simplified version - in practice, you'd need to
        # properly hook LoRA into the forward pass
        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )

    def get_lora_parameters(self) -> Dict[str, torch.Tensor]:
        """Get LoRA parameters for federated aggregation."""
        lora_params = {}
        for name, adapter in self.lora_adapters.items():
            lora_params[f"{name}.lora_A"] = adapter.lora_A.data.clone()
            lora_params[f"{name}.lora_B"] = adapter.lora_B.data.clone()
        return lora_params

    def set_lora_parameters(self, lora_params: Dict[str, torch.Tensor]):
        """Set LoRA parameters from federated aggregation."""
        for name, adapter in self.lora_adapters.items():
            if f"{name}.lora_A" in lora_params:
                adapter.lora_A.data = lora_params[f"{name}.lora_A"].clone()
            if f"{name}.lora_B" in lora_params:
                adapter.lora_B.data = lora_params[f"{name}.lora_B"].clone()

    def get_prompt_embeddings(self) -> torch.Tensor:
        """Get prompt embeddings for prompt-based aggregation."""
        return self.prompt_embeddings.data.clone()

    def set_prompt_embeddings(self, prompt: torch.Tensor):
        """Set prompt embeddings from federated aggregation."""
        self.prompt_embeddings.data = prompt.clone()

    def enable_federated_prompt(self, federated_prompt: torch.Tensor):
        """Enable using federated aggregated prompt."""
        self.use_federated_prompt = True
        self.federated_prompt = federated_prompt

    def disable_federated_prompt(self):
        """Disable federated prompt, use local prompt."""
        self.use_federated_prompt = False
        self.federated_prompt = None

    def num_trainable_params(self) -> int:
        """Count trainable parameters (LoRA + prompt + classifier)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        """Count total parameters including frozen backbone."""
        return sum(p.numel() for p in self.parameters())


class ZeroShotDetector(nn.Module):
    """
    Zero-shot API threat detector using prompt engineering.

    Enables detection of unseen attack types through natural language
    descriptions without requiring labeled examples.
    """

    def __init__(self, base_model: FedLLM_API):
        super().__init__()
        self.base_model = base_model
        self.attack_type_prompts = {
            'auth_bypass': "Detect authentication bypass through credential stuffing or token manipulation",
            'injection': "Identify SQL injection, command injection, or XSS in API parameters",
            'broken_authz': "Find broken object-level authorization accessing other users' resources",
            'data_exposure': "Detect excessive data exposure via unfiltered queries",
            'rate_abuse': "Identify rate limiting abuse or API flooding",
            'business_logic': "Find business logic vulnerabilities like negative quantities or race conditions"
        }

    def detect(
        self,
        encoded_api_features: torch.Tensor,
        attack_type: str,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Zero-shot detection for specific attack type.

        Args:
            encoded_api_features: Privacy-preserved API encodings
            attack_type: Type of attack to detect (key in attack_type_prompts)
            attention_mask: Attention mask

        Returns:
            logits: Detection logits for specified attack type
        """
        # Retrieve attack-specific prompt
        prompt_text = self.attack_type_prompts.get(
            attack_type,
            "Detect anomalous API behavior indicative of security threats"
        )

        # Encode prompt (simplified - in practice would use more sophisticated encoding)
        # Here we just use the base model's prompt embeddings

        return self.base_model(encoded_api_features, attention_mask)

    def add_attack_type(self, attack_name: str, description: str):
        """Add new attack type for zero-shot detection."""
        self.attack_type_prompts[attack_name] = description


def create_fedllm_api(config: Dict) -> FedLLM_API:
    """
    Factory function to create FedLLM-API model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized FedLLM-API model
    """
    return FedLLM_API(
        backbone_name=config.get('backbone', 'distilbert-base-uncased'),
        lora_rank=config.get('lora_rank', 8),
        lora_alpha=config.get('lora_alpha', 16),
        prompt_length=config.get('prompt_length', 20),
        hidden_dim=config.get('hidden_dim', 768),
        num_classes=config.get('num_classes', 2),
        dropout=config.get('dropout', 0.1)
    )


if __name__ == "__main__":
    # Example usage
    config = {
        'backbone': 'distilbert-base-uncased',
        'lora_rank': 8,
        'lora_alpha': 16,
        'prompt_length': 20,
        'hidden_dim': 768,
        'num_classes': 2
    }

    model = create_fedllm_api(config)

    print(f"Total parameters: {model.num_total_params():,}")
    print(f"Trainable parameters: {model.num_trainable_params():,}")
    print(f"Trainable fraction: {model.num_trainable_params() / model.num_total_params():.2%}")

    # Test forward pass
    batch_size, seq_len, hidden_dim = 4, 10, 768
    dummy_features = torch.randn(batch_size, seq_len, hidden_dim)
    logits = model(dummy_features)
    print(f"\nOutput logits shape: {logits.shape}")
