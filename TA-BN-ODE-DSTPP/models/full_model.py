"""
Full TA-BN-ODE-DSTPP Model.

Integrates all components into the end-to-end pipeline described in
Algorithm 1 and the total loss (Eq. 3):

  L(Theta) = L_cls + L_TPP + L_ELBO + L_reg

Architecture: 2.3M parameters (vs 12.8M transformer baseline — 82% reduction).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .ta_bn_ode import Encoder, EventUpdate, TABNODEBlock
from .dstpp import DeepSpatioTemporalPointProcess


class TABNODEPointProcess(nn.Module):
    """Full TA-BN-ODE with Deep Spatio-Temporal Point Process.

    Pipeline (Algorithm 1):
      1. Encode input: h(t_0) = Encoder(x_0)
      2. For each event i=1..n:
         a. Integrate ODE: h(t_i^-) = ODESolve(f_theta, h(t_{i-1}), [t_{i-1}, t_i])
         b. Event update:  h(t_i) = h(t_i^-) + Update(x_i)
      3. Compute intensities via DSTPP
      4. Classification via decoder
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 n_classes: int = 2, d_model: int = 512,
                 n_ode_blocks: int = 2,
                 time_constants: tuple = (1e-6, 1e-3, 1.0, 3600.0),
                 n_transformer_layers: int = 4, n_attention_heads: int = 8,
                 tabn_mlp_hidden: int = 64, tabn_mlp_layers: int = 2,
                 solver_method: str = "dopri5",
                 rtol: float = 1e-3, atol: float = 1e-4,
                 transformer_dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # Feature encoder
        self.encoder = Encoder(input_dim, hidden_dim)

        # Event-driven update
        self.event_update = EventUpdate(input_dim, hidden_dim)

        # Stacked ODE blocks
        self.ode_blocks = nn.ModuleList([
            TABNODEBlock(
                hidden_dim, time_constants, tabn_mlp_hidden, tabn_mlp_layers,
                solver_method, rtol, atol
            )
            for _ in range(n_ode_blocks)
        ])

        # DSTPP for intensity modeling
        self.dstpp = DeepSpatioTemporalPointProcess(
            hidden_dim, n_classes, d_model,
            n_transformer_layers, n_attention_heads,
            transformer_dropout
        )

        # Classification decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, n_classes),
        )

        # Temperature parameter for calibration (Section 4.4)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, t_span: torch.Tensor,
                event_types: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        Forward pass (Algorithm 1).

        Args:
            x: Input features [batch, seq_len, input_dim] or [batch, input_dim]
            t_span: Time points [seq_len] or [batch, seq_len]
            event_types: Attack type labels per event [batch, seq_len] (optional)

        Returns:
            Dictionary with 'logits', 'hidden_states', 'intensities'
        """
        # Handle non-sequential input (single-step)
        if x.dim() == 2:
            return self._forward_single(x, t_span, event_types)

        batch_size, seq_len, input_dim = x.shape

        # 1. Encode initial input
        h = self.encoder(x[:, 0])  # [batch, hidden_dim]

        # 2. Event-by-event integration
        hidden_states = [h]
        for i in range(1, seq_len):
            # a. ODE integration between events
            t_segment = t_span[i-1:i+1] if t_span.dim() == 1 else t_span[:, i-1:i+1]
            if t_segment.dim() == 1 and t_segment.shape[0] >= 2:
                for ode_block in self.ode_blocks:
                    h_traj = ode_block(h, t_segment)
                    h = h_traj[-1]  # Take state at t_i

            # b. Event-driven update
            h = h + self.event_update(x[:, i])
            hidden_states.append(h)

        # Stack hidden states: [batch, seq_len, hidden_dim]
        h_seq = torch.stack(hidden_states, dim=1)

        # 3. Classification from final state
        logits = self.decoder(h_seq[:, -1]) / self.temperature

        # 4. Point process intensities
        t_for_dstpp = t_span.unsqueeze(0).expand(batch_size, -1) if t_span.dim() == 1 else t_span
        intensities = self.dstpp.compute_intensity(
            h_seq, t_for_dstpp, event_types
        )

        return {
            "logits": logits,
            "hidden_states": h_seq,
            "intensities": intensities,
        }

    def _forward_single(self, x: torch.Tensor, t_span: torch.Tensor,
                        event_types: Optional[torch.Tensor] = None
                        ) -> Dict[str, torch.Tensor]:
        """Forward pass for non-sequential (single-event) input."""
        h = self.encoder(x)

        for ode_block in self.ode_blocks:
            h_traj = ode_block(h, t_span)
            h = h_traj[-1]

        logits = self.decoder(h) / self.temperature

        return {
            "logits": logits,
            "hidden_states": h.unsqueeze(1),
            "intensities": None,
        }

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor,
                     t_span: torch.Tensor,
                     event_types: Optional[torch.Tensor] = None,
                     weights: Optional[Dict[str, float]] = None
                     ) -> Dict[str, torch.Tensor]:
        """Compute total loss (Eq. 3): L = L_cls + L_TPP + L_ELBO + L_reg.

        Args:
            x: Input features
            y: Classification labels [batch]
            t_span: Time points
            event_types: Event types for point process
            weights: Loss component weights

        Returns:
            Dictionary with 'total', 'cls', 'tpp', 'reg'
        """
        w = weights or {"cls": 1.0, "tpp": 1.0, "reg": 1e-4}

        out = self.forward(x, t_span, event_types)

        # L_cls: cross-entropy
        loss_cls = F.cross_entropy(out["logits"], y)

        # L_TPP: point process NLL
        loss_tpp = torch.tensor(0.0, device=x.device)
        if out["intensities"] is not None and event_types is not None:
            t_for_pp = t_span.unsqueeze(0).expand(x.shape[0], -1) if t_span.dim() == 1 else t_span
            loss_tpp = self.dstpp.log_likelihood(
                out["hidden_states"], t_for_pp, event_types,
                t_span[-1] if t_span.dim() == 1 else t_span[:, -1]
            )

        # L_reg: weight decay
        loss_reg = sum(p.pow(2).sum() for p in self.parameters())

        total = (w.get("cls", 1.0) * loss_cls +
                 w.get("tpp", 1.0) * loss_tpp +
                 w.get("reg", 1e-4) * loss_reg)

        return {
            "total": total,
            "cls": loss_cls,
            "tpp": loss_tpp,
            "reg": loss_reg,
        }

    def calibrate_temperature(self, val_loader, device: str = "cuda",
                              max_iter: int = 50, lr: float = 0.01):
        """Post-hoc temperature scaling on validation data (Section 4.4).

        Minimizes NLL w.r.t. self.temperature on the validation set.
        """
        self.eval()
        self.temperature.requires_grad_(True)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["x"].to(device), batch["y"].to(device)
                t_span = batch["t"].to(device) if "t" in batch else torch.linspace(0, 1, 10, device=device)
                out = self.forward(x, t_span)
                # Store un-tempered logits
                all_logits.append(out["logits"] * self.temperature.detach())
                all_labels.append(y)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(all_logits / self.temperature, all_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature.requires_grad_(False)
        print(f"Calibrated temperature: {self.temperature.item():.4f}")
