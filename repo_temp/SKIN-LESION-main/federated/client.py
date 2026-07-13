"""
Federated Learning Client — Local Training Node (Scaffold / Stub).

Each FL client represents a hospital or dermatology clinic that trains
the shared global model on its private, locally-stored dermoscopic dataset.

Privacy Model:
    - Raw patient images NEVER leave the local node.
    - Only model weight deltas (ΔW) are transmitted to the server.
    - Differential Privacy (DP) noise injection point is annotated below.

Local Training:
    Each client runs E local epochs of SGD on its private data before
    transmitting the updated weights. Larger E reduces communication
    overhead but may cause client drift (divergence from global optimum).

NOTE: Production implementation requires:
    - Secure aggregation protocol (e.g., SecAgg)
    - Differential Privacy with (ε, δ)-DP guarantees
    - TLS-encrypted weight transmission
    - Authenticated client registration
"""

from typing import Dict, Optional, Tuple
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FederatedClient:
    """
    Local FL training node simulating a hospital dermoscopy department.

    Args:
        client_id: Unique identifier for this client node.
        local_loader: DataLoader for the client's private local dataset.
        device: Training device.
        local_epochs: Number of local SGD epochs per round (default 5).
        lr: Local learning rate.
        dp_noise_multiplier: σ for DP Gaussian noise injection.
                             Set to 0.0 to disable DP (default).
        dp_max_grad_norm: Clipping norm C for DP gradient clipping.
    """

    def __init__(
        self,
        client_id: int,
        local_loader: DataLoader,
        device: torch.device,
        local_epochs: int = 5,
        lr: float = 1e-4,
        dp_noise_multiplier: float = 0.0,
        dp_max_grad_norm: float = 1.0,
    ):
        self.client_id = client_id
        self.local_loader = local_loader
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.dp_noise_multiplier = dp_noise_multiplier
        self.dp_max_grad_norm = dp_max_grad_norm
        self._n_samples = len(local_loader.dataset)

    def local_train(
        self,
        global_weights: Dict[str, torch.Tensor],
        model_template: Optional[nn.Module] = None,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Train the global model on local private data for E epochs.

        Args:
            global_weights: State dict from the server's global model.
            model_template: An uninitialized model instance (architecture only).
                            If None, a stub dict update is returned.

        Returns:
            updated_weights: State dict after local training.
            n_samples: Number of local training samples (for FedAvg weighting).
        """
        if model_template is None:
            # Scaffold: return global weights unchanged (no actual training)
            print(f"  [Client {self.client_id}] Stub: returning global weights unchanged.")
            return copy.deepcopy(global_weights), self._n_samples

        # Load global weights
        local_model = copy.deepcopy(model_template)
        local_model.load_state_dict(global_weights)
        local_model = local_model.to(self.device)
        local_model.train()

        optimizer = torch.optim.AdamW(local_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.local_epochs):
            for batch in self.local_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits, *_ = local_model(images)
                loss = criterion(logits, labels)
                loss.backward()

                # ── DP Gradient Clipping + Noise Injection ──────────────────
                if self.dp_noise_multiplier > 0:
                    # Clip gradients to bound sensitivity
                    torch.nn.utils.clip_grad_norm_(
                        local_model.parameters(), self.dp_max_grad_norm
                    )
                    # Inject calibrated Gaussian noise (DP mechanism)
                    with torch.no_grad():
                        for param in local_model.parameters():
                            if param.grad is not None:
                                noise = torch.randn_like(param.grad)
                                noise *= self.dp_noise_multiplier * self.dp_max_grad_norm
                                param.grad += noise
                # ────────────────────────────────────────────────────────────

                optimizer.step()

        updated_weights = local_model.state_dict()
        print(f"  [Client {self.client_id}] Local training done | n={self._n_samples}")
        return updated_weights, self._n_samples
