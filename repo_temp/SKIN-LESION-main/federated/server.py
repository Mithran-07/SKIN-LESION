"""
Federated Learning Server — FedAvg Aggregation (Scaffold / Stub).

Federated Learning (FL) resolves the data silo problem by keeping raw
patient dermoscopic images on local hospital servers while sharing only
model weight updates (gradients) with a central aggregation server.

FedAvg Algorithm (McMahan et al., 2017):
    1. Initialize global model weights W_0 on server.
    2. For each round r:
        a. Broadcast W_r to all K clients.
        b. Each client k trains locally: W_k^{r+1} = LocalTrain(W_r, D_k)
        c. Server aggregates: W_{r+1} = Σ_k (n_k / n) * W_k^{r+1}
           where n_k = local dataset size, n = total samples.
    3. Redistribute W_{r+1} to all clients.

Privacy Extension:
    Differential Privacy (DP) can be applied by injecting Gaussian noise
    into the gradients before transmission:
        W_k^{noisy} = W_k + N(0, σ²) * clip(W_k, C)
    where C is the gradient clipping norm and σ is the noise multiplier.

NOTE: This is a SCAFFOLD implementation demonstrating the architecture.
      Production deployment requires a secure network stack (e.g., PySyft,
      FATE, or Flower framework) and rigorous DP auditing.
"""

from typing import List, Dict, Optional
import copy
import numpy as np
import torch
import torch.nn as nn


class FederatedServer:
    """
    Central FL server responsible for model initialization and FedAvg aggregation.

    Args:
        global_model: The initial global model instance.
        num_clients: Number of participating client nodes.
        aggregation: Aggregation strategy ('fedavg' supported).
    """

    def __init__(
        self,
        global_model: nn.Module,
        num_clients: int = 5,
        aggregation: str = "fedavg",
    ):
        self.global_model = global_model
        self.num_clients = num_clients
        self.aggregation = aggregation
        self.round_history: List[Dict] = []

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Return current global model state dict for distribution to clients."""
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(
        self,
        client_updates: List[Dict],
        client_sizes: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        FedAvg aggregation: weighted average of client weight updates.

        Args:
            client_updates: List of state_dicts from each participating client.
            client_sizes: List of local dataset sizes for weighted averaging.
                          If None, uniform weights are used.

        Returns:
            New aggregated global state dict.
        """
        assert len(client_updates) > 0, "No client updates to aggregate."

        if client_sizes is None:
            client_sizes = [1] * len(client_updates)
        total = sum(client_sizes)

        # Weighted average of all parameter tensors
        aggregated = copy.deepcopy(client_updates[0])
        for key in aggregated.keys():
            aggregated[key] = torch.zeros_like(aggregated[key], dtype=torch.float32)
            for client_sd, n_k in zip(client_updates, client_sizes):
                weight = n_k / total
                aggregated[key] += weight * client_sd[key].float()

        # Load aggregated weights into global model
        self.global_model.load_state_dict(aggregated)

        round_log = {
            "num_clients": len(client_updates),
            "client_sizes": client_sizes,
        }
        self.round_history.append(round_log)

        return aggregated

    def run_rounds(
        self,
        clients: list,
        num_rounds: int = 20,
    ) -> nn.Module:
        """
        Orchestrate the full FL training loop across multiple rounds.

        NOTE: In this scaffold, clients are instantiated in-memory.
              In production, this would involve network RPC calls.

        Args:
            clients: List of FederatedClient instances.
            num_rounds: Number of federated rounds.

        Returns:
            Final global model after all rounds.
        """
        print(f"[FedServer] Starting FL: {num_rounds} rounds, {len(clients)} clients")

        for r in range(num_rounds):
            global_weights = self.get_global_weights()
            client_updates = []
            client_sizes = []

            for i, client in enumerate(clients):
                # Distribute global weights to each client
                local_weights, local_n = client.local_train(global_weights)
                client_updates.append(local_weights)
                client_sizes.append(local_n)
                print(f"  [Round {r+1}] Client {i+1}/{len(clients)} trained on {local_n} samples")

            # Aggregate
            self.aggregate(client_updates, client_sizes)
            print(f"  [Round {r+1}] Aggregation complete")

        print(f"[FedServer] FL complete.")
        return self.global_model
