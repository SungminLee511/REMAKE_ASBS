"""Efficient replay buffer for (x0, x1) endpoint pairs."""

import torch


class ReplayBuffer:
    """
    Stores (X_0, X_1) endpoint pairs from previous model evaluations.
    Fixed max size, FIFO eviction, random sampling.
    Contiguous tensor storage for efficiency.
    """

    def __init__(self, max_size: int, dim: int, device: str = "cpu"):
        self.max_size = max_size
        self.dim = dim
        self.device = device
        self.x0 = torch.zeros(max_size, dim)
        self.x1 = torch.zeros(max_size, dim)
        self.ptr = 0
        self.size = 0

    def add(self, x0: torch.Tensor, x1: torch.Tensor):
        """Add a batch of (x0, x1) pairs."""
        batch_size = x0.shape[0]
        x0_cpu = x0.detach().cpu()
        x1_cpu = x1.detach().cpu()

        for i in range(batch_size):
            self.x0[self.ptr] = x0_cpu[i]
            self.x1[self.ptr] = x1_cpu[i]
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> tuple:
        """Randomly sample a mini-batch."""
        indices = torch.randint(0, self.size, (batch_size,))
        x0 = self.x0[indices].to(self.device)
        x1 = self.x1[indices].to(self.device)
        return x0, x1

    def __len__(self):
        return self.size
