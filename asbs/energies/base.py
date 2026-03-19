"""Abstract base class for energy functions."""

import torch
from abc import ABC, abstractmethod


class EnergyFunction(ABC):
    """Base class for energy functions E(x)."""

    @abstractmethod
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute E(x). Input [batch, dim], output [batch]."""

    def grad_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ∇E(x) via autograd. Input/output [batch, dim]."""
        x = x.detach().requires_grad_(True)
        e = self.energy(x)
        grad = torch.autograd.grad(e.sum(), x, create_graph=False)[0]
        return grad.detach()

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the sample space."""
