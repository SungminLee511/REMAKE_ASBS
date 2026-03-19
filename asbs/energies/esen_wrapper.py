"""
eSEN energy model wrapper for conformer generation.

Wraps the eSEN foundation model from HuggingFace for computing
DFT-accuracy molecular energies.

Requires: fairchem library, HuggingFace access

NOTE: This is a stub for Phase 8 (amortized conformer generation).
"""

import torch
from .base import EnergyFunction


class ESENEnergy(EnergyFunction):
    """
    Wraps the eSEN model for molecular energy evaluation.

    Input: atomic positions [n_atoms, 3] + atomic numbers + bond info
    Output: energy scalar

    TODO: Implement when ready for Phase 8.
    """

    def __init__(self, model_name: str = "esen-base", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._dim = None  # Variable per molecule
        self._initialized = False

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise ValueError("ESENEnergy dim depends on the molecule.")
        return self._dim

    def _lazy_init(self):
        if self._initialized:
            return
        try:
            # TODO: Load eSEN from HuggingFace
            # from fairchem import ...
            self._initialized = True
        except ImportError:
            raise ImportError("eSEN requires: pip install fairchem")

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        self._lazy_init()
        raise NotImplementedError("eSEN energy not yet implemented. See Phase 8.")
