"""
Equivariant Graph Neural Network (EGNN) for particle systems.

Based on Satorras et al. (2021): "E(n) Equivariant Graph Neural Networks"

Key properties:
- SE(3) equivariant: rotation/translation of input → same transformation of output
- Uses only pairwise distances (invariant) in MLPs
- Position updates are equivariant via direction vectors
"""

import torch
import torch.nn as nn
from .time_embedding import SinusoidalTimeEmbedding


class EGNNLayer(nn.Module):
    """Single EGNN message-passing layer."""

    def __init__(self, hidden_dim: int, coord_dim: int = 3):
        super().__init__()

        # Edge MLP: takes (h_i, h_j, ||x_i - x_j||^2)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Coordinate MLP: scalar weight for position update
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        # Small init for stability
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.01)

        # Node MLP: update node features
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, x, edge_index):
        """
        h: [batch * n_particles, hidden_dim] node features
        x: [batch * n_particles, coord_dim] positions
        edge_index: [2, n_edges] source/target indices

        Returns: updated (h, x)
        """
        src, dst = edge_index  # source, destination

        # Relative positions and squared distances
        rel_pos = x[src] - x[dst]  # [n_edges, coord_dim]
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)  # [n_edges, 1]

        # Edge messages
        edge_input = torch.cat([h[src], h[dst], dist_sq], dim=-1)
        m_ij = self.edge_mlp(edge_input)  # [n_edges, hidden_dim]

        # Coordinate update (equivariant)
        coord_weight = self.coord_mlp(m_ij)  # [n_edges, 1]
        coord_agg = torch.zeros_like(x)
        coord_agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(rel_pos), rel_pos * coord_weight)
        x = x + coord_agg

        # Node feature update (invariant)
        msg_agg = torch.zeros_like(h)
        msg_agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(m_ij), m_ij)
        node_input = torch.cat([h, msg_agg], dim=-1)
        h = h + self.node_mlp(node_input)

        return h, x


class EGNN(nn.Module):
    """
    Equivariant Graph Neural Network for particle systems.

    Input: positions [batch, n_particles, coord_dim]
    Output: position updates [batch, n_particles, coord_dim] (equivariant, zero CoM)
    """

    def __init__(
        self,
        n_particles: int,
        coord_dim: int = 3,
        hidden_dim: int = 128,
        n_layers: int = 4,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.n_particles = n_particles
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Initial node features: just time embedding (broadcast to all particles)
        self.node_embed = nn.Linear(time_embed_dim, hidden_dim)

        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, coord_dim) for _ in range(n_layers)
        ])

        # Build fully-connected edge index (precomputed)
        src, dst = [], []
        for i in range(n_particles):
            for j in range(n_particles):
                if i != j:
                    src.append(i)
                    dst.append(j)
        self.register_buffer(
            "_edge_index_single",
            torch.tensor([src, dst], dtype=torch.long),
        )

    def _build_batch_edge_index(self, batch_size: int, device: torch.device):
        """Build edge index for a batch of graphs."""
        edge_index = self._edge_index_single.to(device)
        offsets = torch.arange(batch_size, device=device).unsqueeze(1) * self.n_particles
        # [2, n_edges] → [2, batch * n_edges]
        batch_edges = (edge_index.unsqueeze(0) + offsets.unsqueeze(1)).reshape(2, -1)
        return batch_edges

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        t: [batch] scalar time
        x: [batch, n_particles * coord_dim] flattened positions

        Returns: [batch, n_particles * coord_dim] position updates (zero CoM)
        """
        batch_size = x.shape[0]
        device = x.device

        if t.dim() == 0:
            t = t.expand(batch_size)

        # Reshape to [batch, n_particles, coord_dim]
        pos = x.view(batch_size, self.n_particles, self.coord_dim)

        # Time embedding → node features
        t_emb = self.time_embed(t)  # [batch, time_embed_dim]
        h = self.node_embed(t_emb)  # [batch, hidden_dim]
        # Broadcast to all particles
        h = h.unsqueeze(1).expand(-1, self.n_particles, -1)  # [batch, n_particles, hidden_dim]

        # Flatten batch dimension for message passing
        h = h.reshape(batch_size * self.n_particles, self.hidden_dim)
        pos_flat = pos.reshape(batch_size * self.n_particles, self.coord_dim)
        pos_input = pos_flat.clone()

        # Edge index for batch
        edge_index = self._build_batch_edge_index(batch_size, device)

        # Message passing
        for layer in self.layers:
            h, pos_flat = layer(h, pos_flat, edge_index)

        # Position updates = final - initial
        dx = pos_flat - pos_input  # [batch * n_particles, coord_dim]
        dx = dx.view(batch_size, self.n_particles, self.coord_dim)

        # Remove center of mass
        dx = dx - dx.mean(dim=1, keepdim=True)

        # Flatten
        return dx.view(batch_size, -1)


class CorrectorEGNN(nn.Module):
    """
    Corrector h_ϕ(x) using EGNN — no time dependence.
    Zero-initialized output.
    """

    def __init__(
        self,
        n_particles: int,
        coord_dim: int = 3,
        hidden_dim: int = 128,
        n_layers: int = 4,
    ):
        super().__init__()
        self.n_particles = n_particles
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim

        # Constant node features (learnable)
        self.node_embed = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)

        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, coord_dim) for _ in range(n_layers)
        ])

        # Output scaling (zero-init so h^(0) ≡ 0)
        self.output_scale = nn.Parameter(torch.zeros(1))

        # Build fully-connected edge index
        src, dst = [], []
        for i in range(n_particles):
            for j in range(n_particles):
                if i != j:
                    src.append(i)
                    dst.append(j)
        self.register_buffer(
            "_edge_index_single",
            torch.tensor([src, dst], dtype=torch.long),
        )

    def _build_batch_edge_index(self, batch_size, device):
        edge_index = self._edge_index_single.to(device)
        offsets = torch.arange(batch_size, device=device).unsqueeze(1) * self.n_particles
        return (edge_index.unsqueeze(0) + offsets.unsqueeze(1)).reshape(2, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, n_particles * coord_dim]
        Returns: [batch, n_particles * coord_dim]
        """
        batch_size = x.shape[0]
        device = x.device

        pos = x.view(batch_size, self.n_particles, self.coord_dim)

        h = self.node_embed.expand(batch_size, self.n_particles, -1)
        h = h.reshape(batch_size * self.n_particles, self.hidden_dim)
        pos_flat = pos.reshape(batch_size * self.n_particles, self.coord_dim)
        pos_input = pos_flat.clone()

        edge_index = self._build_batch_edge_index(batch_size, device)

        for layer in self.layers:
            h, pos_flat = layer(h, pos_flat, edge_index)

        dx = pos_flat - pos_input
        dx = dx.view(batch_size, self.n_particles, self.coord_dim)
        dx = dx - dx.mean(dim=1, keepdim=True)

        return (dx * self.output_scale).view(batch_size, -1)
