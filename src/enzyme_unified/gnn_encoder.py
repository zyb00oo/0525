from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import TAGConv, global_max_pool, global_mean_pool


class TAGConvGNNEncoder(nn.Module):
    def __init__(
        self,
        d_atom: int = 37,
        d_hidden: int = 256,
        d_output: int = 256,
        n_layers: int = 3,
        k_hops: int = 3,
        dropout: float = 0.2,
        pooling: str = "mean",
    ):
        super().__init__()
        self.pooling = pooling

        self.input_proj = nn.Sequential(
            nn.Linear(d_atom, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.convs = nn.ModuleList([TAGConv(d_hidden, d_hidden, K=k_hops) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(d_hidden) for _ in range(n_layers)])
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_hidden, d_hidden * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_hidden * 2, d_hidden),
                )
                for _ in range(n_layers)
            ]
        )
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(n_layers)])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.output_proj = nn.Linear(d_hidden, d_output) if d_output != d_hidden else nn.Identity()
        self.output_norm = nn.LayerNorm(d_output)
        self.mean_max_proj = nn.Linear(d_output * 2, d_output) if pooling == "mean_max" else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        return_graph: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.input_proj(x)

        for conv, norm, ffn, ffn_norm in zip(self.convs, self.norms, self.ffns, self.ffn_norms):
            residual = h
            h = conv(h, edge_index)
            h = norm(h)
            h = self.activation(h)
            h = self.dropout(h)
            h = h + residual
            h = ffn_norm(h + ffn(h))

        atom_embeds = self.output_norm(self.output_proj(h))

        if not return_graph:
            return atom_embeds, None

        if self.pooling == "mean":
            graph_embed = global_mean_pool(atom_embeds, batch)
        elif self.pooling == "max":
            graph_embed = global_max_pool(atom_embeds, batch)
        elif self.pooling == "mean_max":
            mean_pool = global_mean_pool(atom_embeds, batch)
            max_pool = global_max_pool(atom_embeds, batch)
            graph_embed = self.mean_max_proj(torch.cat([mean_pool, max_pool], dim=-1))
        else:
            raise ValueError(f"不支持的 pooling: {self.pooling}")

        return atom_embeds, graph_embed
