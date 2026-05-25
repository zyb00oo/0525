import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, TransformerConv


class Protein3DEncoder(nn.Module):
    def __init__(
        self,
        d_node: int,
        d_edge: int,
        d_hidden: int = 256,
        d_output: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
        encoder_type: str = "transformer",
        num_heads: int = 4,
    ):
        super().__init__()
        if encoder_type not in {"transformer", "gine"}:
            raise ValueError(f"不支持的 protein3d_encoder: {encoder_type}")
        self.encoder_type = encoder_type

        self.input_proj = nn.Sequential(
            nn.Linear(d_node, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(d_edge, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            if encoder_type == "transformer":
                self.convs.append(
                    TransformerConv(
                        in_channels=d_hidden,
                        out_channels=d_hidden,
                        heads=num_heads,
                        concat=False,
                        edge_dim=d_hidden,
                        dropout=dropout,
                    )
                )
            else:
                nn_module = nn.Sequential(
                    nn.Linear(d_hidden, d_hidden * 2),
                    nn.GELU(),
                    nn.Linear(d_hidden * 2, d_hidden),
                )
                self.convs.append(GINEConv(nn_module, edge_dim=d_hidden))

        self.norms = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(n_layers)])
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
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_hidden, d_output) if d_hidden != d_output else nn.Identity()
        self.output_norm = nn.LayerNorm(d_output)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        edge_hidden = self.edge_proj(edge_attr) if edge_attr.numel() > 0 else edge_attr.new_zeros((0, h.size(-1)))

        for conv, norm, ffn, ffn_norm in zip(self.convs, self.norms, self.ffns, self.ffn_norms):
            residual = h
            h = conv(h, edge_index, edge_hidden)
            h = norm(residual + self.dropout(h))
            h = ffn_norm(h + ffn(h))
        return self.output_norm(self.output_proj(h))
