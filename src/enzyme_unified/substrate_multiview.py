import torch
import torch.nn as nn


class SubstrateMultiViewFuser(nn.Module):
    def __init__(self, d_molt5: int, d_gnn: int, d_proj: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.molt5_proj = (
            nn.Sequential(nn.Linear(d_molt5, d_proj), nn.LayerNorm(d_proj))
            if d_molt5 != d_proj
            else nn.LayerNorm(d_proj)
        )
        self.gnn_proj = nn.Sequential(nn.Linear(d_gnn, d_proj), nn.LayerNorm(d_proj))
        self.type_embedding = nn.Embedding(2, d_proj)
        self.fusion_attn = nn.MultiheadAttention(d_proj, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_proj)
        self.ffn = nn.Sequential(
            nn.Linear(d_proj, d_proj * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj * 2, d_proj),
        )
        self.norm2 = nn.LayerNorm(d_proj)

    def forward(
        self,
        molt5_tokens: torch.Tensor,
        molt5_mask: torch.Tensor,
        gnn_atom_padded: torch.Tensor,
        gnn_atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = molt5_tokens.shape
        atom_len = gnn_atom_padded.size(1)
        device = molt5_tokens.device

        seq_tokens = self.molt5_proj(molt5_tokens)
        atom_tokens = self.gnn_proj(gnn_atom_padded)

        seq_type = self.type_embedding(torch.zeros(batch_size, seq_len, dtype=torch.long, device=device))
        atom_type = self.type_embedding(torch.ones(batch_size, atom_len, dtype=torch.long, device=device))

        seq_tokens = seq_tokens + seq_type
        atom_tokens = atom_tokens + atom_type

        fused_tokens = torch.cat([seq_tokens, atom_tokens], dim=1)
        fused_mask = torch.cat([molt5_mask.bool(), gnn_atom_mask.bool()], dim=1)
        padding_mask = ~fused_mask

        attn_out, _ = self.fusion_attn(
            query=fused_tokens,
            key=fused_tokens,
            value=fused_tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        fused_tokens = self.norm1(fused_tokens + self.attn_dropout(attn_out))
        fused_tokens = self.norm2(fused_tokens + self.ffn(fused_tokens))
        fused_tokens = fused_tokens * fused_mask.unsqueeze(-1).float()
        return fused_tokens, fused_mask
