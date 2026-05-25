import torch
import torch.nn as nn


class ProteinMultiViewFuser(nn.Module):
    def __init__(self, d_prott5: int, d_protein3d: int, d_proj: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.prott5_proj = nn.LayerNorm(d_proj) if d_prott5 == d_proj else nn.Sequential(nn.Linear(d_prott5, d_proj), nn.LayerNorm(d_proj))
        self.protein3d_proj = nn.Sequential(nn.Linear(d_protein3d, d_proj), nn.LayerNorm(d_proj))
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
        prott5_tokens: torch.Tensor,
        prott5_mask: torch.Tensor,
        protein3d_padded: torch.Tensor,
        protein3d_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = prott5_tokens.shape
        structure_len = protein3d_padded.size(1)
        device = prott5_tokens.device

        seq_tokens = self.prott5_proj(prott5_tokens)
        structure_tokens = self.protein3d_proj(protein3d_padded)
        seq_type = self.type_embedding(torch.zeros(batch_size, seq_len, dtype=torch.long, device=device))
        structure_type = self.type_embedding(torch.ones(batch_size, structure_len, dtype=torch.long, device=device))

        seq_tokens = seq_tokens + seq_type
        structure_tokens = structure_tokens + structure_type
        fused_tokens = torch.cat([seq_tokens, structure_tokens], dim=1)
        fused_mask = torch.cat([prott5_mask.bool(), protein3d_mask.bool()], dim=1)
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
