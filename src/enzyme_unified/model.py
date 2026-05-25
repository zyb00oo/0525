from typing import Dict

import torch
import torch.nn as nn

from .gnn_encoder import TAGConvGNNEncoder
from .mol_graph_utils import ATOM_FEATURE_DIM, pad_atom_embeddings
from .protein3d_encoder import Protein3DEncoder
from .protein3d_utils import PROTEIN3D_EDGE_DIM, PROTEIN3D_FEATURE_DIM, pad_residue_embeddings
from .protein_multiview import ProteinMultiViewFuser
from .substrate_multiview import SubstrateMultiViewFuser


class MaskedAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), -1e4)
        weights = torch.softmax(scores, dim=1) * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class BiCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn_e = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_s = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_e1 = nn.LayerNorm(hidden_dim)
        self.norm_s1 = nn.LayerNorm(hidden_dim)
        self.ffn_e = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ffn_s = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm_e2 = nn.LayerNorm(hidden_dim)
        self.norm_s2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        e: torch.Tensor,
        s: torch.Tensor,
        e_padding_mask: torch.Tensor,
        s_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # MultiheadAttention 里 key_padding_mask=True 表示 padding 位
        e2, _ = self.attn_e(
            query=e,
            key=s,
            value=s,
            key_padding_mask=s_padding_mask,
            need_weights=False,
        )
        s2, _ = self.attn_s(
            query=s,
            key=e,
            value=e,
            key_padding_mask=e_padding_mask,
            need_weights=False,
        )
        e = self.norm_e1(e + e2)
        s = self.norm_s1(s + s2)
        e = self.norm_e2(e + self.ffn_e(e))
        s = self.norm_s2(s + self.ffn_s(s))
        return e, s


class EnzymeUnifiedModel(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        substrate_dim: int,
        maccs_dim: int = 167,
        physchem_dim: int = 22,
        hidden_dim: int = 768,
        num_heads: int = 8,
        cross_layers: int = 1,
        dropout: float = 0.1,
        use_physchem: bool = False,
        use_gnn: bool = False,
        gnn_d_atom: int = ATOM_FEATURE_DIM,
        gnn_hidden_dim: int = 256,
        gnn_output_dim: int = 256,
        gnn_layers: int = 3,
        gnn_k_hops: int = 3,
        gnn_dropout: float = 0.2,
        gnn_pooling: str = "mean",
        gnn_max_atoms: int = 128,
        gnn_fuse_dim: int = 768,
        use_protein3d: bool = False,
        protein3d_d_node: int = PROTEIN3D_FEATURE_DIM,
        protein3d_d_edge: int = PROTEIN3D_EDGE_DIM,
        protein3d_hidden_dim: int = 256,
        protein3d_output_dim: int = 256,
        protein3d_layers: int = 3,
        protein3d_dropout: float = 0.1,
        protein3d_max_residues: int = 1024,
        protein3d_fuse_dim: int = 768,
        protein3d_encoder: str = "transformer",
    ):
        super().__init__()
        self.use_physchem = use_physchem
        self.use_gnn = use_gnn
        self.use_protein3d = use_protein3d
        self.gnn_max_atoms = gnn_max_atoms
        self.protein3d_max_residues = protein3d_max_residues
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.substrate_proj = nn.Linear(substrate_dim, hidden_dim)
        self.maccs_proj = nn.Linear(maccs_dim, hidden_dim)

        self.cross_blocks = nn.ModuleList(
            [BiCrossAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(cross_layers)]
        )
        self.protein_readout = MaskedAttentionPooling(hidden_dim)
        self.substrate_readout = MaskedAttentionPooling(hidden_dim)

        self.attn_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        if self.use_gnn:
            self.gnn_encoder = TAGConvGNNEncoder(
                d_atom=gnn_d_atom,
                d_hidden=gnn_hidden_dim,
                d_output=gnn_output_dim,
                n_layers=gnn_layers,
                k_hops=gnn_k_hops,
                dropout=gnn_dropout,
                pooling=gnn_pooling,
            )
            self.substrate_fuser = SubstrateMultiViewFuser(
                d_molt5=hidden_dim,
                d_gnn=gnn_output_dim,
                d_proj=gnn_fuse_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            if gnn_fuse_dim != hidden_dim:
                raise ValueError("当前实现要求 gnn_fuse_dim 与 hidden_dim 一致。")
        else:
            self.gnn_encoder = None
            self.substrate_fuser = None

        if self.use_protein3d:
            self.protein3d_encoder = Protein3DEncoder(
                d_node=protein3d_d_node,
                d_edge=protein3d_d_edge,
                d_hidden=protein3d_hidden_dim,
                d_output=protein3d_output_dim,
                n_layers=protein3d_layers,
                dropout=protein3d_dropout,
                encoder_type=protein3d_encoder,
            )
            self.protein_fuser = ProteinMultiViewFuser(
                d_prott5=hidden_dim,
                d_protein3d=protein3d_output_dim,
                d_proj=protein3d_fuse_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            if protein3d_fuse_dim != hidden_dim:
                raise ValueError("当前实现要求 protein3d_fuse_dim 与 hidden_dim 一致。")
        else:
            self.protein3d_encoder = None
            self.protein_fuser = None

    def forward(self, feat: Dict[str, torch.Tensor]) -> torch.Tensor:
        e = self.protein_proj(feat["protein_token"])
        s = self.substrate_proj(feat["substrate_token"])
        e_mask = feat["protein_mask"] > 0
        s_mask = feat["substrate_mask"] > 0

        if self.use_protein3d:
            if self.protein3d_encoder is None or self.protein_fuser is None:
                raise RuntimeError("use_protein3d=True 但蛋白 3D 模块未正确初始化。")
            if "protein3d_batch" not in feat:
                raise KeyError("use_protein3d=True 时，输入特征必须包含 protein3d_batch。")

            protein3d_batch = feat["protein3d_batch"]
            residue_embeds = self.protein3d_encoder(
                x=protein3d_batch.x,
                edge_index=protein3d_batch.edge_index,
                edge_attr=protein3d_batch.edge_attr,
            )
            residue_padded, residue_mask = pad_residue_embeddings(
                residue_embeds=residue_embeds,
                batch_vector=protein3d_batch.batch,
                batch_size=protein3d_batch.num_graphs,
                max_residues=self.protein3d_max_residues,
            )
            if hasattr(protein3d_batch, "protein3d_missing"):
                missing = protein3d_batch.protein3d_missing.to(residue_mask.device).bool()
                residue_mask = residue_mask & ~missing.unsqueeze(-1)
            e, e_mask = self.protein_fuser(
                prott5_tokens=e,
                prott5_mask=e_mask,
                protein3d_padded=residue_padded,
                protein3d_mask=residue_mask,
            )

        if self.use_gnn:
            if self.gnn_encoder is None or self.substrate_fuser is None:
                raise RuntimeError("use_gnn=True 但 GNN 模块未正确初始化。")
            if "mol_graph_batch" not in feat:
                raise KeyError("use_gnn=True 时，输入特征必须包含 mol_graph_batch。")

            graph_batch = feat["mol_graph_batch"]
            atom_embeds, _ = self.gnn_encoder(
                x=graph_batch.x,
                edge_index=graph_batch.edge_index,
                batch=graph_batch.batch,
                return_graph=False,
            )
            atom_padded, atom_mask = pad_atom_embeddings(
                atom_embeds=atom_embeds,
                batch_vector=graph_batch.batch,
                max_atoms=self.gnn_max_atoms,
            )
            s, s_mask = self.substrate_fuser(
                molt5_tokens=s,
                molt5_mask=s_mask,
                gnn_atom_padded=atom_padded,
                gnn_atom_mask=atom_mask,
            )

        e_padding_mask = ~e_mask
        s_padding_mask = ~s_mask

        for block in self.cross_blocks:
            e, s = block(e, s, e_padding_mask=e_padding_mask, s_padding_mask=s_padding_mask)

        e_pool = self.protein_readout(e, e_mask)
        s_pool = self.substrate_readout(s, s_mask)
        maccs_proj = self.maccs_proj(feat["maccs"])
        attn_input = torch.cat([e_pool, s_pool, maccs_proj], dim=-1)
        y_attn = self.attn_head(attn_input)
        return y_attn.squeeze(-1)

