from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data


PROTEIN3D_FEATURE_DIM = 33
PROTEIN3D_EDGE_DIM = 18


def load_protein3d_index(index_path: Optional[str], cache_dir: Optional[str] = None) -> Dict[str, str]:
    if not index_path:
        return {}
    df = pd.read_csv(index_path)
    if "sequence" not in df.columns:
        return {}

    mapping: Dict[str, str] = {}
    base_dir = Path(cache_dir) if cache_dir else Path(index_path).parent / "cache"
    for _, row in df.iterrows():
        protein_id = str(row.get("protein_id", "")).strip()
        sequence = str(row.get("sequence", "")).strip()
        status = str(row.get("status", "ok")).strip()
        if not protein_id or not sequence or status not in {"ok", ""}:
            continue
        cache_path = str(row.get("cache_path", "")).strip()
        if cache_path:
            path = Path(cache_path)
        else:
            path = base_dir / f"{protein_id}.npz"
        if path.exists():
            mapping[sequence] = str(path)
    return mapping


def empty_protein3d_graph() -> Data:
    return Data(
        x=torch.zeros(1, PROTEIN3D_FEATURE_DIM, dtype=torch.float),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, PROTEIN3D_EDGE_DIM, dtype=torch.float),
        pos=torch.zeros(1, 3, dtype=torch.float),
        protein3d_missing=torch.tensor([1], dtype=torch.bool),
    )


def load_protein3d_graph(cache_path: Optional[str], max_residues: Optional[int] = None) -> Data:
    if not cache_path:
        return empty_protein3d_graph()
    path = Path(cache_path)
    if not path.exists():
        return empty_protein3d_graph()

    payload = np.load(path)
    x_np = payload["residue_feat"].astype(np.float32)
    pos_np = payload["residue_pos"].astype(np.float32)
    edge_index_np = payload["edge_index"].astype(np.int64)
    edge_attr_np = payload["edge_attr"].astype(np.float32)

    if max_residues is not None and max_residues > 0 and x_np.shape[0] > max_residues:
        keep = max_residues
        edge_mask = (edge_index_np[0] < keep) & (edge_index_np[1] < keep)
        x_np = x_np[:keep]
        pos_np = pos_np[:keep]
        edge_index_np = edge_index_np[:, edge_mask]
        edge_attr_np = edge_attr_np[edge_mask]

    if x_np.size == 0:
        return empty_protein3d_graph()

    return Data(
        x=torch.tensor(x_np, dtype=torch.float),
        edge_index=torch.tensor(edge_index_np, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr_np, dtype=torch.float),
        pos=torch.tensor(pos_np, dtype=torch.float),
        protein3d_missing=torch.tensor([0], dtype=torch.bool),
    )


def collate_protein3d_graphs(graph_list: List[Data]) -> Batch:
    return Batch.from_data_list(graph_list)


def pad_residue_embeddings(
    residue_embeds: torch.Tensor,
    batch_vector: torch.Tensor,
    batch_size: int,
    max_residues: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden_dim = residue_embeds.size(1)
    if residue_embeds.numel() == 0:
        max_len = max_residues or 1
        return (
            torch.zeros(batch_size, max_len, hidden_dim, device=residue_embeds.device),
            torch.zeros(batch_size, max_len, dtype=torch.bool, device=residue_embeds.device),
        )

    counts = torch.bincount(batch_vector, minlength=batch_size)
    max_len = int(counts.max().item()) if max_residues is None else max_residues
    padded = torch.zeros(batch_size, max_len, hidden_dim, device=residue_embeds.device)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=residue_embeds.device)

    cursor = 0
    for idx in range(batch_size):
        n_residues = int(counts[idx].item())
        fill_n = min(n_residues, max_len)
        padded[idx, :fill_n] = residue_embeds[cursor : cursor + fill_n]
        mask[idx, :fill_n] = True
        cursor += n_residues
    return padded, mask
