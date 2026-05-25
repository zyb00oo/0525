from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .mol_graph_utils import collate_mol_graphs, smiles_to_graph
from .protein3d_utils import collate_protein3d_graphs, load_protein3d_graph, load_protein3d_index


@dataclass
class SplitData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def load_task_dataframe(csv_path: str, label_col: str, log_target: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"Sequence", "Smiles", "fold", label_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺失列: {sorted(missing)}")

    df = df.dropna(subset=["Sequence", "Smiles", "fold", label_col]).copy()
    df["fold"] = df["fold"].astype(int)
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col]).copy()
    if log_target:
        df = df[df[label_col] > 0].copy()
    return df.reset_index(drop=True)


def build_fold_split(
    df: pd.DataFrame,
    total_folds: int,
    test_fold: int,
    strategy: str,
    seed: int,
) -> SplitData:
    if not (0 <= test_fold < total_folds):
        raise ValueError(f"test_fold={test_fold} 超出范围 [0, {total_folds - 1}]")

    test_df = df[df["fold"] == test_fold].copy()
    pool_df = df[df["fold"] != test_fold].copy()
    if strategy == "modulo1":
        val_fold = (test_fold + 1) % total_folds
        val_df = pool_df[pool_df["fold"] == val_fold].copy()
        train_df = pool_df[pool_df["fold"] != val_fold].copy()
    elif strategy == "random90_10":
        rng = np.random.default_rng(seed + test_fold)
        idx = np.arange(len(pool_df))
        rng.shuffle(idx)
        cut = int(len(idx) * 0.9)
        train_idx = idx[:cut]
        val_idx = idx[cut:]
        train_df = pool_df.iloc[train_idx].copy()
        val_df = pool_df.iloc[val_idx].copy()
    else:
        raise ValueError(f"未知划分策略: {strategy}")

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("划分后存在空集合，请检查 fold 或策略。")
    return SplitData(train_df=train_df, val_df=val_df, test_df=test_df)


class EnzymeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str,
        use_gnn: bool = False,
        use_protein3d: bool = False,
        protein3d_index: str | None = None,
        protein3d_cache_dir: str | None = None,
        protein3d_max_residues: int | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.use_gnn = use_gnn
        self.use_protein3d = use_protein3d
        self.protein3d_max_residues = protein3d_max_residues
        self.mol_graphs = None
        self.protein3d_paths = load_protein3d_index(protein3d_index, protein3d_cache_dir) if self.use_protein3d else {}
        if self.use_gnn:
            smiles_list = self.df["Smiles"].astype(str).tolist()
            self.mol_graphs = [smiles_to_graph(smiles) for smiles in smiles_list]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        sample = {
            "sequence": str(row["Sequence"]),
            "smiles": str(row["Smiles"]),
            "label_raw": float(row[self.label_col]),
        }
        if self.use_gnn and self.mol_graphs is not None:
            sample["mol_graph"] = self.mol_graphs[idx]
        if self.use_protein3d:
            sequence = sample["sequence"]
            sample["protein3d"] = load_protein3d_graph(
                self.protein3d_paths.get(sequence),
                max_residues=self.protein3d_max_residues,
            )
        return sample


def collate_samples(samples: List[Dict[str, object]]) -> Dict[str, object]:
    collated = {
        "sequence": [item["sequence"] for item in samples],
        "smiles": [item["smiles"] for item in samples],
        "label_raw": torch.tensor([item["label_raw"] for item in samples], dtype=torch.float32),
    }
    if "mol_graph" in samples[0]:
        collated["mol_graph_batch"] = collate_mol_graphs([item["mol_graph"] for item in samples])
    if "protein3d" in samples[0]:
        collated["protein3d_batch"] = collate_protein3d_graphs([item["protein3d"] for item in samples])
    return collated

