from typing import List, Optional, Tuple

import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data

from .chem_utils import smiles_to_mol_safe


ATOM_LIST = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "other"]
DEGREE_LIST = [0, 1, 2, 3, 4, 5]
CHARGE_LIST = [-2, -1, 0, 1, 2]
CHIRAL_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
NUMH_LIST = [0, 1, 2, 3, 4]
HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

ATOM_FEATURE_DIM = (
    len(ATOM_LIST)
    + len(DEGREE_LIST)
    + len(CHARGE_LIST)
    + len(CHIRAL_LIST)
    + len(NUMH_LIST)
    + len(HYBRIDIZATION_LIST)
    + 1
    + 1
)

BOND_TYPE_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
STEREO_LIST = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]
BOND_FEATURE_DIM = len(BOND_TYPE_LIST) + 1 + 1 + len(STEREO_LIST)


def _one_hot(value, allowed_values):
    encoding = [0.0] * len(allowed_values)
    if value in allowed_values:
        encoding[allowed_values.index(value)] = 1.0
    else:
        encoding[-1] = 1.0
    return encoding


def _atom_features(atom: Chem.rdchem.Atom) -> List[float]:
    features: List[float] = []
    features += _one_hot(atom.GetSymbol(), ATOM_LIST)
    features += _one_hot(atom.GetTotalDegree(), DEGREE_LIST)
    features += _one_hot(atom.GetFormalCharge(), CHARGE_LIST)
    features += _one_hot(atom.GetChiralTag(), CHIRAL_LIST)
    features += _one_hot(atom.GetTotalNumHs(), NUMH_LIST)
    features += _one_hot(atom.GetHybridization(), HYBRIDIZATION_LIST)
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    features.append(1.0 if atom.IsInRing() else 0.0)
    return features


def _bond_features(bond: Chem.rdchem.Bond) -> List[float]:
    features: List[float] = []
    features += _one_hot(bond.GetBondType(), BOND_TYPE_LIST)
    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    features.append(1.0 if bond.IsInRing() else 0.0)
    features += _one_hot(bond.GetStereo(), STEREO_LIST)
    return features


def smiles_to_graph(smiles: str) -> Data:
    mol = smiles_to_mol_safe(smiles)
    if mol is None:
        return Data(
            x=torch.zeros(1, ATOM_FEATURE_DIM, dtype=torch.float),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, BOND_FEATURE_DIM, dtype=torch.float),
            num_atoms=1,
        )

    atom_features = [_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_attrs.append(bf)
        edge_attrs.append(bf)

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, BOND_FEATURE_DIM, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_atoms=int(x.shape[0]))


def collate_mol_graphs(graph_list: List[Data]) -> Batch:
    return Batch.from_data_list(graph_list)


def pad_atom_embeddings(
    atom_embeds: torch.Tensor,
    batch_vector: torch.Tensor,
    max_atoms: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if atom_embeds.numel() == 0:
        raise ValueError("atom_embeds 不能为空")

    batch_size = int(batch_vector.max().item()) + 1
    hidden_dim = atom_embeds.size(1)

    counts = torch.bincount(batch_vector, minlength=batch_size)
    max_len = int(counts.max().item()) if max_atoms is None else max_atoms

    padded = torch.zeros(batch_size, max_len, hidden_dim, device=atom_embeds.device)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=atom_embeds.device)

    cursor = 0
    for i in range(batch_size):
        n = int(counts[i].item())
        fill_n = min(n, max_len)
        padded[i, :fill_n] = atom_embeds[cursor : cursor + fill_n]
        mask[i, :fill_n] = True
        cursor += n

    return padded, mask
