import torch

from src.enzyme_unified.gnn_encoder import TAGConvGNNEncoder
from src.enzyme_unified.model import EnzymeUnifiedModel
from src.enzyme_unified.mol_graph_utils import collate_mol_graphs, pad_atom_embeddings, smiles_to_graph
from src.enzyme_unified.protein3d_encoder import Protein3DEncoder
from src.enzyme_unified.protein3d_utils import PROTEIN3D_EDGE_DIM, PROTEIN3D_FEATURE_DIM, collate_protein3d_graphs
from src.enzyme_unified.protein_multiview import ProteinMultiViewFuser
from src.enzyme_unified.substrate_multiview import SubstrateMultiViewFuser
from torch_geometric.data import Data


def test_smiles_to_graph():
    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "InvalidSMILES"]
    for smi in smiles_list:
        graph = smiles_to_graph(smi)
        assert graph.x.shape[1] == 37


def test_gnn_encoder():
    gnn = TAGConvGNNEncoder(d_atom=37, d_hidden=64, d_output=64, n_layers=2, k_hops=2)
    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "CC(C)CC"]
    batch = collate_mol_graphs([smiles_to_graph(s) for s in smiles_list])
    atom_embeds, graph_embed = gnn(batch.x, batch.edge_index, batch.batch)
    assert graph_embed.shape == (len(smiles_list), 64)
    assert atom_embeds.shape[1] == 64


def test_pad_atom_embeddings():
    gnn = TAGConvGNNEncoder(d_atom=37, d_hidden=32, d_output=32, n_layers=2, k_hops=2)
    batch = collate_mol_graphs([smiles_to_graph("CCO"), smiles_to_graph("c1ccccc1")])
    atom_embeds, _ = gnn(batch.x, batch.edge_index, batch.batch)
    padded, mask = pad_atom_embeddings(atom_embeds, batch.batch, max_atoms=10)
    assert padded.shape == (2, 10, 32)
    assert mask.shape == (2, 10)


def test_multiview_fuser():
    fuser = SubstrateMultiViewFuser(d_molt5=768, d_gnn=64, d_proj=768)
    batch_size, token_len, atom_len = 2, 20, 10
    mol_tokens = torch.randn(batch_size, token_len, 768)
    mol_mask = torch.ones(batch_size, token_len, dtype=torch.bool)
    atom_tokens = torch.randn(batch_size, atom_len, 64)
    atom_mask = torch.ones(batch_size, atom_len, dtype=torch.bool)
    fused_tokens, fused_mask = fuser(mol_tokens, mol_mask, atom_tokens, atom_mask)
    assert fused_tokens.shape == (batch_size, token_len + atom_len, 768)
    assert fused_mask.shape == (batch_size, token_len + atom_len)


def _protein3d_graph(num_residues: int) -> Data:
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.randn(edge_index.size(1), PROTEIN3D_EDGE_DIM)
    return Data(
        x=torch.randn(num_residues, PROTEIN3D_FEATURE_DIM),
        edge_index=edge_index,
        edge_attr=edge_attr,
        protein3d_missing=torch.tensor([0], dtype=torch.bool),
    )


def test_protein3d_encoder():
    encoder = Protein3DEncoder(
        d_node=PROTEIN3D_FEATURE_DIM,
        d_edge=PROTEIN3D_EDGE_DIM,
        d_hidden=32,
        d_output=32,
        n_layers=2,
        encoder_type="transformer",
        num_heads=4,
    )
    batch = collate_protein3d_graphs([_protein3d_graph(3), _protein3d_graph(3)])
    residue_embeds = encoder(batch.x, batch.edge_index, batch.edge_attr)
    assert residue_embeds.shape == (6, 32)


def test_protein_multiview_fuser():
    fuser = ProteinMultiViewFuser(d_prott5=32, d_protein3d=16, d_proj=32, num_heads=4)
    prott5_tokens = torch.randn(2, 5, 32)
    prott5_mask = torch.ones(2, 5, dtype=torch.bool)
    protein3d_tokens = torch.randn(2, 4, 16)
    protein3d_mask = torch.ones(2, 4, dtype=torch.bool)
    fused_tokens, fused_mask = fuser(prott5_tokens, prott5_mask, protein3d_tokens, protein3d_mask)
    assert fused_tokens.shape == (2, 9, 32)
    assert fused_mask.shape == (2, 9)


def test_model_with_protein3d_forward():
    model = EnzymeUnifiedModel(
        protein_dim=32,
        substrate_dim=32,
        hidden_dim=32,
        num_heads=4,
        cross_layers=1,
        use_protein3d=True,
        protein3d_hidden_dim=32,
        protein3d_output_dim=32,
        protein3d_layers=1,
        protein3d_max_residues=4,
        protein3d_fuse_dim=32,
        protein3d_encoder="transformer",
    )
    feat = {
        "protein_token": torch.randn(2, 5, 32),
        "protein_mask": torch.ones(2, 5, dtype=torch.long),
        "substrate_token": torch.randn(2, 4, 32),
        "substrate_mask": torch.ones(2, 4, dtype=torch.long),
        "maccs": torch.randn(2, 167),
        "protein3d_batch": collate_protein3d_graphs([_protein3d_graph(3), _protein3d_graph(3)]),
    }
    pred = model(feat)
    assert pred.shape == (2,)
