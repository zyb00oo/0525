import argparse
import json
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from src.enzyme_unified.gnn_encoder import TAGConvGNNEncoder
from src.enzyme_unified.mol_graph_utils import smiles_to_graph


class ReactionDataset(Dataset):
    def __init__(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as f:
            self.reactions: List[Dict[str, object]] = json.load(f)

    def __len__(self) -> int:
        return len(self.reactions)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.reactions[idx]
        subs = [smiles_to_graph(x) for x in row["substrates"]]
        pros = [smiles_to_graph(x) for x in row["products"]]
        return {"substrates": subs, "products": pros}


def reaction_collate_fn(batch):
    all_sub, all_pro = [], []
    sub_counts, pro_counts = [], []
    for item in batch:
        all_sub.extend(item["substrates"])
        all_pro.extend(item["products"])
        sub_counts.append(len(item["substrates"]))
        pro_counts.append(len(item["products"]))
    return {
        "substrate_batch": Batch.from_data_list(all_sub),
        "product_batch": Batch.from_data_list(all_pro),
        "substrate_counts": sub_counts,
        "product_counts": pro_counts,
    }


def compute_reaction_loss(model, batch, margin: float):
    _, sub_embed = model(batch["substrate_batch"].x, batch["substrate_batch"].edge_index, batch["substrate_batch"].batch)
    _, pro_embed = model(batch["product_batch"].x, batch["product_batch"].edge_index, batch["product_batch"].batch)

    sub_sums = []
    ptr = 0
    for count in batch["substrate_counts"]:
        sub_sums.append(sub_embed[ptr : ptr + count].sum(dim=0))
        ptr += count
    sub_sums = torch.stack(sub_sums)

    pro_sums = []
    ptr = 0
    for count in batch["product_counts"]:
        pro_sums.append(pro_embed[ptr : ptr + count].sum(dim=0))
        ptr += count
    pro_sums = torch.stack(pro_sums)

    diff_pos = sub_sums - pro_sums
    l_pos = (diff_pos.square().sum(dim=1)).mean()

    pair_diff = sub_sums.unsqueeze(1) - pro_sums.unsqueeze(0)
    pair_dist = pair_diff.square().sum(dim=2)
    diag_mask = ~torch.eye(pair_dist.size(0), dtype=torch.bool, device=pair_dist.device)
    l_neg = torch.clamp(margin - pair_dist, min=0.0) * diag_mask.float()
    denom = max(1, pair_dist.size(0) * (pair_dist.size(0) - 1))
    return l_pos + l_neg.sum() / denom


def main():
    parser = argparse.ArgumentParser(description="Reaction-aware pretraining for TAGConv GNN.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="checkpoints/pretrained_gnn.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=4.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = TAGConvGNNEncoder(d_atom=37, d_hidden=256, d_output=256, n_layers=3, k_hops=3, pooling="mean").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    loader = DataLoader(
        ReactionDataset(args.data_path),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=reaction_collate_fn,
    )

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for batch in loader:
            batch["substrate_batch"] = batch["substrate_batch"].to(device)
            batch["product_batch"] = batch["product_batch"].to(device)
            loss = compute_reaction_loss(model, batch, margin=args.margin)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += float(loss.item())
            steps += 1

        avg_loss = total / max(steps, 1)
        print(f"[epoch {epoch}] pretrain_loss={avg_loss:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"[save] {args.save_path}")


if __name__ == "__main__":
    main()
