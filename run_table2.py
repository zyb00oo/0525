import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.enzyme_unified.config import TASK_CONFIGS


def parse_args():
    parser = argparse.ArgumentParser(description="Run Table 2 experiments across folds and variants.")
    parser.add_argument("--variants", nargs="+", default=["hybrid", "hybrid_prostt5", "hybrid_pp"])
    parser.add_argument("--tasks", nargs="+", default=["kcat_km", "ph", "topt"])
    parser.add_argument("--split_strategy", choices=["modulo1", "random90_10"], default="modulo1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--freeze_encoders", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--use_gnn", action="store_true")
    parser.add_argument("--gnn_hidden_dim", type=int, default=256)
    parser.add_argument("--gnn_output_dim", type=int, default=256)
    parser.add_argument("--gnn_layers", type=int, default=3)
    parser.add_argument("--gnn_k_hops", type=int, default=3)
    parser.add_argument("--gnn_dropout", type=float, default=0.2)
    parser.add_argument("--gnn_pooling", choices=["mean", "max", "mean_max"], default="mean")
    parser.add_argument("--gnn_max_atoms", type=int, default=128)
    parser.add_argument("--gnn_fuse_dim", type=int, default=768)
    parser.add_argument("--gnn_lr_scale", type=float, default=1.0)
    parser.add_argument("--gnn_pretrained_path", type=str, default=None, help="保留兼容旧命令；当前框架会忽略该参数并从头训练 GNN。")
    parser.add_argument("--freeze_gnn", action="store_true")
    parser.add_argument("--use_protein3d", action="store_true")
    parser.add_argument("--protein3d_index", type=str, default=None)
    parser.add_argument("--protein3d_cache_dir", type=str, default=None)
    parser.add_argument("--protein3d_max_residues", type=int, default=1024)
    parser.add_argument("--protein3d_hidden_dim", type=int, default=256)
    parser.add_argument("--protein3d_output_dim", type=int, default=256)
    parser.add_argument("--protein3d_layers", type=int, default=3)
    parser.add_argument("--protein3d_dropout", type=float, default=0.1)
    parser.add_argument("--protein3d_fuse_dim", type=int, default=768)
    parser.add_argument("--protein3d_encoder", choices=["transformer", "gine"], default="transformer")
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=-1,
        help=">0 时固定值；<=0 时按任务默认全局batch自动计算。",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Per-process batch size.")
    parser.add_argument("--output_root", type=str, default="results/table2")
    parser.add_argument("--python_exec", type=str, default="python")
    parser.add_argument("--launcher", choices=["python", "torchrun"], default="python")
    parser.add_argument("--nproc_per_node", type=int, default=1)
    return parser.parse_args()


def summarize_fold_metrics(task_dir: Path, folds: int) -> dict:
    rows = []
    for fold in range(folds):
        metrics_path = task_dir / f"fold_{fold}" / "metrics.json"
        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        test_metrics = payload["test_metrics"]
        rows.append(
            {
                "fold": fold,
                "rmse": test_metrics["rmse"],
                "pcc": test_metrics["pcc"],
                "scc": test_metrics["scc"],
            }
        )
    df = pd.DataFrame(rows)
    return {
        "rmse_mean": float(df["rmse"].mean()),
        "rmse_std": float(df["rmse"].std(ddof=0)),
        "pcc_mean": float(df["pcc"].mean()),
        "pcc_std": float(df["pcc"].std(ddof=0)),
        "scc_mean": float(df["scc"].mean()),
        "scc_std": float(df["scc"].std(ddof=0)),
    }


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for task in args.tasks:
        task_cfg = TASK_CONFIGS[task]
        folds = task_cfg["folds"]
        for variant in args.variants:
            variant_task_dir = output_root / task / variant
            variant_task_dir.mkdir(parents=True, exist_ok=True)
            for fold in range(folds):
                fold_dir = variant_task_dir / f"fold_{fold}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                if args.launcher == "torchrun":
                    cmd = [
                        "torchrun",
                        "--nproc_per_node",
                        str(args.nproc_per_node),
                        "train_unified.py",
                    ]
                else:
                    cmd = [args.python_exec, "train_unified.py"]

                if args.grad_accum_steps > 0:
                    grad_accum_steps = args.grad_accum_steps
                else:
                    per_proc_bs = args.batch_size if args.batch_size is not None else task_cfg["default_batch_size"]
                    global_bs = task_cfg["default_batch_size"]
                    world = args.nproc_per_node if args.launcher == "torchrun" else 1
                    grad_accum_steps = max(1, global_bs // max(1, per_proc_bs * world))

                cmd += [
                    "--task",
                    task,
                    "--variant",
                    variant,
                    "--test_fold",
                    str(fold),
                    "--split_strategy",
                    args.split_strategy,
                    "--seed",
                    str(args.seed),
                    "--max_epochs",
                    str(args.max_epochs),
                    "--patience",
                    str(args.patience),
                    "--grad_accum_steps",
                    str(grad_accum_steps),
                    "--output_dir",
                    str(fold_dir),
                ]
                if args.batch_size is not None:
                    cmd += ["--batch_size", str(args.batch_size)]
                if args.freeze_encoders:
                    cmd.append("--freeze_encoders")
                if args.mixed_precision:
                    cmd.append("--mixed_precision")
                if args.use_gnn:
                    cmd += [
                        "--use_gnn",
                        "--gnn_hidden_dim",
                        str(args.gnn_hidden_dim),
                        "--gnn_output_dim",
                        str(args.gnn_output_dim),
                        "--gnn_layers",
                        str(args.gnn_layers),
                        "--gnn_k_hops",
                        str(args.gnn_k_hops),
                        "--gnn_dropout",
                        str(args.gnn_dropout),
                        "--gnn_pooling",
                        str(args.gnn_pooling),
                        "--gnn_max_atoms",
                        str(args.gnn_max_atoms),
                        "--gnn_fuse_dim",
                        str(args.gnn_fuse_dim),
                        "--gnn_lr_scale",
                        str(args.gnn_lr_scale),
                    ]
                    if args.gnn_pretrained_path:
                        print(f"[GNN] ignored pretrained weights: {args.gnn_pretrained_path}")
                    if args.freeze_gnn:
                        cmd += ["--freeze_gnn"]
                if args.use_protein3d:
                    cmd += [
                        "--use_protein3d",
                        "--protein3d_max_residues",
                        str(args.protein3d_max_residues),
                        "--protein3d_hidden_dim",
                        str(args.protein3d_hidden_dim),
                        "--protein3d_output_dim",
                        str(args.protein3d_output_dim),
                        "--protein3d_layers",
                        str(args.protein3d_layers),
                        "--protein3d_dropout",
                        str(args.protein3d_dropout),
                        "--protein3d_fuse_dim",
                        str(args.protein3d_fuse_dim),
                        "--protein3d_encoder",
                        args.protein3d_encoder,
                    ]
                    if args.protein3d_index:
                        cmd += ["--protein3d_index", args.protein3d_index]
                    if args.protein3d_cache_dir:
                        cmd += ["--protein3d_cache_dir", args.protein3d_cache_dir]
                print(" ".join(cmd))
                subprocess.run(cmd, check=True)

            metrics = summarize_fold_metrics(variant_task_dir, folds=folds)
            summary_rows.append(
                {
                    "task": task,
                    "variant": variant,
                    **metrics,
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["task", "variant"]).reset_index(drop=True)
    summary_csv = output_root / "table2_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[DONE] summary saved to {summary_csv}")
    print(summary_df)


if __name__ == "__main__":
    main()

