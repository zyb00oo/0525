# Enzyme-Unified + TAGConv-GNN

本项目在原始 **Enzyme-Unified** 基线之上，加入 TAGConv 分子图编码分支，用于增强 MolT5 底物 token 表示。

当前版本已经更新为新的 pipeline：去掉旧的全局拼接预测分支，只保留细粒度酶-底物 cross-attention 主干；GNN 不再加载预训练权重，而是从头与主模型一起训练；GNN 输出的原子级 token 会先与 MolT5 token 做分子端 self-attention 融合，再进入酶-底物双向 cross-attention。

## 1. 目录说明

- `dataset/`: 已处理好的 CSV 数据（你已提供）
- `src/enzyme_unified/`: 模型、特征、训练逻辑
  - `mol_graph_utils.py`: SMILES -> PyG 分子图转换、batch 拼接、atom padding
  - `gnn_encoder.py`: TAGConv 编码器，当前主流程使用原子级嵌入
  - `substrate_multiview.py`: MolT5 token 与 GNN atom token 的 self-attention 融合
  - `model.py`: 细粒度双向 Cross-Attention + Attention Pooling 回归模型
- `train_unified.py`: 单任务单折训练入口
- `run_table2.py`: 按任务 × 变体 × 折批量运行并汇总结果
- `docs/pipeline.md`: 当前 pipeline 的详细说明
- `results/`: 输出目录（checkpoint、metrics、table2_summary）

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

如果是首次安装 PyG，`torch-scatter` / `torch-sparse` 需和本地 PyTorch/CUDA 匹配，必要时按 [PyG 官方安装文档](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) 指定轮子源。

## 3. 当前模型 Pipeline

当前主流程如下：

```text
CSV 数据
→ Dataset / DataLoader
→ ProtT5 或 ProstT5 编码蛋白序列
→ 可选 ColabFold 预计算蛋白 3D residue 图编码
→ MolT5 编码底物 SMILES
→ 可选 TAGConv-GNN 编码分子图原子表示
→ ProtT5 token 与蛋白 3D residue token 做蛋白端 self-attention 融合
→ MolT5 token 与 GNN atom token 做分子端 self-attention 融合
→ 蛋白 token 与融合后的底物 token 做双向 cross-attention
→ attention pooling 读出蛋白表示和底物表示
→ 拼接 MACCS 指纹投影
→ MLP 回归头
→ 输出预测值
```

当前最终预测来自：

```text
交叉注意力后的蛋白表示
+
交叉注意力后的底物表示
+
MACCS 指纹投影
```

当前不再使用旧版本中的：

```text
protein_pool
substrate_pool
GNN graph-level pooling
physchem
concat_head
alpha gate
```

注意：

- `--gnn_pretrained_path` 当前会被忽略，GNN 从头训练。
- `--gnn_pooling` 当前不影响主模型预测，因为主流程只使用 GNN 原子级 token，不使用 graph-level embedding。
- `hybrid_pp` 当前会计算 `physchem`，但模型暂时不消费它，因此不建议把当前 `hybrid_pp` 解释为“加入理化特征”的有效消融。
- 更详细的结构说明见 `docs/pipeline.md`。

## 4.1 可选蛋白 3D 结构分支

数据集没有蛋白结构字段。当前做法是先离线生成结构缓存，再训练时读取：

```bash
python scripts/export_unique_proteins.py \
  --output_dir data/protein3d

colabfold_batch \
  data/protein3d/unique_proteins.fasta \
  data/protein3d/colabfold_out

python scripts/build_protein3d_cache.py \
  --protein_index data/protein3d/protein_index.csv \
  --colabfold_dir data/protein3d/colabfold_out \
  --cache_dir data/protein3d/cache
```

缓存会生成 `data/protein3d/protein3d_index.csv` 和 `data/protein3d/cache/*.npz`。每个 `.npz` 包含 residue-level CA 坐标、pLDDT、backbone dihedral、局部曲率、邻居密度/表面 proxy、SSE fallback、residue kNN 图和距离边特征。

训练时启用：

```bash
python train_unified.py \
  --task kcat_km \
  --variant hybrid \
  --test_fold 0 \
  --use_gnn \
  --use_protein3d \
  --protein3d_index data/protein3d/protein3d_index.csv \
  --protein3d_cache_dir data/protein3d/cache \
  --output_dir results/protein3d/kcat_km_fold0
```

第一版只做 residue/SSE/surface-proxy，不复刻 PRIME 五层 surface mesh。ColabFold 失败或缓存缺失时，样本回退到 ProtT5 token，不中断训练。

## 4. 命令是否可以直接运行

你给出的这类命令可以作为运行方式，但需要按当前 pipeline 调整：

1. `--gnn_lr_scale 0.1` 不推荐继续使用。当前 GNN 从头训练，默认已经改为 `1.0`，建议省略该参数或显式设为 `1.0`。
2. `--gnn_pooling mean` 可以保留但不会影响当前主模型结果，因为 graph-level pooling 已经不进入预测头。
3. 不要再传 `--gnn_pretrained_path`，即使传了也会被忽略。
4. 命令需要在项目根目录运行，并确保 `dataset/`、本地 ProtT5 / ProstT5 / MolT5 权重路径、PyTorch Geometric 依赖都可用。
5. 示例使用 `CUDA_VISIBLE_DEVICES=0,1` 和 `--nproc_per_node 2`，表示用 2 张 GPU。如果只有 1 张 GPU，请把它们改成 `CUDA_VISIBLE_DEVICES=0` 和 `--nproc_per_node 1`。

## 5. 基线训练（不开 GNN）

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 \
python run_table2.py \
  --tasks kcat_km \
  --variants hybrid \
  --launcher torchrun \
  --nproc_per_node 2 \
  --batch_size 8 \
  --grad_accum_steps -1 \
  --split_strategy modulo1 \
  --freeze_encoders \
  --mixed_precision \
  --output_root results/table2/kcat_km


CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 \
python run_table2.py \
  --tasks topt \
  --variants hybrid \
  --launcher torchrun \
  --nproc_per_node 2 \
  --batch_size 8 \
  --grad_accum_steps -1 \
  --split_strategy modulo1 \
  --freeze_encoders \
  --mixed_precision \
  --output_root results/table2/topt


CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 \
python run_table2.py \
  --tasks ph \
  --variants hybrid \
  --launcher torchrun \
  --nproc_per_node 2 \
  --batch_size 8 \
  --grad_accum_steps -1 \
  --split_strategy modulo1 \
  --freeze_encoders \
  --mixed_precision \
  --output_root results/table2/ph
```

## 6. 按当前 Pipeline 开启 TAGConv-GNN 训练

下面命令会完整走当前更新后的 pipeline：

```text
ProtT5 / ProstT5 蛋白 token
+ MolT5 底物 token
+ TAGConv-GNN atom token
→ 分子端 MolT5-GNN self-attention 融合
→ 酶-底物双向 cross-attention
→ attention pooling
→ MACCS 指纹投影
→ MLP 回归预测
```

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 \
python run_table2.py \
  --tasks kcat_km \
  --variants hybrid \
  --launcher torchrun \
  --nproc_per_node 2 \
  --batch_size 8 \
  --grad_accum_steps -1 \
  --split_strategy modulo1 \
  --freeze_encoders \
  --mixed_precision \
  --use_gnn \
  --gnn_hidden_dim 256 \
  --gnn_output_dim 256 \
  --gnn_layers 3 \
  --gnn_k_hops 3 \
  --gnn_max_atoms 128 \
  --gnn_lr_scale 1.0 \
  --output_root results/table2_gnn/kcat_km
```

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 \
python run_table2.py \
  --tasks ph \
  --variants hybrid \
  --launcher torchrun \
  --nproc_per_node 2 \
  --batch_size 8 \
  --grad_accum_steps -1 \
  --split_strategy modulo1 \
  --freeze_encoders \
  --mixed_precision \
  --use_gnn \
  --gnn_hidden_dim 256 \
  --gnn_output_dim 256 \
  --gnn_layers 3 \
  --gnn_k_hops 3 \
  --gnn_max_atoms 128 \
  --gnn_lr_scale 1.0 \
  --output_root results/table2_gnn/ph
```

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 \
python run_table2.py \
  --tasks topt \
  --variants hybrid \
  --launcher torchrun \
  --nproc_per_node 2 \
  --batch_size 8 \
  --grad_accum_steps -1 \
  --split_strategy modulo1 \
  --freeze_encoders \
  --mixed_precision \
  --use_gnn \
  --gnn_hidden_dim 256 \
  --gnn_output_dim 256 \
  --gnn_layers 3 \
  --gnn_k_hops 3 \
  --gnn_max_atoms 128 \
  --gnn_lr_scale 1.0 \
  --output_root results/table2_gnn/topt
```

可选参数：

- `--freeze_gnn`: 冻结 GNN 编码器参数，一般不建议在当前 pipeline 中使用，因为当前设计目标是让 GNN 从头参与训练。
- `--gnn_pooling`: 当前主模型不使用 graph-level pooling，因此该参数不会影响预测结果。
- `--gnn_pretrained_path`: 当前会被忽略，不再加载 GNN 预训练权重。

## 7. 一次性跑完整 GNN Pipeline 实验

如果希望完整按照更新后 pipeline 跑三个任务，可以使用：

```bash
CUDA_VISIBLE_DEVICES=1,2 \
OMP_NUM_THREADS=1 \
python run_table2.py \
  --tasks kcat_km ph topt \
  --variants hybrid \
  --launcher torchrun \
  --nproc_per_node 2 \
  --batch_size 8 \
  --grad_accum_steps -1 \
  --split_strategy modulo1 \
  --freeze_encoders \
  --mixed_precision \
  --use_gnn \
  --gnn_hidden_dim 256 \
  --gnn_output_dim 256 \
  --gnn_layers 3 \
  --gnn_k_hops 3 \
  --gnn_max_atoms 128 \
  --gnn_lr_scale 1.0 \
  --output_root results/0502/table2_gnn_current
```

如果想同时比较 ProtT5 和 ProstT5，可以跑：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 \
python run_table2.py \
  --tasks kcat_km ph topt \
  --variants hybrid hybrid_prostt5 \
  --launcher torchrun \
  --nproc_per_node 2 \
  --batch_size 8 \
  --grad_accum_steps -1 \
  --split_strategy modulo1 \
  --freeze_encoders \
  --mixed_precision \
  --use_gnn \
  --gnn_hidden_dim 256 \
  --gnn_output_dim 256 \
  --gnn_layers 3 \
  --gnn_k_hops 3 \
  --gnn_max_atoms 128 \
  --gnn_lr_scale 1.0 \
  --output_root results/table2_gnn_current_encoder_compare
```

不建议在当前版本中把 `hybrid_pp` 放入主要对比，因为 `physchem` 目前还没有接入预测头。

## 8. 批量跑不开 GNN 的 Table2 基线

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 \
python run_table2.py \
  --tasks kcat_km ph topt \
  --variants hybrid hybrid_prostt5 \
  --launcher torchrun \
  --nproc_per_node 2 \
  --batch_size 8 \
  --grad_accum_steps -1 \
  --split_strategy modulo1 \
  --freeze_encoders \
  --mixed_precision \
  --output_root results/table2
```

当前版本中 `hybrid_pp` 会计算 `physchem`，但模型没有使用它，因此不建议把 `hybrid_pp` 放入主要基线对比。

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 \
python run_table2.py \
  --tasks kcat_km \
  --variants hybrid \
  --launcher torchrun \
  --nproc_per_node 2 \
  --batch_size 8 \
  --grad_accum_steps -1 \
  --split_strategy modulo1 \
  --freeze_encoders \
  --mixed_precision \
  --output_root results/table2/kcat_km
```

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 \
python run_table2.py \
  --tasks ph \
  --variants hybrid \
  --launcher torchrun \
  --nproc_per_node 2 \
  --batch_size 8 \
  --grad_accum_steps -1 \
  --split_strategy modulo1 \
  --freeze_encoders \
  --mixed_precision \
  --output_root results/table2/ph
```

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 \
python run_table2.py \
  --tasks topt \
  --variants hybrid \
  --launcher torchrun \
  --nproc_per_node 2 \
  --batch_size 8 \
  --grad_accum_steps -1 \
  --split_strategy modulo1 \
  --freeze_encoders \
  --mixed_precision \
  --output_root results/table2/topt

```

输出文件：

- 每折指标：`results/table2/<task>/<variant>/fold_<k>/metrics.json`
- 汇总表：`results/table2/table2_summary.csv`

## 9. 与当前代码对齐的关键设置

- 任务级配置：`src/enzyme_unified/config.py`
  - `kcat_km`: 10 折，`log10` 标签变换，默认 `lr=1e-5`
  - `ph`: 5 折，默认 `lr=5e-4`
  - `topt`: 5 折，默认 `lr=1e-3`
- 模型：
  - 当前为单一路径：MolT5-GNN 分子端融合 + 酶-底物 Cross-Attention + Attention Pooling
  - 可选分子图路径：TAGConv atom token + MolT5 token self-attention 融合
  - 不再使用 Global Concat 和 Gate
- 指标：
  - `kcat_km` 训练在 `log10(y)` 空间；默认在 **log 空间**汇报指标（更贴近论文量级）
  - pH / `topt` 在原尺度汇报指标

## 10. 复现一致性建议

- 固定随机种子：`--seed`
- 固定划分策略：`--split_strategy`
- 固定预训练模型版本（`train_unified.py` 参数）
- 若启用 GNN，固定 `gnn_` 超参数与 `--use_gnn`
- 当前 GNN 从头训练，不使用 GNN 预训练权重
- 记录运行配置：每次运行会在输出目录写入 `run_config.json`

## 11. GNN 预训练脚本说明

仓库中仍保留 `pretrain_gnn.py`，但当前主训练 pipeline 不再加载 GNN 预训练权重。也就是说，即使运行下面脚本生成 checkpoint，`train_unified.py` 当前也不会把它加载进主模型。

该脚本仅作为历史代码或后续实验备用。

反应数据格式示例：

```json
[
  {
    "reaction_id": "RHEA:12477",
    "substrates": ["OC(=O)C(N)CC(=O)O", "O"],
    "products": ["OC(=O)C(=O)CC(=O)O", "[NH3]"]
  }
]
```

运行：

```bash
python pretrain_gnn.py \
  --data_path data/rhea_reactions.json \
  --save_path checkpoints/pretrained_gnn.pth \
  --epochs 50 \
  --batch_size 64 \
  --lr 1e-3 \
  --device cuda
```

## 12. 注意事项

- ProtT5 / ProstT5 / MolT5 体积较大，请保证显存与磁盘缓存空间。
- 启用 GNN 后，batch 内会额外构建图数据，建议先从较小 `batch_size` 与 `gnn_max_atoms` 起步。
- 如果结果与论文有偏差，优先检查：
  1. 使用的预训练模型版本是否一致；
  2. 变体设置（hybrid / hybrid_prostt5 / hybrid_pp）是否对应；
  3. 早停策略与 batch size 是否与论文设定一致；
  4. `--use_gnn` 与 `gnn_*` 参数是否和对照实验保持一致；
  5. 当前代码已经移除 Global Concat + Gate，因此不要再按旧 README 的双路径模型解释结果。

