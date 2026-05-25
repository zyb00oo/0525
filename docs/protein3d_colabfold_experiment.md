# 蛋白 3D 结构分支服务器实验落实文档

本文档说明如何从零配置 ColabFold，生成蛋白预测结构缓存，并在 `code_GNN` 中运行 `ProtT5 + protein3D` 和 `TAGConv + protein3D` 实验。

核心原则：

```text
数据集没有 3D 字段
→ 从 Sequence 导出 FASTA
→ 服务器用 ColabFold 预测结构
→ 从 PDB / pLDDT / PAE 构建 residue-level 结构缓存
→ 训练时读取缓存，不在训练 loop 里跑 ColabFold
```

## 1. 目录假设

以下命令默认在项目根目录运行：

```bash
cd /path/to/code_GNN
```

推荐服务器目录：

```text
code_GNN/
├── dataset/
├── scripts/
├── src/
├── data/
│   └── protein3d/
│       ├── unique_proteins.fasta
│       ├── protein_index.csv
│       ├── colabfold_out/
│       ├── cache/
│       └── protein3d_index.csv
└── results/
```

`data/protein3d/` 和 `results/` 可不提交 Git，服务器本地生成即可。

## 2. 环境准备

建议拆两个环境：

- `enzyme`：训练环境，装 PyTorch / PyG / transformers / RDKit / Biopython。
- `colabfold`：结构预测环境，装 ColabFold。

原因：ColabFold 依赖重，和训练环境混在一起容易冲突。

### 2.1 训练环境

示例：

```bash
conda create -n enzyme python=3.10 -y
conda activate enzyme

pip install -r requirements.txt
```

如果 PyG 安装失败，按服务器 CUDA / PyTorch 版本到 PyG 官方轮子源安装。必须保证：

```bash
python - <<'PY'
import torch
import torch_geometric
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("pyg ok")
PY
```

### 2.2 ColabFold 环境

推荐用 `mamba`：

```bash
mamba create -n colabfold -c conda-forge -c bioconda colabfold -y
conda activate colabfold
```

检查：

```bash
colabfold_batch --help
```

如果服务器已有 ColabFold module，直接用服务器版本即可。关键是能运行 `colabfold_batch input.fasta output_dir`。

## 3. 导出唯一蛋白序列

切回训练环境或普通 Python 环境：

```bash
conda activate enzyme
cd /path/to/code_GNN

python scripts/export_unique_proteins.py \
  --output_dir data/protein3d
```

输出：

```text
data/protein3d/unique_proteins.fasta
data/protein3d/protein_index.csv
```

ID 规则：

- `ph` / `topt` 有 `UniProtID`，优先用 `UniProtID`。
- `kcat_km` 没 `UniProtID`，用 `sha1(sequence)` 生成稳定 ID。
- `sha1` 只当文件名，不影响结构预测质量。ColabFold 真正使用的是 FASTA 里的氨基酸序列。

检查导出数量：

```bash
wc -l data/protein3d/protein_index.csv
grep -c "^>" data/protein3d/unique_proteins.fasta
```

## 4. 用 ColabFold 预测结构

切到 ColabFold 环境：

```bash
conda activate colabfold
cd /path/to/code_GNN

colabfold_batch \
  data/protein3d/unique_proteins.fasta \
  data/protein3d/colabfold_out
```

ColabFold 会输出：

```text
*.pdb
*_scores*.json
*_predicted_aligned_error*.json
*_PAE.png
*_plddt.png
```

本项目后续主要用：

- PDB：residue 坐标，尤其 CA。
- PDB B-factor / scores JSON：pLDDT。
- PAE JSON：residue pair 相对误差，可进入边特征。

### 4.1 批量任务建议

蛋白多时，不建议一次性全塞。可按 FASTA 切块提交 SLURM job array。

示例切块：

```bash
mkdir -p data/protein3d/fasta_chunks

python - <<'PY'
from pathlib import Path

inp = Path("data/protein3d/unique_proteins.fasta")
out_dir = Path("data/protein3d/fasta_chunks")
records = []
current = []
for line in inp.read_text().splitlines():
    if line.startswith(">") and current:
        records.append(current)
        current = []
    current.append(line)
if current:
    records.append(current)

chunk_size = 200
for i in range(0, len(records), chunk_size):
    chunk = records[i:i + chunk_size]
    path = out_dir / f"chunk_{i // chunk_size:04d}.fasta"
    path.write_text("\n".join("\n".join(r) for r in chunk) + "\n")
print("chunks", (len(records) + chunk_size - 1) // chunk_size)
PY
```

单 chunk 运行：

```bash
colabfold_batch \
  data/protein3d/fasta_chunks/chunk_0000.fasta \
  data/protein3d/colabfold_out
```

## 5. 生成 protein3D 缓存

切回训练环境：

```bash
conda activate enzyme
cd /path/to/code_GNN

python scripts/build_protein3d_cache.py \
  --protein_index data/protein3d/protein_index.csv \
  --colabfold_dir data/protein3d/colabfold_out \
  --cache_dir data/protein3d/cache \
  --k_neighbors 16 \
  --cutoff 20.0
```

输出：

```text
data/protein3d/cache/{protein_id}.npz
data/protein3d/protein3d_index.csv
```

每个 `.npz` 包含：

```text
residue_pos: CA 坐标，[L, 3]
residue_feat: AA one-hot + pLDDT + phi/psi + curvature + neighbor density + ASA proxy + SSE fallback
edge_index: residue kNN 图
edge_attr: distance RBF + sequence separation + PAE
sse_feat: helix/strand/coil fallback，目前默认 coil
asa_feat: 邻居密度推导的 surface proxy
plddt: 每 residue pLDDT
valid_length: residue 数
```

注意：当前第一版不是精确 surface mesh。surface 信息是轻量 proxy：

```text
CA 邻居少 → 更可能表面暴露
CA 邻居多 → 更可能埋藏
```

## 6. 缓存质量检查

检查成功率：

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("data/protein3d/protein3d_index.csv")
print(df["status"].value_counts(dropna=False))
print("ok ratio:", (df["status"] == "ok").mean())
print(df["num_residues"].describe())
PY
```

检查 pLDDT：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

df = pd.read_csv("data/protein3d/protein3d_index.csv")
rows = []
for _, row in df[df["status"] == "ok"].iterrows():
    arr = np.load(row["cache_path"])
    plddt = arr["plddt"]
    rows.append({
        "protein_id": row["protein_id"],
        "mean_plddt": float(plddt.mean()),
        "low50_ratio": float((plddt < 50).mean()),
        "num_residues": int(arr["valid_length"][0]),
    })
out = pd.DataFrame(rows)
print(out["mean_plddt"].describe())
print("mean_plddt < 70:", (out["mean_plddt"] < 70).mean())
out.to_csv("data/protein3d/protein3d_quality.csv", index=False)
PY
```

经验解释：

```text
mean pLDDT >= 70：结构分支通常可用
50 <= mean pLDDT < 70：谨慎，可保留但观察消融
mean pLDDT < 50：预测结构可信度低，可能拖累 3D 分支
```

## 7. 单折冒烟训练

先跑单 fold，确认能读缓存、能 forward、能保存 metrics。

### 7.1 只加蛋白 3D

```bash
CUDA_VISIBLE_DEVICES=0 \
python train_unified.py \
  --task kcat_km \
  --variant hybrid \
  --test_fold 0 \
  --split_strategy modulo1 \
  --batch_size 8 \
  --grad_accum_steps 32 \
  --freeze_encoders \
  --mixed_precision \
  --use_protein3d \
  --protein3d_index data/protein3d/protein3d_index.csv \
  --protein3d_cache_dir data/protein3d/cache \
  --protein3d_max_residues 1024 \
  --protein3d_encoder transformer \
  --output_dir results/debug/kcat_km_hybrid_protein3d_fold0
```

### 7.2 分子 GNN + 蛋白 3D

```bash
CUDA_VISIBLE_DEVICES=0 \
python train_unified.py \
  --task kcat_km \
  --variant hybrid \
  --test_fold 0 \
  --split_strategy modulo1 \
  --batch_size 8 \
  --grad_accum_steps 32 \
  --freeze_encoders \
  --mixed_precision \
  --use_gnn \
  --gnn_lr_scale 1.0 \
  --use_protein3d \
  --protein3d_index data/protein3d/protein3d_index.csv \
  --protein3d_cache_dir data/protein3d/cache \
  --protein3d_max_residues 1024 \
  --protein3d_encoder transformer \
  --output_dir results/debug/kcat_km_hybrid_gnn_protein3d_fold0
```

成功标志：

```text
results/debug/.../run_config.json
results/debug/.../best_model.pt
results/debug/.../metrics.json
```

若显存不够：

```text
batch_size 降到 4 或 2
protein3d_max_residues 降到 768 或 512
gnn_max_atoms 降到 96 或 64
grad_accum_steps 提高，保持有效 batch
```

## 8. Table2 批量实验

先按任务分开跑，方便失败重启。

### 8.1 baseline hybrid

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
  --output_root results/table2_baseline
```

### 8.2 当前分子 TAGConv

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
  --gnn_lr_scale 1.0 \
  --output_root results/table2_tagconv
```

### 8.3 只加蛋白 3D

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
  --use_protein3d \
  --protein3d_index data/protein3d/protein3d_index.csv \
  --protein3d_cache_dir data/protein3d/cache \
  --protein3d_max_residues 1024 \
  --protein3d_encoder transformer \
  --output_root results/table2_protein3d
```

### 8.4 分子 TAGConv + 蛋白 3D

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
  --gnn_lr_scale 1.0 \
  --use_protein3d \
  --protein3d_index data/protein3d/protein3d_index.csv \
  --protein3d_cache_dir data/protein3d/cache \
  --protein3d_max_residues 1024 \
  --protein3d_encoder transformer \
  --output_root results/table2_tagconv_protein3d
```

`ph` / `topt` 同理改 `--tasks ph` 或 `--tasks topt`。

## 9. 推荐消融顺序

不要直接全开。顺序：

1. `hybrid`：原始 ProtT5 + MolT5 + MACCS。
2. `hybrid_tagconv`：当前分子 TAGConv。
3. `hybrid_protein3d`：只加蛋白 3D。
4. `hybrid_tagconv_protein3d`：分子 GNN + 蛋白 3D。
5. `protein3d_encoder=gine`：替换 transformer，看图卷积是否更稳。
6. `protein3d_max_residues=512/768/1024`：看截断和显存。

主要比较：

```text
RMSE 越低越好
PCC / SCC 越高越好
kcat_km 默认看 log 空间指标
```

汇总文件：

```text
results/table2_*/table2_summary.csv
```

## 10. 常见失败与处理

### 10.1 ColabFold 没生成 PDB

看 `protein3d_index.csv`：

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/protein3d/protein3d_index.csv")
print(df[df["status"] != "ok"].head(20))
PY
```

处理：

- 单独重跑失败 FASTA。
- 保留 missing，训练会回退 ProtT5，不中断。

### 10.2 找不到缓存

确认路径：

```bash
ls data/protein3d/protein3d_index.csv
ls data/protein3d/cache | head
```

训练时必须传：

```bash
--protein3d_index data/protein3d/protein3d_index.csv
--protein3d_cache_dir data/protein3d/cache
```

### 10.3 显存爆

优先改：

```text
--batch_size 4
--protein3d_max_residues 512
--gnn_max_atoms 96
```

再提高：

```text
--grad_accum_steps
```

### 10.4 3D 分支没提升

先别判死刑。检查：

- `mean pLDDT` 是否低。
- 缺失结构比例是否高。
- 当前 split 是否主要测随机泛化，不一定体现结构收益。
- 是否 `protein3d_max_residues` 截掉活性区域。
- 是否蛋白真实活性依赖复合物、金属、辅因子，单体预测结构不足。

## 11. 结果记录模板

每轮实验记录：

```text
实验名:
任务:
fold:
命令:
代码 commit/hash:
protein3d_index:
ColabFold 版本:
mean pLDDT 分布:
batch_size:
grad_accum_steps:
GPU:
结果路径:
RMSE/PCC/SCC:
备注:
```

## 12. 最小可跑清单

服务器上按顺序执行：

```bash
# 1. 导出 FASTA
conda activate enzyme
cd /path/to/code_GNN
python scripts/export_unique_proteins.py --output_dir data/protein3d

# 2. ColabFold 预测
conda activate colabfold
cd /path/to/code_GNN
colabfold_batch data/protein3d/unique_proteins.fasta data/protein3d/colabfold_out

# 3. 构建结构缓存
conda activate enzyme
cd /path/to/code_GNN
python scripts/build_protein3d_cache.py \
  --protein_index data/protein3d/protein_index.csv \
  --colabfold_dir data/protein3d/colabfold_out \
  --cache_dir data/protein3d/cache

# 4. 单折冒烟
CUDA_VISIBLE_DEVICES=0 \
python train_unified.py \
  --task kcat_km \
  --variant hybrid \
  --test_fold 0 \
  --batch_size 8 \
  --grad_accum_steps 32 \
  --freeze_encoders \
  --mixed_precision \
  --use_protein3d \
  --protein3d_index data/protein3d/protein3d_index.csv \
  --protein3d_cache_dir data/protein3d/cache \
  --output_dir results/debug/kcat_km_hybrid_protein3d_fold0
```

冒烟通过后，再跑 Table2 批量实验。
