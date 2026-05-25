# 当前模型 Pipeline 说明

本文档记录当前代码版本的 pipeline，包括：相对于旧框架做了哪些修改、现在数据如何流动、模型前向如何计算、训练与评估如何执行，以及当前版本中需要注意的实验变量。

说明：这里以当前代码为准。旧版 `README.md` 中仍可能保留 `Cross-Attention + Global Concat + Gate` 的描述，但当前模型已经不再使用旧的全局拼接分支和 gate 融合。

## 1. 本次框架修改了什么

这次修改的核心目标是：去掉粗粒度全局预测分支，把模型主干集中到细粒度酶-底物交叉注意力上；同时让 GNN 从头训练，并增强 GNN 原子级表示和 MolT5 底物 token 的融合。

主要修改如下。

### 1.1 移除旧的全局预测分支

旧版本中，模型有两条预测路径：

```text
路径 1：cross-attention 分支
路径 2：global concat 分支
最后通过 alpha gate 融合两个预测结果
```

旧的 global concat 分支使用：

```text
protein_pool
substrate_pool
MACCS
GNN graph-level embedding
physchem
→ concat_head
→ y_concat
```

当前版本已经移除：

```text
concat_head
alpha gate
y_concat
protein_pool 到预测头的连接
substrate_pool 到预测头的连接
GNN graph-level embedding 到预测头的连接
physchem 到预测头的连接
```

当前最终预测只来自交叉注意力路径：

```text
y = y_attn
```

### 1.2 主模型不再使用 GNN graph-level pooling

`TAGConvGNNEncoder` 里仍然保留 `mean`、`max`、`mean_max` 这些图级 pooling 选项，方便兼容测试或后续实验。

但当前主模型调用 GNN 时使用：

```python
return_graph=False
```

也就是说，主训练路径只使用 GNN 的原子级输出：

```text
atom_embeds
```

不再使用：

```text
graph_embed
```

因此，当前主模型中 `--gnn_pooling` 参数不会影响最终预测结果。

### 1.3 GNN 不再加载预训练权重

旧版本支持通过：

```text
--gnn_pretrained_path
```

加载 `pretrain_gnn.py` 生成的 GNN 预训练权重。

当前版本中，这个参数保留用于兼容旧命令，但训练入口不会再加载权重，只会打印 ignored 信息。

当前 GNN 的训练方式是：

```text
随机初始化
→ 与主模型一起端到端训练
```

同时，`gnn_lr_scale` 默认值从 `0.1` 改为 `1.0`，避免随机初始化的 GNN 学习率过小。

### 1.4 增强 GNN 模块表达能力

当前 GNN 编码器不只是 TAGConv 堆叠，还在每层 TAGConv 后加入了 FFN 残差块。

当前 GNN 单层结构大致是：

```text
TAGConv
→ BatchNorm
→ ReLU
→ Dropout
→ residual
→ FFN
→ LayerNorm residual
```

最后输出前还增加了 `LayerNorm`。

这样做的目的是让 GNN 原子级表示更充分，再交给后续的 MolT5-GNN 融合模块。

### 1.5 增强 MolT5 token 与 GNN atom token 的融合

旧版本的分子端融合比较简单：

```text
MolT5 token 投影
GNN atom token 投影
→ 直接 concat
```

当前版本改为：

```text
MolT5 token 投影
GNN atom token 投影
加入 type embedding
→ concat
→ self-attention
→ FFN
→ fused substrate tokens
```

也就是说，底物端在进入酶-底物交叉注意力之前，MolT5 的 SMILES 语义表示和 GNN 的分子拓扑原子表示已经先做了一次内部交互。

### 1.6 Cross-attention 后的读出从 mean pooling 改为 attention pooling

旧版本在 cross-attention 之后使用：

```text
masked_mean
```

当前版本使用：

```text
MaskedAttentionPooling
```

也就是让模型学习每个 token 的重要性权重，然后加权求和。

注意，这里的 attention pooling 不是之前想去掉的“全局池化分支”。它是在细粒度 cross-attention 完成之后，把变长 token 序列读出成固定维度向量的必要步骤。

## 2. 当前总体 Pipeline

当前完整流程可以概括为：

```text
CSV 数据
→ Dataset / DataLoader
→ ProtT5 或 ProstT5 编码蛋白序列
→ MolT5 编码底物 SMILES
→ 可选 TAGConv-GNN 编码分子图原子表示
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

不再来自：

```text
protein_pool
substrate_pool
GNN graph-level pooling
physchem
concat_head
alpha gate
```

## 3. 入口脚本

项目主要有两个入口。

### 3.1 单任务单折训练：`train_unified.py`

`train_unified.py` 用于训练一个任务的一个 fold。

流程：

```text
解析命令行参数
→ 读取 task 配置
→ 读取 CSV
→ 划分 train / val / test
→ 构造 Dataset 和 DataLoader
→ 构造 FeatureEncoder
→ 构造 EnzymeUnifiedModel
→ 调用 train_and_evaluate_one_fold()
→ 保存 run_config.json、best_model.pt、metrics.json
```

### 3.2 批量实验：`run_table2.py`

`run_table2.py` 用于批量跑多个任务、多个变体、多个 fold。

流程：

```text
遍历 tasks
→ 遍历 variants
→ 遍历 folds
→ 拼出 train_unified.py 命令
→ subprocess.run()
→ 读取每个 fold 的 metrics.json
→ 汇总 table2_summary.csv
```

## 4. 任务与数据配置

任务配置位于：

```text
src/enzyme_unified/config.py
```

当前支持三个任务：

| task | CSV | 标签列 | folds | 标签处理 |
| --- | --- | --- | --- | --- |
| `kcat_km` | `dataset/kcat-km_processed_0.4_data_10fold.csv` | `kcat_km` | 10 | `log10` |
| `ph` | `dataset/ph_largeset_data_clustered_0.4_5fold.csv` | `pH` | 5 | 原始值 |
| `topt` | `dataset/topt_data_clustered_0.4_5fold.csv` | `temperature` | 5 | 原始值 |

输入 CSV 至少需要包含：

```text
Sequence
Smiles
fold
任务标签列
```

数据读取时会：

```text
读取 CSV
→ 检查必要列
→ 删除缺失值
→ 标签转数值
→ 如果是 log target，过滤非正标签
→ 根据 fold 构造 train / val / test
```

默认划分策略是 `modulo1`：

```text
test = 指定 test_fold
val = (test_fold + 1) % total_folds
train = 其他 fold
```

也支持 `random90_10`：

```text
test = 指定 test_fold
剩余数据随机 90% train / 10% val
```

## 5. Dataset 与 Batch

Dataset 逻辑位于：

```text
src/enzyme_unified/dataset.py
```

每条样本包含：

```python
{
    "sequence": str,
    "smiles": str,
    "label_raw": float,
}
```

如果开启 `--use_gnn`，还会额外包含：

```python
{
    "mol_graph": torch_geometric.data.Data,
}
```

`collate_samples()` 会把样本合并成 batch：

```python
{
    "sequence": List[str],
    "smiles": List[str],
    "label_raw": Tensor[B],
    "mol_graph_batch": Batch,  # 仅 use_gnn=True 时存在
}
```

注意：DataLoader 阶段没有直接生成 ProtT5、MolT5、MACCS 等深度特征。这些特征是在训练循环中动态编码的。

## 6. 特征编码

特征编码位于：

```text
src/enzyme_unified/features.py
```

核心类是：

```text
FeatureEncoder
```

训练循环中每个 batch 会调用：

```python
feats = feature_encoder.encode_batch(
    sequences=batch["sequence"],
    smiles_list=batch["smiles"],
    use_physchem=use_physchem,
    device=device,
)
```

输出包括：

```python
{
    "protein_token": Tensor[B, Lp, Dp],
    "protein_mask": Tensor[B, Lp],
    "protein_pool": Tensor[B, Dp],
    "substrate_token": Tensor[B, Ls, Ds],
    "substrate_mask": Tensor[B, Ls],
    "substrate_pool": Tensor[B, Ds],
    "maccs": Tensor[B, 167],
    "physchem": Tensor[B, 22],  # 仅 hybrid_pp 时生成
}
```

当前模型实际使用：

```text
protein_token
protein_mask
substrate_token
substrate_mask
maccs
mol_graph_batch  # 仅 use_gnn=True
```

当前模型暂时不使用：

```text
protein_pool
substrate_pool
physchem
```

因此，现在 `hybrid_pp` 虽然会计算 `physchem`，但模型没有把它接入预测头。也就是说，当前版本中 `hybrid_pp` 不能代表真正的“加入蛋白理化特征”消融。

## 7. 蛋白特征

蛋白端默认使用 ProtT5：

```text
/mnt/data/oyangcan/prot_t5_xl_uniref50
```

如果 variant 是 `hybrid_prostt5`，则使用 ProstT5：

```text
/mnt/data/oyangcan/ProstT5
```

蛋白序列预处理：

```text
转大写
→ U/Z/O/B 替换为 X
→ 氨基酸之间加空格
→ 输入 ProtT5 / ProstT5
```

输出：

```text
protein_token
protein_mask
protein_pool
```

其中 `protein_pool` 是 encoder token 的 mask mean，但当前模型不使用它。

## 8. 底物文本特征

底物端使用 MolT5：

```text
/mnt/data/oyangcan/molt5-base-smiles2caption
```

输入是 SMILES 字符串。

输出：

```text
substrate_token
substrate_mask
substrate_pool
```

其中 `substrate_pool` 是 MolT5 token 的 mask mean，但当前模型不使用它。

## 9. MACCS 与 PhysChem

### 9.1 MACCS

MACCS 由 RDKit 生成，维度为 167。

当前 MACCS 会进入最终预测头：

```text
maccs
→ Linear(167, hidden_dim)
→ maccs_proj
```

最终与 `e_pool` 和 `s_pool` 拼接。

### 9.2 PhysChem

`physchem` 是 22 维蛋白理化特征：

```text
20 个氨基酸组成比例
+ 分子量
+ 等电点
```

当前 `hybrid_pp` 会计算它，但主模型不使用它。

如果后续要把 `physchem` 加回来，建议接入当前 attention 分支，而不是恢复旧的全局 concat 分支。例如：

```text
physchem
→ Linear(22, hidden_dim)
→ physchem_proj
concat[e_pool, s_pool, maccs_proj, physchem_proj]
→ MLP head
```

这样可以在保持细粒度 cross-attention 主干的同时，加入蛋白整体理化信息。

## 10. 分子图构建

分子图工具位于：

```text
src/enzyme_unified/mol_graph_utils.py
```

开启 `--use_gnn` 时，每个 SMILES 会通过 RDKit 转为 PyG 图。

节点特征维度为 37，包括：

```text
原子类型
degree
formal charge
chirality
氢原子数
hybridization
是否 aromatic
是否 ring
```

每条化学键会被转为双向边：

```text
i → j
j → i
```

代码也构造了边特征，包括：

```text
键类型
是否共轭
是否在环中
立体信息
```

但当前 `TAGConvGNNEncoder` 没有使用 `edge_attr`，只使用：

```text
x
edge_index
batch
```

如果 SMILES 解析失败，会返回一个单节点空图，避免数据加载过程直接中断。

## 11. GNN 编码器

GNN 位于：

```text
src/enzyme_unified/gnn_encoder.py
```

核心类：

```text
TAGConvGNNEncoder
```

当前结构：

```text
atom feature
→ input_proj
→ TAGConv
→ BatchNorm
→ ReLU
→ Dropout
→ residual
→ FFN
→ LayerNorm residual
→ 重复 n_layers 次
→ output_proj
→ output_norm
→ atom_embeds
```

主模型中调用方式：

```python
atom_embeds, _ = self.gnn_encoder(
    x=graph_batch.x,
    edge_index=graph_batch.edge_index,
    batch=graph_batch.batch,
    return_graph=False,
)
```

输出的 `atom_embeds` 形状为：

```text
[batch 中所有原子数之和, gnn_output_dim]
```

然后通过：

```text
pad_atom_embeddings()
```

变成：

```text
atom_padded: [B, max_atoms, gnn_output_dim]
atom_mask:   [B, max_atoms]
```

`max_atoms` 由 `--gnn_max_atoms` 控制，默认 128。超过该长度的原子会被截断。

## 12. 分子端 MolT5-GNN 融合

分子端融合位于：

```text
src/enzyme_unified/substrate_multiview.py
```

核心类：

```text
SubstrateMultiViewFuser
```

输入：

```text
MolT5 token: [B, Ls, hidden_dim]
MolT5 mask:  [B, Ls]
GNN atom:    [B, La, gnn_output_dim]
GNN mask:    [B, La]
```

处理流程：

```text
MolT5 token → projection / LayerNorm
GNN atom token → Linear 到 hidden_dim / LayerNorm
加入 type embedding 区分两类 token
concat 到序列维度
→ self-attention
→ residual + LayerNorm
→ FFN
→ residual + LayerNorm
→ padding 位置置零
```

输出：

```text
fused substrate tokens: [B, Ls + La, hidden_dim]
fused substrate mask:   [B, Ls + La]
```

这样底物端表示同时包含：

```text
MolT5 的 SMILES 语义信息
+ GNN 的分子拓扑原子信息
```

并且二者在进入酶-底物交叉注意力之前已经完成一次内部融合。

## 13. 酶-底物双向 Cross-Attention

交叉注意力位于：

```text
src/enzyme_unified/model.py
```

核心类：

```text
BiCrossAttentionBlock
```

输入：

```text
e: 蛋白 token 表示
s: 底物 token 表示
```

每层执行：

```text
e ← MultiheadAttention(query=e, key=s, value=s)
s ← MultiheadAttention(query=s, key=e, value=e)
```

也就是：

```text
蛋白 token 根据底物 token 更新
底物 token 根据蛋白 token 更新
```

随后各自经过：

```text
residual
LayerNorm
FFN
LayerNorm
```

层数由 `--cross_layers` 控制，默认 1。

## 14. Attention Pooling 读出

经过 cross-attention 后，蛋白和底物仍然是 token 序列：

```text
e: [B, Lp, hidden_dim]
s: [B, Ls 或 Ls + La, hidden_dim]
```

最终回归头需要固定维度向量，因此需要读出。

当前使用：

```text
MaskedAttentionPooling
```

计算流程：

```text
token sequence
→ Linear
→ Tanh
→ Linear
→ 每个 token 的 score
→ mask 掉 padding token
→ softmax 得到 token 权重
→ 加权求和
→ pooled vector
```

输出：

```text
e_pool: [B, hidden_dim]
s_pool: [B, hidden_dim]
```

这里的 pooling 不是旧版的粗粒度全局分支。它发生在细粒度 cross-attention 之后，是把变长 token 序列变为固定维度向量的读出层。

## 15. 最终预测头

当前最终预测头输入：

```text
e_pool
s_pool
maccs_proj
```

其中：

```text
maccs_proj = Linear(167, hidden_dim)(maccs)
```

最终：

```text
concat[e_pool, s_pool, maccs_proj]
→ Linear(hidden_dim * 3, hidden_dim)
→ GELU
→ Dropout
→ Linear(hidden_dim, 1)
→ squeeze
→ prediction
```

当前没有：

```text
concat_head
alpha gate
y_concat
```

## 16. 当前模型 Forward 总结

当前 `EnzymeUnifiedModel.forward()` 可以概括为：

```text
protein_token → Linear → e
substrate_token → Linear → s

如果 use_protein3d:
    protein3d_batch → Protein3DEncoder → residue_embeds
    residue_embeds → pad_residue_embeddings → residue_padded + residue_mask
    ProtT5 tokens + protein3D residue tokens → ProteinMultiViewFuser → 新的 e

如果 use_gnn:
    mol_graph_batch → TAGConvGNNEncoder → atom_embeds
    atom_embeds → pad_atom_embeddings → atom_padded + atom_mask
    MolT5 tokens + GNN atom tokens → SubstrateMultiViewFuser → 新的 s

e, s → BiCrossAttentionBlock × N
→ MaskedAttentionPooling 得到 e_pool, s_pool
→ MACCS Linear 得到 maccs_proj
→ concat[e_pool, s_pool, maccs_proj]
→ MLP head
→ 预测值
```

也就是说，当前最终预测只来自：

```text
交叉注意力后的蛋白表示
+
交叉注意力后的底物表示
+
MACCS 指纹投影
```

## 17. 训练流程

训练逻辑位于：

```text
src/enzyme_unified/trainer.py
```

每个 fold 的训练流程：

```text
读取数据
→ 构造 train / val / test
→ 构造 DataLoader
→ 构造 FeatureEncoder
→ 构造 EnzymeUnifiedModel
→ 每个 epoch 训练
→ 每个 epoch 验证
→ 按 val_rmse 保存 best_model.pt
→ patience 次不提升则 early stopping
→ 加载 best_model.pt
→ 在 test set 上评估
→ 保存 metrics.json
```

损失函数：

```text
MSE(pred_trans, labels_trans)
```

标签处理：

```text
kcat_km: log10(label)
ph: 原始值
topt: 原始值
```

评估指标：

```text
RMSE
PCC
SCC
```

对于 `kcat_km`，默认主指标在 log 空间计算。如果使用：

```text
--eval_raw_for_log_target
```

则主指标改为在原始尺度计算。

优化器：

```text
AdamW
```

参数分组：

```text
gnn_encoder + substrate_fuser: lr * gnn_lr_scale
其他参数: lr
```

当前默认：

```text
gnn_lr_scale = 1.0
```

支持：

```text
--mixed_precision
--grad_accum_steps
torchrun / DDP
```

## 18. 输出文件

单 fold 训练输出目录包含：

```text
run_config.json
best_model.pt
metrics.json
```

`run_config.json` 保存运行参数。

`best_model.pt` 保存验证集 RMSE 最优模型。

`metrics.json` 保存：

```text
best_epoch
best_val_rmse
test_metrics
history
```

使用 `run_table2.py` 批量运行时，还会生成：

```text
table2_summary.csv
```

其中包含每个 task / variant 的跨 fold 均值和标准差。

## 19. 当前版本需要注意的问题

### 19.1 `physchem` 暂时没有进入预测头

当前 `hybrid_pp` 会计算 `physchem`，但模型没有使用它。

因此当前版本中：

```text
hybrid_pp 并不等价于真正加入理化特征的模型
```

如果后续要验证 `physchem` 是否有帮助，建议把它投影后接入当前 attention head，而不是恢复旧的 global concat 分支。

### 19.2 `gnn_pooling` 当前不影响主模型

因为主模型调用 GNN 时设置：

```text
return_graph=False
```

所以 `gnn_pooling` 不参与最终预测。

### 19.3 GNN 预训练已停用

`--gnn_pretrained_path` 当前不会加载权重。

GNN 当前从头训练，并通过原子级 token 与 MolT5 token 融合。

### 19.4 `protein_pool` 和 `substrate_pool` 仍会被计算

`FeatureEncoder` 仍输出：

```text
protein_pool
substrate_pool
```

但当前模型不使用它们。保留它们主要是为了兼容现有接口和方便后续消融实验。

## 20. 可选蛋白 3D 分支

数据集中没有蛋白 3D 字段，当前结构分支采用离线缓存：

```text
CSV unique Sequence
→ scripts/export_unique_proteins.py
→ unique_proteins.fasta
→ ColabFold batch
→ PDB / pLDDT / PAE
→ scripts/build_protein3d_cache.py
→ protein3d_index.csv + cache/*.npz
```

缓存特征包括：

```text
residue_pos: CA coordinates
residue_feat: amino acid one-hot + pLDDT + phi/psi + curvature + surface proxy + SSE fallback
edge_index: residue kNN graph
edge_attr: distance RBF + sequence separation + optional PAE
```

训练入口新增：

```bash
--use_protein3d
--protein3d_index data/protein3d/protein3d_index.csv
--protein3d_cache_dir data/protein3d/cache
--protein3d_max_residues 1024
--protein3d_encoder transformer
```

模型路径：

```text
src/enzyme_unified/protein3d_encoder.py
src/enzyme_unified/protein_multiview.py
src/enzyme_unified/protein3d_utils.py
```

当前实现是 PRIME 启发的最小版：保留 residue geometry、SSE/surface proxy 和结构图，不复刻完整 surface/atom/residue/SSE/protein 五层层级。

## 21. 当前版本一句话总结

当前模型已经从旧的：

```text
cross-attention 分支
+
global concat 分支
+
alpha gate 融合
```

改为：

```text
以 token 级酶-底物 cross-attention 为核心，
用 GNN 原子级表示增强底物 token，
用 attention pooling 从交互后的 token 序列中读出关键表示，
再结合 MACCS 指纹进行回归预测。
```

这个版本更符合当前设计目标：

```text
去掉粗粒度全局拼接
保留细粒度交叉注意力
GNN 不预训练、从头训练
加强 GNN 与 MolT5 分子端融合
最终通过 attention 读出交互后的 token 表示
```
