# Hybrid 方法改进落地方案

本文档整理当前项目中 `hybrid` 方法的后续改进路线。当前阶段只聚焦 `hybrid` 主线，不把 `hybrid++` 和 `hybrid_prostT5` 纳入主要实验。

## 1. 当前目标

当前项目已经在 Enzyme-Unified 基础上加入 TAGConv 分子图分支，并把 GNN 原子级表示与 MolT5 token 融合后送入酶-底物 cross-attention。

下一步目标不是扩展新 variant，而是把 `hybrid` 做成更强的 enzyme-substrate interaction model：

```text
ProtT5 residue token
+ MolT5 substrate token
+ edge-aware GNN atom token
→ MolT5-token / atom-token 显式对齐
→ residue-token / atom-token 级交互增强
→ atom/residue importance 可解释读出
→ cliff-aware / hard-pair 训练
→ 回归预测
```

核心优化方向：

1. 让底物 GNN 使用化学键特征，而不是只使用原子特征和 `edge_index`。
2. 改造 MolT5 token 与 GNN atom token 的融合方式，避免简单 concat self-attention。
3. 显式建模 protein residue 与 substrate atom 之间的细粒度交互。
4. 引入 substrate cliff / hard pair 训练，使模型更敏感于结构相似但活性差异大的样本。
5. 导出 atom/residue attention，增强模型解释能力。

## 2. 论文启发

### 2.1 TransMA: 多模态分子结构融合与 cliff 样本

`essay/bbaf307.pdf` 提出的 TransMA 虽然研究对象是 LNP 分子性质预测，但对当前底物建模很有启发。

关键启发：

- 单一模态分子表示不够，1D 分子序列表示和细粒度原子/结构表示应该显式对齐。
- mol-attention 不仅提升预测效果，还能输出 atom-level importance。
- cliff 样本很重要：结构相似但标签差异大的样本能检验模型是否真正理解局部结构变化。

对应到本项目：

- MolT5 token 可视为 1D/语义分子视图。
- GNN atom token 可视为细粒度拓扑分子视图。
- 当前 `SubstrateMultiViewFuser` 只是 concat 后 self-attention，后续应增加 MolT5-token 与 atom-token 的显式 cross-attention 或 gate。
- 可在 `kcat_km` 上构建 substrate cliff pairs，增加 ranking/margin loss 和 cliff 专项指标。

### 2.2 MEI: 反应约束分子表示与酶-底物交互模块

`essay/deep-learning-driven-insights-into-enzyme-substrate-interaction-discovery.pdf` 的 MEI 模型强调：

- 分子 GNN 表示可以通过 substrate-product 反应约束进行预训练。
- 酶和底物不应只做全局拼接，而应通过 interaction module 建模。
- 预训练后再迁移到下游小数据任务，有助于提升泛化。

对应到本项目：

- 当前 `pretrain_gnn.py` 已经有 reaction-aware GNN 预训练雏形，但主训练暂时忽略 `--gnn_pretrained_path`。
- 由于当前 CSV 中没有 product / reaction equation，reaction-aware 预训练不应作为第一阶段主线。
- 后续若能准备 Rhea/BRENDA 反应数据，可恢复 GNN 预训练加载，比较 random init 与 reaction-pretrained GNN。

### 2.3 EZSpecificity: residue-atom cross-attention 与 unknown split

`essay/s41586-025-09697-2.pdf` 的 EZSpecificity 最直接相关。

关键启发：

- 酶底物特异性主要来自 active site 中 residue 与 substrate atom 的细粒度相互作用。
- cross-attention 能让模型聚焦关键 residue/atom，而不是平均池化所有 token。
- unknown substrate、unknown enzyme、unknown enzyme and substrate split 比随机划分更能检验泛化。
- 结构、图和 cross-attention 的消融都能带来可解释的性能变化。

对应到本项目：

- 当前已有 protein token 和 atom token，但 atom token 先与 MolT5 token 融合，后续 cross-attention 不容易区分真实 atom 与 SMILES token。
- 后续可增加 residue-token ↔ atom-token 的显式交互模块。
- 暂不建议立即做 docking / SE(3) / active-site 3D complex，因为工程量大、依赖额外结构和活性位点数据。
- 可先基于 2D molecular graph 和 protein token 做轻量级 residue-atom interaction。

## 3. 阶段 0: 固定当前基线

在修改模型前，应先保存当前 `hybrid` 的基线表现。

### 3.1 实验设置

只使用 `variant=hybrid`。

实验：

```text
hybrid_base:
  不启用 --use_gnn

hybrid_tagconv_current:
  启用 --use_gnn
  使用当前 TAGConv + SubstrateMultiViewFuser
```

### 3.2 输出目录建议

```text
results/hybrid_baseline/base
results/hybrid_baseline/tagconv_current
```

### 3.3 目的

确认当前 GNN 是否带来提升。如果当前 TAGConv 效果不稳定，后续 edge-aware GNN 和融合模块的贡献会更容易解释。

## 4. 阶段 1: Edge-aware GNN

### 4.1 当前问题

`mol_graph_utils.py` 已经构造了 `edge_attr`：

```text
bond type
conjugation
ring
stereo
```

但当前 `TAGConvGNNEncoder` 只使用：

```text
x
edge_index
batch
```

没有使用化学键特征。

### 4.2 修改目标

新增一个 edge-aware GNN，与当前 TAGConv 并存。

### 4.3 新增参数

```bash
--gnn_type tagconv
--gnn_type edge_gnn
```

默认值保持 `tagconv`，保证旧实验可复现。

### 4.4 代码改动位置

```text
src/enzyme_unified/gnn_encoder.py
src/enzyme_unified/model.py
train_unified.py
run_table2.py
tests/test_gnn_integration.py
```

### 4.5 实现建议

优先实现 `GINEConv` 版本：

```text
atom x
→ input_proj
→ edge_attr_proj
→ GINEConv × n
→ BatchNorm / LayerNorm
→ FFN residual
→ output_proj
→ atom_embeds
```

要求：

- 支持有边分子。
- 支持无边分子。
- 输出接口与 `TAGConvGNNEncoder` 保持一致：

```python
atom_embeds, graph_embed = encoder(...)
```

### 4.6 实验对照

```text
hybrid_tagconv_current
vs
hybrid_edge_gnn
```

### 4.7 验证标准

- 单测通过。
- `kcat_km fold0` smoke run 无 NaN。
- 全 fold 汇总中 RMSE/PCC/SCC 至少一项稳定提升。

## 5. 阶段 2: MolT5-GNN 显式融合

### 5.1 当前问题

当前融合方式：

```text
MolT5 tokens + GNN atom tokens
→ concat
→ self-attention
```

这没有显式解决 MolT5 token 与 atom token 的对齐问题。

### 5.2 修改目标

给 `SubstrateMultiViewFuser` 增加多种融合模式。

### 5.3 新增参数

```bash
--substrate_fuse_mode concat_self_attn
--substrate_fuse_mode cross_attn
--substrate_fuse_mode atom_gate
```

默认值保持 `concat_self_attn`。

### 5.4 模式设计

#### concat_self_attn

保留当前逻辑：

```text
MolT5 token + atom token
→ concat
→ self-attention
```

#### cross_attn

新增双向对齐：

```text
MolT5 token attends to atom token
atom token attends to MolT5 token
→ residual + LayerNorm
→ concat / merge
```

#### atom_gate

在 atom token 上增加显式重要性权重：

```text
atom_token → MLP → sigmoid score
atom_token = atom_token * score
```

该 score 后续可用于解释。

### 5.5 代码改动位置

```text
src/enzyme_unified/substrate_multiview.py
src/enzyme_unified/model.py
train_unified.py
run_table2.py
tests/test_gnn_integration.py
```

### 5.6 实验对照

```text
hybrid_edge_gnn
vs
hybrid_edge_gnn_cross_fuse
vs
hybrid_edge_gnn_atom_gate
```

## 6. 阶段 3: Residue-Atom 细粒度交互

### 6.1 当前问题

当前 cross-attention 输入是：

```text
protein tokens ↔ fused substrate tokens
```

其中 `fused substrate tokens` 混合了 MolT5 token 和 atom token，不容易区分真实原子级交互。

### 6.2 修改目标

新增可选的 residue-token ↔ atom-token interaction block。

### 6.3 新增参数

```bash
--use_atom_residue_interaction
```

### 6.4 结构建议

当 `use_gnn=True` 时保留三类底物表示：

```text
s_molt5
s_atom
s_fused
```

新增交互：

```text
protein residue token attends to atom token
atom token attends to protein residue token
→ atom-aware protein
→ residue-aware atom
→ pooling
→ prediction head
```

### 6.5 实验对照

```text
hybrid_edge_gnn_atom_gate
vs
hybrid_edge_gnn_atom_gate_residue_atom_attn
```

### 6.6 注意事项

这个阶段会改动模型主干，建议在阶段 1 和阶段 2 有稳定收益后再做。

## 7. 阶段 4: Cliff-aware 训练

### 7.1 当前问题

普通 MSE 对结构相似但标签差异大的样本不够敏感。

### 7.2 Cliff pair 定义

在训练集中构建样本对 `(i, j)`：

```text
substrate_similarity(i, j) >= 0.85 或 0.90
abs(log10(y_i) - log10(y_j)) >= 1.0
```

满足条件则认为是 substrate cliff pair。

### 7.3 相似度

优先使用 Morgan fingerprint Tanimoto。

也可以作为消融比较：

```text
Morgan Tanimoto
MACCS Tanimoto
```

### 7.4 新增参数

```bash
--use_cliff_loss
--cliff_similarity_threshold 0.9
--cliff_label_delta 1.0
--cliff_loss_weight 0.1
--cliff_margin 0.2
```

### 7.5 Loss 形式

主损失保持 MSE：

```text
loss = mse_loss + lambda * cliff_loss
```

cliff loss 使用 pairwise ranking：

```text
max(0, margin - sign(y_i - y_j) * (pred_i - pred_j))
```

### 7.6 第一版实现建议

先做 batch-level cliff loss，不做全训练集 pair 预计算。

原因：

- 实现更简单。
- 显存和时间可控。
- 更适合快速验证是否有效。

### 7.7 代码改动位置

```text
src/enzyme_unified/trainer.py
src/enzyme_unified/features.py 或新增 src/enzyme_unified/cliff_utils.py
train_unified.py
run_table2.py
```

### 7.8 新增指标

```text
cliff_pair_acc
cliff_pair_rmse
```

这些指标能直接说明模型是否更会处理结构相似但活性差异大的底物。

## 8. 阶段 5: Attention 与 Atom Importance 导出

### 8.1 目的

导出 attention 不是为了替代性能指标，而是用于：

- 检查模型是否聚焦少数关键 residue / atom。
- 支持论文中的可解释性分析。
- 辅助诊断模型失败案例。

### 8.2 新增参数

```bash
--save_attention
--attention_output_dir results/attention/...
```

### 8.3 输出格式

建议使用 JSONL，每条样本一行：

```json
{
  "sequence_id": "...",
  "smiles": "...",
  "label": 1.23,
  "pred": 1.18,
  "top_residues": [12, 119, 117],
  "top_atoms": [3, 5, 7],
  "atom_scores": [],
  "residue_scores": []
}
```

### 8.4 代码改动位置

```text
src/enzyme_unified/model.py
src/enzyme_unified/trainer.py
```

## 9. 阶段 6: Reaction-aware GNN 预训练

这个阶段不放在第一轮主线，但作为后续增强保留。

### 9.1 前提

需要额外准备 reaction 数据，例如：

```text
substrates
products
reaction_id
```

当前主 CSV 没有 product / reaction equation，因此不能直接做 reaction-aware 训练。

### 9.2 可复用代码

项目已有：

```text
pretrain_gnn.py
```

但当前主训练入口会忽略 `--gnn_pretrained_path`。

### 9.3 后续实现

后续可恢复：

```bash
--gnn_pretrained_path checkpoints/pretrained_gnn.pth
```

比较：

```text
random init GNN
reaction-pretrained GNN
reaction-pretrained + freeze
reaction-pretrained + finetune
```

## 10. 推荐实验矩阵

当前阶段只保留：

```bash
--variant hybrid
```

### 10.1 结构消融

```text
A0 hybrid_base
A1 hybrid_tagconv_current
A2 hybrid_edge_gnn
A3 hybrid_edge_gnn_cross_fuse
A4 hybrid_edge_gnn_atom_gate
A5 hybrid_edge_gnn_atom_gate_residue_atom_attn
```

### 10.2 训练目标消融

```text
B0 best_from_A
B1 best_from_A + cliff_loss
B2 best_from_A + cliff_loss + attention_export
```

### 10.3 泛化压力评估

如果时间允许，可以增加：

```text
unknown substrate split
unknown enzyme-like split
substrate scaffold split
```

但第一轮必须保留当前 fold 设置，以便和原始 Enzyme-Unified 结果对齐。

## 11. 推荐开发顺序

后续代码修改建议按以下顺序执行：

1. Edge-aware GNN
   - 风险低，收益明确。
   - 先把已经存在的 `edge_attr` 用起来。

2. Substrate fuser 改造
   - 增加 `cross_attn` 和 `atom_gate`。
   - 让 MolT5/GNN 融合更有理论支撑。

3. Attention / atom score 导出
   - 方便调试，也方便后续论文解释。

4. Cliff loss
   - 训练逻辑稍复杂，等模型结构稳定后再加。

5. Residue-atom explicit interaction
   - 如果前面提升明显，再做这个增强。
   - 否则容易改动过大，不好归因。

6. Reaction-aware GNN pretraining
   - 作为后续扩展，不放进第一轮。

## 12. 成功标准

一个改动能留下来，至少满足以下条件之一：

- `kcat_km` 全 fold 平均 RMSE 下降。
- PCC/SCC 提升，尤其 SCC 提升。
- cliff pair 排序准确率提升。
- attention / atom score 可解释性明显变好。
- 对 unknown substrate / scaffold split 更稳。

如果某个模块只提升单折、不提升全 fold，不应写成核心贡献，只作为探索结果。

## 13. 论文叙事

可以将当前阶段写成：

```text
在 Enzyme-Unified Hybrid 框架基础上，我们聚焦底物结构与酶-底物细粒度交互建模。
受多模态分子表示、reaction-aware molecule learning 和 enzyme-specificity cross-attention 启发，
我们引入 edge-aware molecular GNN、MolT5-atom 双向融合、atom-level importance gate
以及 cliff-aware 训练目标，从而增强模型对底物局部结构变化和 enzyme-substrate interaction 的表达能力。
```

该路线与三篇参考论文强相关，同时不偏离当前“先做好 Hybrid”的阶段目标。
