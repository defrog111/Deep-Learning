# Transformer 完整学习路线

这个目录按“一个知识点、一个可运行文件”组织。从基础张量开始，逐步手写
Attention、Encoder、Decoder，最后从 UCI 下载真实 CSV，用手写 Transformer
完成分类。

## 环境

```bash
cd transformer_learning
python -m pip install -r requirements.txt
```

建议使用 Python 3.10+ 和 PyTorch 2.1+。核心示例只依赖 PyTorch；画图额外依赖
Matplotlib。

## 推荐学习顺序

| 顺序 | 文件 | 必须理解的内容 |
|---:|---|---|
| 1 | `01_embeddings.py` | token id、Embedding、`(B,T,D)` shape |
| 2 | `02_positional_encoding.py` | 为什么 Attention 本身不知道顺序、sin/cos 位置编码 |
| 3 | `03_scaled_dot_product_attention.py` | `QKᵀ/√d`、softmax、对 V 加权 |
| 4 | `04_multi_head_attention.py` | split heads、并行关系、concat 与输出投影 |
| 5 | `05_residual_norm_ffn.py` | LayerNorm、残差、逐 token MLP |
| 6 | `06_encoder_block.py` | Self-Attention + FFN、Pre-LN、反向传播 |
| 7 | `07_attention_masks.py` | padding mask、causal mask、广播 |
| 8 | `08_decoder_block.py` | masked self-attention、cross-attention |
| 9 | `09_encoder_decoder_transformer.py` | 完整 Seq2Seq 数据流 |
| 10 | `10_tiny_causal_language_model.py` | GPT 风格 causal LM、next-token loss、生成 |
| 11 | `11_download_iris_csv.py` | 下载、解压、清洗并保存 CSV |
| 12 | `12_csv_transformer_classifier.py` | 特征 token、CLS、训练/验证/测试、防止泄漏 |
| 13 | `13_csv_classifier_inference.py` | checkpoint、标准化参数、概率输出 |
| 14 | `14_pytorch_builtin_comparison.py` | 手写代码与官方模块的对应关系 |
| 15 | `15_attention_visualization.py` | 查看 CLS 对各输入特征的注意力 |
| 16 | `16_torch_transformer_encoder.py` | 官方 `nn.TransformerEncoder`、padding mask、分类 |
| 17 | `17_torch_transformer_decoder.py` | 官方 `nn.TransformerDecoder`、causal mask、Cross-Attention |

核心实现集中在 `components.py`，不是对 `nn.Transformer` 的简单包装。建议先读每个
小例子，再回到 `components.py` 连起来阅读。

### 官方 Encoder/Decoder API 的关键区别

`nn.TransformerEncoder` 只接收 `src`，常用于理解整段输入；分类时通常对 Encoder
输出做 `[CLS]` pooling 或 masked mean pooling。

```python
encoded = encoder(
    src=encoder_input,
    src_key_padding_mask=source_padding_mask,
)
```

`nn.TransformerDecoder` 同时接收 `tgt` 和 Encoder 的 `memory`。`tgt_mask` 防止
目标 token 偷看未来；`memory_key_padding_mask` 防止 Cross-Attention 关注源序列
中的 padding。

```python
decoded = decoder(
    tgt=target_input,
    memory=encoder_output,
    tgt_mask=causal_mask,
    tgt_key_padding_mask=target_padding_mask,
    memory_key_padding_mask=source_padding_mask,
)
```

注意两类 mask 的语义：

- `src/tgt/memory_key_padding_mask` 的 shape 通常是 `(B, T)`，`True` 表示屏蔽。
- `tgt_mask` 的 shape 通常是 `(T, T)`，用于表达目标序列的因果关系。

## 一键运行核心例子和测试

```bash
python run_core_examples.py
python -m unittest discover -s tests -v
```

## 真实 CSV 分类完整流程

### 1. 下载数据

```bash
python 11_download_iris_csv.py
```

脚本从 UCI 官方仓库下载 Iris 压缩包，读取其中的 `iris.data`，增加 CSV 表头，
并生成 `data/iris.csv`。数据有 150 行、4 个连续特征、3 个类别，无缺失值。

数据来源：R. A. Fisher, *Iris*, UCI Machine Learning Repository,
<https://archive.ics.uci.edu/dataset/53/iris>，许可为 CC BY 4.0。

### 2. 训练、验证和测试

```bash
python 12_csv_transformer_classifier.py
```

模型不把整行直接送给 MLP，而是把每一个数值特征变成一个 token：

```text
CSV row: [sepal_length, sepal_width, petal_length, petal_width]
                         │
                         ▼
[CLS, sepal-token, sepal-token, petal-token, petal-token]
                         │
                         ▼
              2 × Transformer Encoder
                         │
                         ▼
                CLS representation → 3 classes
```

切分采用每类分层的 60%/20%/20%。均值和标准差只用训练集计算，避免测试数据泄漏。
验证集选择最佳 checkpoint，测试集只用于最终评估。模型和标准化参数保存在
`artifacts/iris_transformer.pt`。

### 3. 推理与可视化

```bash
python 13_csv_classifier_inference.py --features 5.1 3.5 1.4 0.2
python 15_attention_visualization.py
```

第二条命令生成 `artifacts/iris_attention.png`。注意力权重可以帮助检查模型，但
不能直接等同于严格的因果解释或特征重要性。

## 学透 Transformer 时要能回答

1. 为什么 logits 要除以 `sqrt(d_k)`？
2. 多头与单头的 shape 如何变化？
3. 没有位置编码会失去什么信息？
4. padding mask 和 causal mask 分别屏蔽什么？
5. Self-Attention 与 Cross-Attention 的 Q/K/V 来自哪里？
6. 残差、LayerNorm、FFN 分别解决什么问题？
7. Encoder-only、Decoder-only、Encoder-Decoder 分别适合什么任务？
8. 训练时 teacher forcing 与推理时自回归有什么差别？
9. 为什么训练集标准化统计不能使用验证集或测试集？
10. Attention 的时间和显存复杂度为什么通常是 `O(T²)`？

## 继续学习的方向

- BERT：Encoder-only、Masked Language Modeling。
- GPT：Decoder-only、Causal Language Modeling、KV cache。
- T5：Encoder-Decoder、text-to-text。
- ViT：把图像 patch 当 token。
- 长序列：FlashAttention、稀疏/线性 Attention。
- 工程训练：mixed precision、梯度累积、分布式训练、checkpoint 恢复。
- 微调：LoRA、SFT、DPO/PPO/GRPO；仓库根目录保留了相应示例。
