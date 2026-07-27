# 改进计划 v3：从 BLEU 26.6 冲击 BLEU 30+

> 基线：BLEU 24.57 → 经过两轮优化 → BLEU 26.6
> 目标：BLEU 30-33
> 创建日期：2026-06-09

## 诊断总结

| 问题 | 证据 | 严重程度 |
|------|------|----------|
| UNK 太多 | 37.3% 测试句含 UNK token | ★★★ 最严重 |
| 过拟合 | train_loss=0.82 vs val_loss=2.30，gap=1.48 | ★★☆ |
| 数据不够多样 | 仅 18,000 训练对，句子重复度高 | ★★☆ |
| 解码重复 | 部分预测出现连续重复 token | ★☆☆ 已缓解 |

## 三项改进

### 改进 1：字符级中文分词（消除 UNK 瓶颈）

**文件：** `cmn-eng-simple/preprocess/tokenizer.py`

当前用 jieba 做词级分词，低频词被过滤为 UNK。改为逐字符分词后，
常用汉字约 3,000 个，几乎全部频率 >= 3，UNK 率接近 0。

**当前代码（第 21-28 行）：**

```python
        cn_sentence = ''
        for word in list(jieba.cut(sentence[1])):
            word = re.sub(r'[ \n\t\r]', '', word)
            if word == '':
                continue
            cn_sentence += cc.convert(word) + ' '
        cn.append(cn_sentence)
```

**改为：**

```python
        cn_sentence = ''
        for char in sentence[1].strip():
            if char in (' ', '\n', '\t', '\r'):
                continue
            cn_sentence += cc.convert(char) + ' '
        cn.append(cn_sentence)
```

**效果示例：**
- 词级："她 对 绘画 有 很好 的 鉴赏力 。" → "绘画"、"鉴赏力" 可能是 UNK
- 字符级："她 對 繪 畫 有 很 好 的 鑒 賞 力 。" → 每个字都高频，无 UNK

**附带修改：** `build_dataset.py` 第 37 行频率阈值可从 `count >= 3` 降为 `count >= 2`。

**预期提升：** +3~5 BLEU

---

### 改进 2：R-Drop 正则化（对抗过拟合）

**文件：** `train_transformer.py` 的 `train_one_epoch` 函数

**原理：** 同一 batch 过两次 forward（dropout mask 不同），除了正常 CE loss，
额外加两次输出之间的 KL 散度 loss，迫使模型对 dropout 噪声保持一致。
这比单纯增大 dropout 更有效，不会削弱模型容量。

**当前核心代码（第 427-435 行）：**

```python
        with amp_context:
            logits = model(src=src, tgt_in=tgt_in, ...)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
```

**改为：**

```python
        with amp_context:
            logits1 = model(src=src, tgt_in=tgt_in, ...)
            logits2 = model(src=src, tgt_in=tgt_in, ...)  # 第二次 forward

            ce_loss = 0.5 * (
                criterion(logits1.reshape(-1, V), tgt_out.reshape(-1))
                + criterion(logits2.reshape(-1, V), tgt_out.reshape(-1))
            )

            # 双向 KL 散度（只算非 PAD 位置）
            p = F.log_softmax(logits1, dim=-1)
            q = F.log_softmax(logits2, dim=-1)
            pad_mask = tgt_out.ne(tgt_pad_idx).unsqueeze(-1)  # [B, T, 1]
            kl = 0.5 * (
                F.kl_div(p, q.detach().exp(), reduction='none')
                + F.kl_div(q, p.detach().exp(), reduction='none')
            )
            kl_loss = (kl * pad_mask).sum() / pad_mask.sum()

            loss = ce_loss + rdrop_alpha * kl_loss
```

**新增 CLI 参数：** `--rdrop_alpha`，默认 `3.0`，控制 KL loss 权重。

**注意事项：**
- 训练时间约增加 60-80%（每步两次 forward）
- 但可配合减少 epoch 数来平衡
- 只在 `model.train()` 时启用，`evaluate` 不受影响

**预期提升：** +1~2 BLEU

---

### 改进 3：源端噪声数据增强（增加数据多样性）

**文件：** `train_transformer.py` 的 `TranslationDataset`

当前已有 word_dropout（替换为 UNK），在此基础上增加：
- **Token delete：** 随机删除 token（概率 = word_dropout）
- **Token swap：** 随机交换相邻 token（概率 = word_dropout）

**在 `TranslationDataset` 中增加方法：**

```python
    def _apply_src_noise(self, ids: List[int]) -> List[int]:
        """源端噪声：word dropout → token delete → token swap。"""
        if self.word_dropout <= 0.0:
            return ids
        # 1) token delete
        result = [tok for tok in ids if random.random() >= self.word_dropout]
        if not result:
            return ids  # 防止全删空
        # 2) token swap
        for i in range(len(result) - 1):
            if random.random() < self.word_dropout:
                result[i], result[i + 1] = result[i + 1], result[i]
        return result
```

**修改 `__getitem__`：**
- 源端：调用 `_apply_src_noise`（替代原来的 `_apply_word_dropout`）
- 目标端：保持原来的 word_dropout 不变（目标端不做 swap/delete）

**预期提升：** +0.5~1.5 BLEU

---

## 配置变更

**文件：** `configs/train_cuda.yaml`

| 参数 | 当前值 | 新值 | 原因 |
|------|--------|------|------|
| `run_tag` | `laptop4060_opt_v2` | `v3_char_rdrop` | 区分实验 |
| `epochs` | `45` | `60` | 字符级序列更长，需更多 epoch |
| `max_tgt_len` | `64` | `96` | 字符级序列变长 |
| `dropout` | `0.15` | `0.15` | 不变，靠 R-Drop 替代 |
| `word_dropout` | `0.02` | `0.05` | 噪声增强略加强 |
| `label_smoothing` | `0.1` | `0.15` | 配合 R-Drop 略加大 |
| `rdrop_alpha` | (无) | `3.0` | R-Drop KL 损失权重 |
| `patience` | `10` | `12` | 给模型更多收敛机会 |

---

## 执行步骤（按顺序）

### Step 1：改预处理代码

- [x] 修改 `cmn-eng-simple/preprocess/tokenizer.py`：中文端改为字符级
- [x] 修改 `cmn-eng-simple/preprocess/build_dataset.py`：频率阈值降为 2

### Step 2：重新生成数据

```powershell
cd cmn-eng-simple/preprocess
python tokenizer.py
subword-nmt learn-bpe -s 5000 < en.txt > en_code.txt
subword-nmt apply-bpe -c en_code.txt < en.txt > en_refine.txt
subword-nmt get-vocab --input en_refine.txt --output en_vocab.txt
python build_dataset.py
```

- [x] 确认新的 `word2int_cn.json` 已生成，词表大小 `2250`，测试集 UNK 率约 `0.17%`
- [x] 确认 `training.txt`/`validation.txt`/`testing.txt` 格式正确（tab 分隔检查通过）

### Step 3：改训练代码

- [x] `train_transformer.py`：添加 R-Drop 到 `train_one_epoch`
- [x] `train_transformer.py`：添加 `_apply_src_noise` 到 `TranslationDataset`
- [x] `train_transformer.py`：添加 `--rdrop_alpha` CLI 参数
- [x] 更新 `configs/train_cuda.yaml`

### Step 4：验证改动

```powershell
# 快速 sanity check（5 个样本，1 epoch）
python train_transformer.py --config configs/train_cuda.yaml --epochs 1 --max_bleu_samples 5
```

- [x] 确认训练能正常启动，loss 正常下降，无报错（1 epoch sanity 已完成）

### Step 5：正式训练

```powershell
python train_transformer.py --config configs/train_cuda.yaml
```

- [x] 训练完成后读取 `metrics_train.json` 检查最终 BLEU（简体重跑：`final_val_bleu=41.43`，`test_bleu=40.89`）

### Step 6：评估与对比

- [x] 对比 v2 vs v3 的 test BLEU（v2: `26.61` → v3: `40.89`，提升约 `+14.28`）
- [x] 检查 UNK 率是否降至 < 5%（当前约 `0.17%`）
- [x] 检查过拟合 gap 是否缩小（末轮 `train_loss=2.582` vs `val_loss=2.764`，gap≈`0.182`，较基线 `1.48` 明显缩小）

---

## 预期效果

| 阶段 | BLEU | 备注 |
|------|------|------|
| 基线 | 24.57 | 初始训练 |
| v2（正则化+checkpoint avg） | 26.61 | 已完成 |
| + 字符级分词 | 29-31 | 消除 UNK |
| + R-Drop | 30-33 | 减少过拟合 |
| + 噪声增强 | 31-34 | 增加数据多样性 |

---

## 风险与回退

1. **字符级序列太长导致训练变慢：** max_tgt_len 设为 96 而非更大，
   若显存不够则降 batch_size 到 64。
2. **R-Drop 训练不稳定：** 若 loss 爆炸，降低 rdrop_alpha 到 1.0 或关闭。
3. **改预处理后数据集变小：** 字符级 UNK 少→ 过滤掉的句子也少→ 数据量可能反而增加。
