---
source_pdf: NLP课件_jfyu_第七章_循环神经网络_release.pdf
part: 7.5
keywords: gru, lstm, gate-mechanism, update-gate, reset-gate, forget-gate, cell-state
---

# GRU与LSTM（★★★）

#nlp-deep-learning #rnn #gru #lstm #gate-mechanism #update-gate #reset-gate #forget-gate #input-gate #output-gate #cell-state #concept

## 概览表（一目了然）
| 条目 | 要点 |
|------|------|
| 门控动机 | 有选择地**加入新信息**、有选择地**遗忘旧信息**，缓解 #gradient-vanishing |
| GRU | 2 门（#update-gate + #reset-gate），结构简洁，参数较少 |
| LSTM | 3 门 + #cell-state，表达力更强，经典长程依赖方案 |
| 核心思想 | 通过 sigmoid 门（0~1）控制信息流的**比例**而非硬开关 |
| 选型 | GRU 训练快、参数少；LSTM 复杂任务/长序列往往更稳 |

## 门控机制（Gate Mechanism）

SRNN 的隐状态每次被完全重写，导致长程梯度连乘衰减。门控 RNN 引入 **#gate-mechanism**：

1. **有选择地加入新信息**：控制当前输入有多少写入状态
2. **有选择地遗忘旧信息**：控制历史状态有多少被保留

门值由 sigmoid 函数产生，取值 **(0, 1)**，实现**软性的、可微的**信息开关。

```
门控信息流:
旧状态 hₜ₋₁ ──→ [遗忘/更新门] ──→ 保留部分
新输入 xₜ    ──→ [输入/重置门] ──→ 写入部分
                              ↓
                           新状态 hₜ
```

---

## GRU（Gated Recurrent Unit）

GRU 将 LSTM 的遗忘门与输入门**合并**为更新门，共 **2 个门**：

### 完整公式

**更新门（Update Gate）**：
**zₜ = σ(Wz·xₜ + Uz·hₜ₋₁ + bz)**

**重置门（Reset Gate）**：
**rₜ = σ(Wr·xₜ + Ur·hₜ₋₁ + br)**

**候选隐状态（Candidate State）**：
**h̃ₜ = tanh(Wh·xₜ + Uh·(rₜ ⊙ hₜ₋₁) + bh)**

**最终隐状态（Final State）**：
**hₜ = zₜ ⊙ hₜ₋₁ + (1 - zₜ) ⊙ h̃ₜ**

其中 **⊙** 表示逐元素乘（Hadamard 积），**σ** 为 sigmoid 函数。

### 各门作用

| 门 | 标签 | 作用 |
|----|------|------|
| **更新门 zₜ** | #update-gate | z→1 保留旧状态；z→0 写入新候选 |
| **重置门 rₜ** | #reset-gate | r→0 忽略历史，仅基于 xₜ 计算候选；r→1 正常融合 |

```
GRU 数据流:
xₜ ──┬──→ zₜ (更新门) ──────────────┐
     ├──→ rₜ (重置门) → ⊙ hₜ₋₁ ──┐  │
     └──→ tanh ──→ h̃ₜ ────────────┼──→ hₜ = z⊙hₜ₋₁ + (1-z)⊙h̃ₜ
hₜ₋₁ ─────────────────────────────┘
```

> [!tip] 更新门的直觉
> hₜ 是 hₜ₋₁ 与 h̃ₜ 的**凸组合**：zₜ 越大，越"记住"过去；越小，越"接受"新信息。

---

## LSTM（Long Short-Term Memory）

LSTM 引入独立的 **#cell-state**（记忆单元 cₜ）与 **3 个门**，将"记忆"与"输出"解耦。

### 完整公式

**输入门（Input Gate）**：
**iₜ = σ(Wi·xₜ + Ui·hₜ₋₁ + bi)**

**遗忘门（Forget Gate）**：
**fₜ = σ(Wf·xₜ + Uf·hₜ₋₁ + bf)**

**输出门（Output Gate）**：
**oₜ = σ(Wo·xₜ + Uo·hₜ₋₁ + bo)**

**候选记忆（Candidate Cell）**：
**c̃ₜ = tanh(Wc·xₜ + Uc·hₜ₋₁ + bc)**

**记忆单元（Cell State）**：
**cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ**

**隐状态（Hidden State）**：
**hₜ = oₜ ⊙ tanh(cₜ)**

### 各门作用

| 门/状态 | 标签 | 作用 |
|---------|------|------|
| **遗忘门 fₜ** | #forget-gate | 控制 cₜ₋₁ 有多少被保留 |
| **输入门 iₜ** | #input-gate | 控制 c̃ₜ 有多少写入 cell state |
| **输出门 oₜ** | #output-gate | 控制 cell state 有多少暴露为 hₜ |
| **记忆单元 cₜ** | #cell-state | 长期信息高速公路，梯度可沿加法路径传播 |

```
LSTM 数据流:
cₜ₋₁ ──→ [fₜ] ──→ 保留 ──┐
                          (+) ──→ cₜ ──→ tanh ──→ [oₜ] ──→ hₜ
c̃ₜ  ──→ [iₜ] ──→ 写入 ──┘
         ↑
    xₜ, hₜ₋₁ 共同驱动三门
```

> [!important] 梯度高速公路
> cₜ = fₜ⊙cₜ₋₁ + iₜ⊙c̃ₜ 中的**加法结构**使 ∂cₜ/∂cₜ₋₁ 含 fₜ 项，当 fₜ≈1 时梯度近似恒等传播，有效缓解 #gradient-vanishing。

---

## LSTM 变体

| 变体 | 改动 | 说明 |
|------|------|------|
| **无遗忘门** | 去掉 fₜ，cₜ = cₜ₋₁ + iₜ⊙c̃ₜ | 原始 LSTM（1997），无法主动遗忘 |
| **耦合门** | iₜ = 1 - fₜ | 输入与遗忘互补，减少参数量 |
| **Peephole 连接** | 门控输入加入 cₜ₋₁ | 门直接"窥视"cell state，细粒度控制 |

---

## GRU vs LSTM 对比

| 维度 | GRU | LSTM |
|------|-----|------|
| **门数量** | 2（更新、重置） | 3（输入、遗忘、输出） |
| **记忆结构** | 无独立 cell state | 独立 #cell-state cₜ |
| **参数量** | 较少 | 较多（约 4/3 倍） |
| **训练速度** | 通常更快 | 稍慢 |
| **长程依赖** | 良好 | 通常更强 |
| **适用场景** | 中小规模、数据有限 | 长序列、复杂依赖 |
| **更新方式** | hₜ = z⊙hₜ₋₁ + (1-z)⊙h̃ₜ | cₜ 加法更新 + oₜ 过滤输出 |

> [!warning] 没有绝对优劣
> 实际选型应通过验证集对比；GRU 在多数 NLP 任务上与 LSTM 性能接近，但训练更高效。

---

## 考试/测试常见模式
| 场景/关键词 | 答案 |
|-------------|------|
| "GRU 更新门" | zₜ=σ(·)；hₜ=z⊙hₜ₋₁+(1-z)⊙h̃ₜ |
| "GRU 重置门" | rₜ=σ(·)；控制 hₜ₋₁ 是否参与候选计算 |
| "LSTM cell 更新" | cₜ = fₜ⊙cₜ₋₁ + iₜ⊙c̃ₜ |
| "LSTM 隐状态" | hₜ = oₜ ⊙ tanh(cₜ) |
| "遗忘门作用" | 控制旧记忆 cₜ₋₁ 保留比例 |
| "为何缓解梯度消失" | cell state 加法路径 + 门控选择性传播 |
| "GRU vs LSTM" | GRU 更简、更快；LSTM 有独立 cell、三门 |
| "Peephole" | 门控输入额外连接 cₜ₋₁ |

## 相关笔记
- [[RNN基础与参数学习]]
- [[深度RNN与递归神经网络]]
