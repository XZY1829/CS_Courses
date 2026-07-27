---
source_pdf: 05 深度强化学习.pdf
part: 5
keywords: DQN, target network, experience replay, double DQN, dueling DQN
---

# DQN（★★★）

#deep-rl #dqn #experience-replay

## 概览表（一目了然）

| 条目 | 要点 |
|------|------|
| 核心思想 | 深度网络逼近 $Q^*$，端到端从像素到动作 |
| 关键创新1 | 目标网络（固定TD目标） |
| 关键创新2 | 经验回放（打破数据相关性） |
| 变体 | Double DQN、Dueling DQN、Prioritized ER |

## 基本思想

$$
\hat{Q}(s,a,w) \approx Q^*(s,a)
$$

输入：原始像素（连续4帧灰度图像）→ CNN → 输出每个动作的Q值

## 损失函数

$$
J(w) = E\left[(R + \gamma\max_{a'}\hat{Q}(S',a',w^-) - \hat{Q}(S,A,w))^2\right]
$$

梯度更新（半梯度）：
$$
\Delta w = \alpha(R + \gamma\max_a\hat{Q}(s',a,w^-) - \hat{Q}(s,a,w))\nabla_w\hat{Q}(s,a,w)
$$

## 两大问题及解决

### 问题1：追逐不稳定目标

TD目标随 $w$ 变化 → "追逐移动靶子"

**解决 → 目标网络**：

| 网络 | 参数 | 作用 | 更新方式 |
|------|------|------|---------|
| 在线网络 | $w$ | 选择动作+梯度更新 | 每步SGD |
| 目标网络 | $w^-$ | 计算TD目标 | 每N步复制/软更新 |

### 问题2：过高估计

$\max$ 操作 + 噪声估计 → 系统性正偏差

$$
E[\max_a\hat{Q}(s,a)] \geq \max_a E[\hat{Q}(s,a)]
$$

**解决 → Double DQN**（见变体节）

## DQN完整算法

```
初始化 在线网络 w, 目标网络 w⁻=w, 缓冲区 D
对每个 episode:
    初始化 s（4帧图像）
    对每步:
        以 ε-贪心选择 a
        执行 a → r, s'
        存储 (s,a,r,s') 到 D
        从 D 随机采样 mini-batch
        y_j = r_j + γ max_a' Q(s_j',a';w⁻)
        L = (y_j - Q(s_j,a_j;w))²
        梯度下降更新 w
        每 C 步：w⁻ ← w
```

## 重要变体

### Double DQN

用在线网络**选择**动作，目标网络**评估**：
$$
y = R + \gamma\hat{Q}(S', \arg\max_a\hat{Q}(S',a,w), w^-)
$$

> [!tip] 关键
> 解耦"选择最优动作"和"评估其价值"，减少过高估计。

### Dueling DQN

将Q分解为状态值+优势：
$$
Q(s,a;w) = V(s;w_V) + A(s,a;w_A) - \frac{1}{|A|}\sum_{a'}A(s,a';w_A)
$$

### Prioritized Experience Replay

按TD误差大小赋予优先级，优先采样"惊讶度高"的样本。

---

## 考试/测试常见模式

| 场景/关键词 | 答案 |
|-------------|------|
| "DQN为什么需要目标网络" | 固定TD目标N步不变，防止追逐移动目标 |
| "DQN为什么需要经验回放" | 打破数据时序相关性，稳定训练 |
| "过高估计的原因" | max操作+噪声估计→系统正偏差 |
| "Double DQN怎么解决" | 在线网络选动作，目标网络评估 |

## 相关笔记
- [[线性与非线性方法]]
- [[批量方法与经验回放]]
- [[PPO]]
- [[A3C]]
