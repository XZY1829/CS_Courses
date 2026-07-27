---
source_pdf: 05 深度强化学习.pdf
part: 5
keywords: A3C, asynchronous, parallel, entropy regularization, multi-worker
---

# A3C（★★）

#deep-rl #a3c #actor-critic

## 概览表（一目了然）

| 条目 | 要点 |
|------|------|
| 全称 | Asynchronous Advantage Actor-Critic |
| 核心思想 | 多Worker异步并行 + 共享全局模型 |
| 特色 | 不需要经验回放（并行提供多样性） |
| 优势 | 多核CPU即可训练，适合离散+连续动作 |

## 架构

```
            ┌──────────────────┐
            │  Global Network  │
            │   (θ, θ_v)       │
            └──┬──┬──┬──┬──┬──┘
   ┌──────────┘  │  │  │  └──────────┐
   ▼             ▼  ▼  ▼             ▼
Worker 1      Worker 2  ...      Worker n
 Env 1         Env 2               Env n
```

## 每个Worker的流程

1. **复制**全局网络参数到本地
2. 与本地环境**交互**，收集一段经验
3. **计算**策略梯度和值函数梯度
4. **异步推送**梯度给全局网络
5. 全局网络**更新参数**

## 梯度计算

**Actor（策略梯度）**：
$$
\nabla_\theta J = E[\nabla_\theta\log\pi_\theta(a_t|s_t)\cdot A(s_t,a_t)]
$$

**Critic（值函数损失）**：
$$
L_V = (R - \hat{V}(s_t;\theta_v))^2
$$

**优势函数**：$A(s_t,a_t) = r_t + \gamma V(s_{t+1}) - V(s_t)$

## 熵正则化

总损失加入策略熵：
$$
L = L_\text{policy} + \lambda L_V - \beta H(\pi)
$$

其中 $H(\pi) = -\sum_a\pi(a|s)\log\pi(a|s)$

> [!tip] 熵的作用
> 鼓励探索，防止策略过早收敛到次优解。$\beta$ 控制探索强度。

## A3C vs DQN

| 特性 | DQN | A3C |
|------|-----|-----|
| 方法类型 | 值函数方法 | 策略梯度（Actor-Critic） |
| 并行方式 | 单进程+经验回放 | 多进程异步 |
| 硬件 | GPU | **多核CPU** |
| 经验回放 | 需要 | 不需要 |
| 动作空间 | 仅离散 | 离散+连续 |
| 数据相关性 | 经验回放打破 | 多环境并行打破 |

> [!warning] 异步更新的代价
> Worker使用的可能是旧参数计算的梯度（梯度延迟），但实践中多样性增加的好处大于噪声的坏处。

---

## 考试/测试常见模式

| 场景/关键词 | 答案 |
|-------------|------|
| "A3C为什么不需经验回放" | 多Worker异步并行本身提供数据多样性 |
| "熵正则化的作用" | 鼓励探索，防止策略过早收敛 |
| "A3C vs DQN硬件需求" | A3C用多核CPU，DQN用GPU |
| "异步带来的问题" | 梯度延迟（Worker可能用旧参数） |

## 相关笔记
- [[Actor-Critic]]
- [[PPO]]
- [[DQN]]
