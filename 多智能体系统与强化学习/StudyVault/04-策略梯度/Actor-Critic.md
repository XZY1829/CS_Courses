---
source_pdf: 04 策略梯度.pdf
part: 4
keywords: actor-critic, A2C, QAC, policy network, value network
---

# Actor-Critic（★★★）

#rl #policy-gradient #actor-critic

## 概览表（一目了然）

| 条目 | 要点 |
|------|------|
| 核心思想 | Actor（策略网络）+ Critic（值网络）协作 |
| Actor | 负责选择动作 $\pi_\theta(a|s)$ |
| Critic | 负责评估动作 $V_w(s)$ 或 $Q_w(s,a)$ |
| 优势 | 比REINFORCE方差低，每步可更新 |

## 架构

```
         Environment
         ↓ state, reward
    ┌─────────┐
    │ Critic  │──── TD误差 δ ────► ┌─────────┐
    │ V(s;w)  │                    │  Actor  │
    └─────────┘                    │ π(a|s;θ)│──► action
                                   └─────────┘
```

- **Critic** 提供评估信号（TD误差/优势函数）
- **Actor** 根据评估信号调整策略

## QAC算法（Q Actor-Critic）

```
初始化 Actor参数θ, Critic参数w
对每个 episode:
    采样轨迹 {s₁,a₁,r₁, s₂,a₂,r₂, ...}
    对每步:
        δ_t = r_t + γQ_w(s_{t+1},a_{t+1}) - Q_w(s_t,a_t)
    更新 Actor: θ ← θ + β∇_θ log π_θ(a_t|s_t) · Q_w(s_t,a_t)
    更新 Critic: w ← w + α δ_t ∇_w Q_w(s_t,a_t)
```

## Advantage Actor-Critic（A2C）

用优势函数代替Q值，只需一个V网络：

**Actor更新**：
$$
\theta \leftarrow \theta + \beta\nabla_\theta\log\pi_\theta(a_t|s_t)\cdot A(s_t,a_t)
$$

**Critic更新**：
$$
w \leftarrow w + \alpha\delta_t\nabla_w V_w(s_t)
$$

其中 $A(s_t,a_t) \approx \delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)$

> [!tip] 为什么A2C只需V网络
> 因为 $A \approx \delta = r + \gamma V(s') - V(s)$，不需要显式的Q网络。

## Actor-Critic vs REINFORCE

| 特性 | REINFORCE | Actor-Critic |
|------|----------|--------------|
| Critic | 无 | 有（值函数网络） |
| 方差 | 高 | **低**（Critic降低方差） |
| 偏差 | 无偏 | 有偏（Critic估计误差） |
| 更新时机 | episode结束 | **每步可更新** |
| Bootstrap | 否 | 是 |
| 适用性 | 只能episodic | 也适用连续任务 |

> [!warning] 偏差来源
> Actor-Critic的偏差来自**Critic的估计误差**（V网络不完美），不是来自基线本身。

## 训练技巧

- **共享网络**：Actor和Critic可以共享底层特征提取层
- **熵正则化**：在损失中加入策略熵 $H(\pi)$ 鼓励探索
- **梯度裁剪**：防止梯度爆炸

---

## 考试/测试常见模式

| 场景/关键词 | 答案 |
|-------------|------|
| "Actor和Critic分别做什么" | Actor选动作，Critic评估动作好坏 |
| "A2C相比REINFORCE好在哪" | 方差低、每步可更新、可用于连续任务 |
| "A2C的偏差从何而来" | Critic（V网络）的估计误差 |
| "为什么只需V网络" | $A \approx \delta_t = r + \gamma V(s') - V(s)$ |

## 相关笔记
- [[策略梯度定理与REINFORCE]]
- [[基线与优势函数]]
- [[PPO]]
- [[A3C]]
