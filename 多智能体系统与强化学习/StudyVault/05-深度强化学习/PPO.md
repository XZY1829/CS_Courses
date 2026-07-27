---
source_pdf: 05 深度强化学习.pdf
part: 5
keywords: PPO, proximal policy optimization, importance sampling, clip, TRPO
---

# PPO（★★★）

#deep-rl #ppo #importance-sampling

## 概览表（一目了然）

| 条目 | 要点 |
|------|------|
| 核心问题 | On-Policy样本效率低（数据只能用一次） |
| 解决思路 | 重要性采样 + 约束策略更新幅度 |
| PPO-KL | 加KL散度惩罚项 |
| PPO-Clip | 裁剪概率比率（更常用） |

## 动机：On-Policy → Off-Policy

On-Policy问题：$\pi_\theta$ 收集的数据只能更新 $\theta$ **一次**，更新后旧数据过期。

**目标**：用旧策略 $\pi_{\theta_k}$ 的数据多次更新 $\theta$。

## 重要性采样

$$
E_{x\sim p}[f(x)] = E_{x\sim q}\left[f(x)\frac{p(x)}{q(x)}\right]
$$

应用到策略梯度：
$$
J_{\theta_k}(\theta) = E_{(s,a)\sim\pi_{\theta_k}}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)}A_{\theta_k}(s,a)\right]
$$

> [!warning] 偏差问题
> 当 $\pi_\theta$ 和 $\pi_{\theta_k}$ 差距大时，重要性权重可能极大 → 方差爆炸。因此**必须限制策略更新幅度**。

## PPO-KL（KL惩罚版本）

$$
J_{PPO}^{\theta_k}(\theta) = J_{\theta_k}(\theta) - \beta\text{KL}(\theta, \theta_k)
$$

**自适应 $\beta$**：
- $\text{KL} > \text{KL}_{max}$ → 增大 $\beta$（更保守）
- $\text{KL} < \text{KL}_{min}$ → 减小 $\beta$（更大胆）

## PPO-Clip（Clip版本，最常用）

记 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_k}(a_t|s_t)}$：

$$
J_{PPO2} = \sum_{s_t,a_t}\min\left(r_t(\theta)A_t, \;\text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)A_t\right)
$$

### Clip机制直觉

| 情况 | 效果 |
|------|------|
| $A > 0$（好动作），$r_t > 1+\varepsilon$ | 截断 → 防止过度增加 |
| $A < 0$（坏动作），$r_t < 1-\varepsilon$ | 截断 → 防止过度降低 |

**核心效果**：策略比率被限制在 $[1-\varepsilon, 1+\varepsilon]$，更新不会太激进。

## PPO vs TRPO

| 特性 | TRPO | PPO |
|------|------|-----|
| 约束方式 | 硬约束 $\text{KL} < \delta$ | 软惩罚或Clip |
| 优化方法 | 共轭梯度+线搜索 | 标准SGD |
| 实现难度 | **高** | **低** |
| 效果 | 好 | 同样好或更好 |

## PPO算法流程

```
初始化 θ₀
对每轮 k:
    用 θ_k 收集轨迹数据
    计算优势函数 A_{θ_k}(s_t,a_t)
    对数据做多个epoch优化：
        r_t(θ) = π_θ(a_t|s_t) / π_{θ_k}(a_t|s_t)
        L = min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)
        θ ← θ + α∇_θ L
    θ_{k+1} ← θ
```

> [!important] PPO的工业地位
> PPO是目前工业界最广泛使用的RL算法（如ChatGPT的RLHF就用PPO），因为它实现简单、训练稳定、效果好。

---

## 考试/测试常见模式

| 场景/关键词 | 答案 |
|-------------|------|
| "PPO为什么需要Clip" | 限制策略更新幅度，防止重要性权重过大 |
| "PPO-Clip公式" | $\min(r_t A, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)A)$ |
| "PPO和TRPO区别" | PPO用Clip/KL惩罚替代TRPO的硬约束 |
| "重要性采样的作用" | 用旧策略数据更新新策略 |

## 相关笔记
- [[Actor-Critic]]
- [[A3C]]
- [[DQN]]
- [[考试陷阱]]
