---
source_pdf: 04 策略梯度.pdf
part: 4
keywords: policy gradient theorem, REINFORCE, score function, log-derivative trick
---

# 策略梯度定理与REINFORCE（★★★）

#rl #policy-gradient

## 概览表（一目了然）

| 条目 | 要点 |
|------|------|
| 核心思想 | 直接参数化策略，通过梯度上升优化 |
| 策略梯度定理 | 给出目标函数梯度的解析形式 |
| REINFORCE | 最基础的策略梯度算法（MC采样） |
| 关键技巧 | Log-Derivative Trick（Score Function） |

## 为什么要策略梯度？

值函数方法的局限：
- 连续动作空间：$\arg\max_a Q(s,a)$ 本身是困难优化
- 确定性策略不适合随机博弈（如石头剪刀布）
- 值函数微小变化→策略剧烈变化（不稳定）

**策略梯度优势**：
- 天然支持连续动作
- 可学随机策略
- 策略变化**平滑**

## 策略参数化

**离散动作 — Softmax**：
$$
\pi_\theta(a|s) = \frac{e^{h(s,a,\theta)}}{\sum_{a'}e^{h(s,a',\theta)}}
$$

**连续动作 — 高斯**：
$$
\pi_\theta(a|s) = \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(a-\mu_\theta(s))^2}{2\sigma^2}\right)
$$

## 策略梯度定理

$$
\nabla_\theta J(\theta) = E_{\pi_\theta}\left[\sum_{t=0}^T Q^{\pi_\theta}(s_t,a_t)\nabla_\theta\log\pi_\theta(a_t|s_t)\right]
$$

**直觉**：
- $\nabla\log\pi_\theta(a|s)$：增加动作 $a$ 概率的方向
- $Q^\pi(s,a)$：该动作的好坏（权重）
- 好动作→增加概率，坏动作→降低概率

## Log-Derivative Trick

$$
\nabla_\theta\pi_\theta(a|s) = \pi_\theta(a|s)\nabla_\theta\log\pi_\theta(a|s)
$$

因此：
$$
\nabla_\theta E_{\pi_\theta}[f(a)] = E_{\pi_\theta}[f(a)\nabla_\theta\log\pi_\theta(a|s)]
$$

> [!tip] 关键意义
> 将**无法直接计算的梯度**转化为可通过**采样估计**的期望形式。

## REINFORCE算法

用MC返回 $G_t$ 替代 $Q^\pi(s_t,a_t)$：

$$
\theta \leftarrow \theta + \alpha\gamma^t G_t\nabla_\theta\log\pi_\theta(a_t|s_t)
$$

```
初始化策略参数 θ
对每个 episode:
    用 π_θ 采样轨迹 (s₀,a₀,r₀,...,sₜ,aₜ,rₜ)
    对每一步 t:
        G_t = Σ_{t'=t}^T γ^{t'-t} r_{t'}
        θ ← θ + α · γ^t · G_t · ∇_θ log π_θ(aₜ|sₜ)
```

| 优点 | 缺点 |
|------|------|
| 无偏 | **高方差**（MC返回的随机性） |
| 理论简洁 | 慢收敛 |
| 容易实现 | 必须等episode结束 |

---

## 考试/测试常见模式

| 场景/关键词 | 答案 |
|-------------|------|
| "策略梯度定理写出来" | $\nabla J = E[\sum_t Q^\pi(s_t,a_t)\nabla\log\pi_\theta(a_t|s_t)]$ |
| "Log-Trick的作用" | 将梯度转化为可采样的期望形式 |
| "REINFORCE的缺点" | 高方差、需等episode结束 |
| "策略梯度vs值函数方法" | PG支持连续动作+随机策略，但方差大 |

## 相关笔记
- [[基线与优势函数]]
- [[Actor-Critic]]
- [[PPO]]
