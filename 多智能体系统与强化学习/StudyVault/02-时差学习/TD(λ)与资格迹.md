---
source_pdf: 课程1,2.pdf
part: 2
keywords: TD-lambda, eligibility trace, n-step TD, forward view, backward view
---

# TD(λ)与资格迹（★★）

#rl #td

## 概览表（一目了然）

| 条目 | 要点 |
|------|------|
| N步TD | TD(0)和MC之间的折中 — 看"更远的未来" |
| TD(λ) | 用指数加权平均**所有**N步回报 |
| 资格迹 | 后向视角实现，记录"哪些状态该为当前TD误差负责" |
| λ=0 | 退化为TD(0) |
| λ=1 | 退化为MC |

## N步TD预测

$$
R_t^{(n)} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1}r_{t+n-1} + \gamma^n V(s_{t+n})
$$

更新：$\Delta V(s_t) = \alpha[R_t^{(n)} - V(s_t)]$

| $n$ | 方法 | 说明 |
|-----|------|------|
| 1 | TD(0) | 只看一步 |
| 2 | 2步TD | 看两步 + Bootstrap |
| $\infty$ | MC | 看到episode结束 |

**误差缩减性质**：
$$
|E_\pi[R_t^{(n)}|s_t=s] - V^\pi(s)| \leq \gamma^n\max_s|V(s) - V^\pi(s)|
$$

## λ-返回

用指数衰减权重加权所有N步回报：

$$
R_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}R_t^{(n)}
$$

- 权重 $(1-\lambda)\lambda^{n-1}$ 随步数指数衰减，归一化到1
- $\lambda=0$：只保留 $R_t^{(1)}$ → TD(0)
- $\lambda=1$：等权重全部 → MC

## 前向视角 vs 后向视角

| 视角 | 实现方式 | 特点 |
|------|---------|------|
| 前向 | 从当前看未来，加权所有N步回报 | 概念直观，但需要整条轨迹 |
| 后向 | 用资格迹 + TD误差在线更新 | **可在线实现**，每步更新所有状态 |

## 资格迹（Eligibility Trace）

$$
e_t(s) = \begin{cases}\gamma\lambda\cdot e_{t-1}(s) + 1 & \text{if } s = s_t \ \gamma\lambda\cdot e_{t-1}(s) & \text{if } s \neq s_t\end{cases}
$$

- 当前访问的状态资格 +1
- 所有状态资格按 $\gamma\lambda$ 衰减
- **含义**：资格越高 → 越该为当前TD误差负责 → 更新量越大

## TD(λ)算法

```
初始化 V(s), e(s)=0
对每个 episode:
    初始化 s
    对每一步:
        选择动作 a（ε-贪心）
        执行 a → 观察 r, s'
        δ ← r + γV(s') - V(s)        // TD误差
        e(s) ← e(s) + 1               // 资格+1
        对所有状态 s:
            V(s) ← V(s) + αδe(s)      // 按资格加权更新
            e(s) ← γλe(s)             // 资格衰减
        s ← s'
```

> [!warning] 计算复杂度
> TD(λ)每步需要更新**所有**状态的值函数和资格迹，计算代价为 $O(|S|)$。实际中常用稀疏实现。

---

## 考试/测试常见模式

| 场景/关键词 | 答案 |
|-------------|------|
| "λ=0退化为什么" | **TD(0)** |
| "λ=1退化为什么" | **MC** |
| "资格迹的作用" | 记录各状态应为TD误差承担多少更新 |
| "前向后向视角等价" | 数学上等价，后向可在线实现 |

## 相关笔记
- [[蒙特卡罗与TD方法]]
- [[SARSA与Q-Learning]]
- [[值函数逼近框架]]
