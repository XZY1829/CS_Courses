---
source_pdf: 课程1,2.pdf
part: 2
keywords: SARSA, Q-learning, on-policy, off-policy, epsilon-greedy
---

# SARSA与Q-Learning（★★★）

#rl #td #sarsa #q-learning

## 概览表（一目了然）

| 条目 | 要点 |
|------|------|
| SARSA | On-Policy TD控制，用实际选择的 $a'$ 更新 |
| Q-Learning | Off-Policy TD控制，用 $\max$ 操作更新 |
| 核心区别 | 行为策略与目标策略是否相同 |
| 应用 | SARSA更保守（安全场景）；Q-Learning更激进（学最优） |

## SARSA（State-Action-Reward-State-Action）

**On-Policy**：用当前策略实际选择的下一个动作 $a'$ 来更新。

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha[r_{t+1} + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)]
$$

```
初始化 Q(s,a) 任意值
对每个 episode:
    初始化 s，根据 Q + ε-贪心选择 a
    对每一步:
        执行 a → 观察 r, s'
        根据 Q + ε-贪心选择 a'    ← 用当前策略选
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        s ← s', a ← a'
```

## Q-Learning

**Off-Policy**：不管当前策略选什么，直接用 $\max$ 操作更新。

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha[r_{t+1} + \gamma\max_a Q(s_{t+1},a) - Q(s_t,a_t)]
$$

```
初始化 Q(s,a) 任意值
对每个 episode:
    初始化 s
    对每一步:
        根据 Q + ε-贪心选择 a
        执行 a → 观察 r, s'
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]  ← 用max
        s ← s'
```

## 核心对比

| 特性 | SARSA | Q-Learning |
|------|-------|------------|
| 类型 | On-Policy | Off-Policy |
| 更新目标 | $r + \gamma Q(s',a')$，$a'$由当前策略选 | $r + \gamma\max_a Q(s',a)$ |
| 行为 | **保守**，受探索惩罚影响 | **激进**，直接学最优 |
| 收敛到 | 当前策略的Q值 | 最优Q值 $Q^*$ |
| 行为策略=目标策略 | 是 | 否 |

> [!tip] 悬崖行走例子
> - SARSA会避开悬崖边缘（因为考虑了ε概率跌落的惩罚）
> - Q-Learning会靠近悬崖走最短路（因为max假设下一步选最优）

## 算法设计三大议题

1. **值函数表达**：V函数（只能用于已知模型）vs Q函数（可直接选动作）
2. **探索机制**：ε-贪心、UCB、Boltzmann等
3. **更新方式**：On-Policy（SARSA）vs Off-Policy（Q-Learning）

## ε-贪心策略

$$
a = \begin{cases}\arg\max_a Q(s,a) & \text{概率 } 1-\varepsilon \ \text{随机选择} & \text{概率 } \varepsilon\end{cases}
$$

- 初期 $\varepsilon$ 大 → 探索多
- 后期 $\varepsilon$ 小 → 利用多

---

## 考试/测试常见模式

| 场景/关键词 | 答案 |
|-------------|------|
| "SARSA和Q-Learning的更新公式" | 写出两个公式，指出 $a'$ vs $\max$ 的区别 |
| "On-Policy含义" | 用于更新的动作**来自当前策略** |
| "Off-Policy含义" | 更新目标**独立于**行为策略的动作选择 |
| "哪个更安全" | **SARSA**（考虑了探索风险） |
| "哪个学到Q*" | **Q-Learning** |

## 相关笔记
- [[蒙特卡罗与TD方法]]
- [[TD(λ)与资格迹]]
- [[DQN]]
