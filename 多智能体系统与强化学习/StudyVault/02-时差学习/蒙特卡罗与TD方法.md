---
source_pdf: 课程1,2.pdf
part: 2
keywords: monte carlo, temporal difference, TD error, bootstrapping, sampling
---

# 蒙特卡罗与TD方法（★★★）

#rl #td #mc

## 概览表（一目了然）

| 条目 | 要点 |
|------|------|
| 动机 | DP需要完全模型 → 需要**无模型**方法 |
| MC | 用完整轨迹的返回值更新（Sampling，无Bootstrap） |
| TD | 用单步转移+估计值更新（Sampling + Bootstrap） |
| 核心区别 | MC无偏高方差；TD有偏低方差 |

## 三种方法的统一视角

```
DP：V(s) ← E[r + γV(s')]          需要模型，用所有可能的s'
MC：V(s) ← V(s) + α[G_t - V(s)]   不需模型，用一条完整轨迹
TD：V(s) ← V(s) + α[r + γV(s') - V(s)]  不需模型，用一步采样
```

| | Bootstrap | Sampling | 需要模型 |
|--|-----------|----------|----------|
| DP | ✅ | ❌ | ✅ |
| MC | ❌ | ✅ | ❌ |
| TD | ✅ | ✅ | ❌ |

## 蒙特卡罗方法（MC）

**核心思想**：等到episode结束，用**完整返回值** $G_t$ 更新。

$$
V(s_t) \leftarrow V(s_t) + \alpha[G_t - V(s_t)]
$$

其中 $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$

- **First-Visit MC**：每个episode中，只在首次访问状态 $s$ 时更新
- **Every-Visit MC**：每次访问都更新

| 优点 | 缺点 |
|------|------|
| 无偏估计 | 必须等episode结束 |
| 不需模型 | 高方差 |
| 概念简单 | 只适用有终止状态的问题 |

## TD(0)方法

**核心思想**：每步用 $r + \gamma V(s')$ 替代完整返回值。

$$
V(s_t) \leftarrow V(s_t) + \alpha[\underbrace{r_t + \gamma V(s_{t+1})}_{\text{TD目标}} - V(s_t)]
$$

**TD误差**：$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

| 优点 | 缺点 |
|------|------|
| 每步可更新（在线学习） | 有偏（Bootstrap） |
| 低方差 | 依赖初始值函数 |
| 适用连续任务 | 对步长敏感 |

## MC vs TD 关键对比

| 维度 | MC | TD |
|------|----|----|
| 更新时机 | episode结束 | 每一步 |
| 偏差 | 无偏 | 有偏 |
| 方差 | 高 | 低 |
| 收敛速度 | 慢 | 通常更快 |
| 是否需终止 | 是 | 否 |
| 使用信息 | 完整轨迹 | 单步转移 |

> [!warning] 偏差-方差权衡
> MC无偏但方差大（整条轨迹的随机性都累积）；TD有偏但方差小（只用一步随机性）。实践中TD通常收敛更快。

---

## 考试/测试常见模式

| 场景/关键词 | 答案 |
|-------------|------|
| "TD和MC的本质区别" | **是否Bootstrap**（TD用估计更新估计） |
| "哪个无偏" | **MC**（TD有偏因为V(s')有误差） |
| "哪个方差小" | **TD**（只用一步随机性） |
| "连续任务用哪个" | **TD**（MC需要episode结束） |

## 相关笔记
- [[SARSA与Q-Learning]]
- [[TD(λ)与资格迹]]
- [[动态规划]]
