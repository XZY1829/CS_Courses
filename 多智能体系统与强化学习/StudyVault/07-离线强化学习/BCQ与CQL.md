---
source_pdf: 07 离线强化学习(1).pptx
part: 7
keywords: BCQ, CQL, conservative Q-learning, batch-constrained, VAE, offline policy
---

# BCQ与CQL（★★★）

#rl #offline-rl #bcq #cql

## 概览表（一目了然）

| 条目 | 要点 |
|------|------|
| 共同目标 | 解决离线RL中OOD动作的Q值过高估计 |
| BCQ策略 | **限制动作空间** — 只在数据支撑上选动作 |
| CQL策略 | **压低Q值** — 正则化使OOD动作Q值低 |
| 关系 | BCQ=空间约束；CQL=价值压制 |

## BCQ（Batch-Constrained Q-Learning）

### 核心思想

**只在数据集支撑内的动作上做TD更新**，不外推到未见过的动作。

### 表格型BCQ

$$
Q(s,a) \leftarrow r + \gamma\max_{a': (s',a')\in D}Q(s',a')
$$

只对**数据中出现过的** $(s', a')$ 取max。

### 连续动作BCQ

用**VAE（变分自编码器）**生成数据分布内的动作：

1. 训练VAE学习数据中的动作分布 $G_\omega(s)$
2. 生成N个候选动作 $\{a_i \sim G_\omega(s)\}$
3. 用扰动网络微调：$a' = a_i + \Phi_\xi(s, a_i)$
4. 选Q值最高的：$\pi(s) = \arg\max_{a'} Q(s, a')$

> [!tip] n和Φ的权衡
> - n和Φ越小 → 越接近BC（保守但性能有限）
> - n和Φ越大 → 越接近标准RL（激进但易OOD）

### BCQ的双Q学习

用权重系数 $\lambda$ 在最小Q和最大Q之间权衡：
$$
Q_\text{target} = \lambda\min_{i=1,2}Q_i(s',a') + (1-\lambda)\max_{i=1,2}Q_i(s',a')
$$

## CQL（Conservative Q-Learning）

### 核心思想

通过正则化使Q值整体偏低（保守），确保策略不会选择Q被高估的OOD动作。

### CQL目标函数

$$
\min_Q \alpha\left(E_{a\sim\mu}[Q(s,a)] - E_{a\sim\hat{\pi}_\beta}[Q(s,a)]\right) + \text{标准TD损失}
$$

分解理解：
- $E_{a\sim\mu}[Q(s,a)]$ → **最小化**所有动作（尤其OOD）的Q值
- $E_{a\sim\hat{\pi}_\beta}[Q(s,a)]$ → **最大化**数据内动作的Q值
- 效果：OOD动作Q值被压低，数据内动作Q值被保持

> [!important] CQL的下界保证
> CQL学到的Q值是真实Q值的**下界**，因此策略选择的动作至少比数据中的不差。

### CQL实践版本

使用 $\mu = \text{Uniform}$ 或 $\mu = \pi_\theta$（当前策略）的分布来做最小化项。

加入熵项控制动作收敛性。

## BCQ vs CQL

| 维度 | BCQ | CQL |
|------|-----|-----|
| 核心策略 | 限制动作空间（空间约束） | 压低Q值（价值压制） |
| 实现方式 | VAE生成+扰动 | Q函数正则化 |
| 理论保证 | 动作在数据支撑内 | Q值是下界 |
| 对稀疏奖励 | 可能过于保守 | 相对灵活 |
| 超参敏感度 | n, Φ的选择 | α的选择 |
| 适用场景 | 数据质量较高 | 数据多样性大 |

---

## 考试/测试常见模式

| 场景/关键词 | 答案 |
|-------------|------|
| "BCQ怎么解决OOD问题" | 用VAE约束动作只在数据分布内 |
| "CQL怎么解决OOD问题" | 正则化压低OOD动作的Q值 |
| "BCQ和CQL的区别" | BCQ约束动作空间；CQL约束Q值 |
| "CQL为什么是保守的" | 学到的Q是下界，宁可低估不高估 |

## 相关笔记
- [[离线RL与分布偏移]]
- [[DQN]]
- [[SARSA与Q-Learning]]
- [[考试陷阱]]
