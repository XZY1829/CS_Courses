# 第三章：特征归一化与 Fisher 线性判别分析（FLD）笔记

> 课程：模式识别与计算机视觉（PRCV）  
> 对应课件：`3_Normalization_FLD.pdf`  
> 复习重点：**FLD 的目标函数、矩阵化、优化推导、多分类扩展、数值稳定性**

---

## 0. 本章主线

本章可以理解为两步：

1. **先把输入特征处理好（归一化）**，避免量纲和尺度干扰模型。  
2. **再做监督判别特征提取（FLD）**，直接围绕“分类可分性”优化投影方向。

一句话对比：

- PCA：强调“表示/重构方差”，不直接面向分类。
- FLD：强调“类间远、类内紧”，直接面向分类。

---

## 1. 特征归一化（页 14–21）

## 1.1 每维 Min-Max 归一化

对第 j 维特征：

$$
x_{ij}'=\frac{x_{ij}-x_{\min,j}}{x_{\max,j}-x_{\min,j}}
$$

映射到 [0,1]，也可映射到 [-1,1]。

作用：消除量纲影响，让不同维度数值范围可比。  
注意：

- 若 $x_{\max,j}=x_{\min,j}$，该维常数，需特殊处理（置零/删除该维）。
- 稀疏数据要谨慎，可能破坏“0 的稀疏结构”。

## 1.2 $\ell_2/\ell_1$ 归一化（按样本向量）

给定样本向量 $\mathbf{x}_i\in\mathbb{R}^d$：

$$
\mathbf{x}_i'=\frac{\mathbf{x}_i}{\|\mathbf{x}_i\|_2},\quad
\|\mathbf{x}_i\|_2=\sqrt{\mathbf{x}_i^\top\mathbf{x}_i}
$$

或

$$
\mathbf{x}_i'=\frac{\mathbf{x}_i}{\|\mathbf{x}_i\|_1},\quad
\|\mathbf{x}_i\|_1=\sum_{j=1}^d |x_{ij}|
$$

$\ell_1$ 常用于直方图、非负特征。

## 1.3 Z-score 标准化

按维度估计训练集统计量：

$$
\hat\mu_j=\frac{1}{n}\sum_{i=1}^n x_{ij},\quad
\hat\sigma_j^2=\frac{1}{n}\sum_{i=1}^n (x_{ij}-\hat\mu_j)^2
$$

变换为：

$$
x_{ij}'=\frac{x_{ij}-\hat\mu_j}{\hat\sigma_j}
$$

目标是各维近似 N(0,1)。

## 1.4 Robust Scaling（抗离群值）

$$
x_{ij}'=\frac{x_{ij}-\mathrm{median}_j}{\mathrm{IQR}_j},
\quad \mathrm{IQR}_j=Q3_j-Q1_j
$$

比均值/方差更抗异常值。

## 1.5 测试集归一化铁律（高频考点）

**只能用训练集参数归一化测试集**，不能在测试集重新估计 $min/max$、$\mu,\sigma$、median、IQR。  
交叉验证同理：每一折都只用该折训练子集估参数。

---

## 2. 为什么需要 FLD（页 22–24）

设监督分类任务，类别为 $y\in\{1,\dots,C\}$。

- PCA 最大化总体方差，不关心标签，可能投影后类别仍混在一起。
- FLD 使用标签信息，专门优化“可分性”。

FLD 核心思想：

> 找投影方向 $\mathbf{w}$，使**投影后类别中心尽量远**，同时**各类内部尽量紧凑**。

---

## 3. 二分类 FLD 的完整推导（重点，页 25–30）

以下先讲二分类 $y\in\{1,2\}$。

## 3.1 问题设定

数据点 $\mathbf{x}_i\in\mathbb{R}^d$，线性投影到一维：

$$
u_i=\mathbf{w}^\top \mathbf{x}_i
$$

两类均值：

$$
\boldsymbol{\mu}_1=\frac{1}{N_1}\sum_{y_i=1}\mathbf{x}_i,\quad
\boldsymbol{\mu}_2=\frac{1}{N_2}\sum_{y_i=2}\mathbf{x}_i
$$

投影后均值：

$$
m_1=\mathbf{w}^\top\boldsymbol{\mu}_1,\quad
m_2=\mathbf{w}^\top\boldsymbol{\mu}_2
$$

## 3.2 为什么不能只最大化 (m_2-m_1)^2

若只优化

$$
\max_{\mathbf{w}} (m_2-m_1)^2
$$

则 $\mathbf{w}\to \alpha\mathbf{w}$ 时目标会随 $\alpha^2$ 无界放大，且不考虑类内分散。  
因此需要同时惩罚类内散布。

## 3.3 Fisher 准则

定义每类投影后散度（类内散度）：

$$
s_k^2=\sum_{y_i=k}(u_i-m_k)^2,\quad k=1,2
$$

总类内散度：s_1^2+s_2^2。

Fisher 准则：

$$
J(\mathbf{w})=\frac{(m_2-m_1)^2}{s_1^2+s_2^2}
$$

目标：

$$
\max_{\mathbf{w}} J(\mathbf{w})
$$

## 3.4 矩阵化推导

先化分子：

$$
m_2-m_1=\mathbf{w}^\top(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)
$$

$$
(m_2-m_1)^2
=
\mathbf{w}^\top
(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)^\top
\mathbf{w}
$$

定义类间散度矩阵：

$$
S_B=(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)^\top
$$

则分子为 $\mathbf{w}^\top$ $S_B\mathbf{w}$。

再化分母：

$$
s_k^2=\sum_{y_i=k}\left(\mathbf{w}^\top(\mathbf{x}_i-\boldsymbol{\mu}_k)\right)^2
=
\mathbf{w}^\top
\left(\sum_{y_i=k}(\mathbf{x}_i-\boldsymbol{\mu}_k)(\mathbf{x}_i-\boldsymbol{\mu}_k)^\top\right)
\mathbf{w}
$$

定义第 k 类散度矩阵：

$$
S_k=\sum_{y_i=k}(\mathbf{x}_i-\boldsymbol{\mu}_k)(\mathbf{x}_i-\boldsymbol{\mu}_k)^\top
$$

类内散度矩阵：

$$
S_W=S_1+S_2
$$

所以：

$$
s_1^2+s_2^2=\mathbf{w}^\top S_W\mathbf{w}
$$

最终得到广义 Rayleigh 商：

$$
J(\mathbf{w})=
\frac{\mathbf{w}^\top S_B\mathbf{w}}{\mathbf{w}^\top S_W\mathbf{w}}
$$

## 3.5 拉格朗日法求解

把分母约束为 1：

$$
\max_{\mathbf{w}} \mathbf{w}^\top S_B\mathbf{w},
\quad \text{s.t. } \mathbf{w}^\top S_W\mathbf{w}=1
$$

拉格朗日函数：

$$
\mathcal{L}(\mathbf{w},\lambda)=
\mathbf{w}^\top S_B\mathbf{w}
-\lambda(\mathbf{w}^\top S_W\mathbf{w}-1)
$$

对 $\mathbf{w}$ 求导并令零：

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{w}}
=2S_B\mathbf{w}-2\lambda S_W\mathbf{w}=0
$$

得到广义特征值问题：

$$
S_B\mathbf{w}=\lambda S_W\mathbf{w}
$$

## 3.6 二分类闭式解

由于

$$
S_B=(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)^\top
$$

是秩 1 矩阵，可得最优方向与下式同向：

$$
\mathbf{w}^\star \propto S_W^{-1}(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)
$$

通常再做归一化：

$$
\mathbf{w}^\star \leftarrow \frac{\mathbf{w}^\star}{\|\mathbf{w}^\star\|_2}
$$

## 3.7 分类阈值（实战常用）

投影后 $u=\mathbf{w}^{\star\top}\mathbf{x}$。  
简单阈值可取两类投影均值中点：

$$
t=\frac{m_1+m_2}{2}
$$

若先验或代价不平衡，可改成偏置阈值（贝叶斯决策）。

---

## 4. 二分类 FLD 算法步骤（可背）

1. 计算 $\boldsymbol{\mu}_1,\boldsymbol{\mu}_2$  
2. 计算 S_W=S_1+S_2  
3. 计算 $\mathbf{w}^\star=S_W^{-1}(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1$)  
4. 归一化 $\mathbf{w}^\star$  
5. 对样本做投影 $u=\mathbf{w}^{\star\top}\mathbf{x}$ 并设阈值分类

---

## 5. S_W 不可逆时怎么办（页 31）

常见于 n<d（小样本高维）：

- 用伪逆：

$$
\mathbf{w}^\star=S_W^+(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)
$$

- 或正则化：

$$
S_W' = S_W+\epsilon I,\quad
\mathbf{w}^\star=(S_W+\epsilon I)^{-1}(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)
$$

Moore-Penrose 伪逆思路：  
若 $S_W=E\Lambda E^\top$，则

$$
S_W^+=E\Lambda^+E^\top,\quad
\Lambda^+_{ii}=
\begin{cases}
1/\Lambda_{ii}, & \Lambda_{ii}>0\\
0, & \Lambda_{ii}=0
\end{cases}
$$

---

## 6. 多分类 FLD（页 32–33，重点）

## 6.1 多类散度矩阵

设类别 $i=1,\dots,C$，样本数 N_i，总样本数 $N=\sum_i$ N_i。

各类均值 $\boldsymbol{\mu}_i$，全局均值：

$$
\boldsymbol{\mu}=\frac{1}{N}\sum_{i=1}^{C}N_i\boldsymbol{\mu}_i
$$

类内散度：

$$
S_W=\sum_{i=1}^{C}\sum_{y_n=i}
(\mathbf{x}_n-\boldsymbol{\mu}_i)(\mathbf{x}_n-\boldsymbol{\mu}_i)^\top
$$

类间散度：

$$
S_B=\sum_{i=1}^{C}N_i
(\boldsymbol{\mu}_i-\boldsymbol{\mu})
(\boldsymbol{\mu}_i-\boldsymbol{\mu})^\top
$$

总散度：

$$
S_T=
\sum_{n=1}^{N}(\mathbf{x}_n-\boldsymbol{\mu})(\mathbf{x}_n-\boldsymbol{\mu})^\top
=S_W+S_B
$$

## 6.2 优化问题

求投影矩阵 $W=[\mathbf{w}_1,\dots,\mathbf{w}_r$] 使

$$
\max_W \frac{|W^\top S_B W|}{|W^\top S_W W|}
$$

等价于解广义特征值问题：

$$
S_B\mathbf{w}_i=\lambda_i S_W\mathbf{w}_i
$$

取最大特征值对应的前 r 个方向。

## 6.3 为什么最多只有 C-1 个有效方向

因为

$$
\mathrm{rank}(S_B)\le C-1
$$

直观上：C 个类均值相对全局均值的偏移向量线性相关（和为0），独立维数最多 C-1。  
所以 FLD 的判别子空间维度上限是 C-1。

---

## 7. PCA + FLD（Fisherfaces，页 40）

经典流程（人脸识别高频）：

1. 先 PCA 降到 k 维（常取 $k\le$ n-C）  
2. 再在 PCA 子空间内做 FLD

好处：

- 缓解 S_W 奇异问题
- 降噪、降计算量
- 在人脸识别中通常优于纯 PCA（Eigenfaces）

---

## 8. FLD 主要变体（页 34–39）

- 正则化 $FLD$：$S_W+\epsilon I$，提升稳定性
- Kernel FLD：核映射处理非线性可分
- 2D-FLD：直接在图像矩阵上双向投影
- 稀疏 FLD：加 $\ell_1$ 约束做特征筛选
- 鲁棒 FLD：用 median/鲁棒协方差抗异常点
- 增量 FLD：在线更新均值与散度，适合流式数据

---

## 9. 易错点与考试速记

## 9.1 易错点

1. 把 PCA 当分类器：错，PCA 不用标签。  
2. 忘记分母类内散度：只拉开均值不够。  
3. 测试集单独归一化：数据泄漏。  
4. S_W 奇异时硬求逆：应使用伪逆/正则/PCA 预降维。  
5. 多类 FLD 维度上限写错：应是 C-1，不是 C。

## 9.2 公式速查

$$
J(\mathbf{w})=\frac{\mathbf{w}^\top S_B\mathbf{w}}{\mathbf{w}^\top S_W\mathbf{w}}
$$

$$
S_B\mathbf{w}=\lambda S_W\mathbf{w}
$$

$$
\mathbf{w}^\star \propto S_W^{-1}(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)
$$

$$
S_B=\sum_{i=1}^{C}N_i(\boldsymbol{\mu}_i-\boldsymbol{\mu})(\boldsymbol{\mu}_i-\boldsymbol{\mu})^\top,\quad
\mathrm{rank}(S_B)\le C-1
$$

---

## 10. 一句话总结

归一化决定“输入是否可学”，FLD决定“投影是否可分”。  
本章的核心能力是：**会写出 Fisher 准则、会从散度定义推到矩阵形式、会从拉格朗日推到广义特征值问题、知道多类上限为何是 C-1**。
