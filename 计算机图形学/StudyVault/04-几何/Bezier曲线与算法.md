---
source_pdf: GAMES101_Lecture_11.pdf
part: 11
keywords: Bezier curve, De Casteljau, Bernstein polynomial, continuity, B-spline
---

# Bezier曲线与算法（★★★）

#computer-graphics #geometry #bezier #concept

## 概览表（一目了然）
| 条目 | 要点 |
|------|------|
| De Casteljau | 递归线性插值，t∈[0,1] |
| Bernstein基 | bⁿᵢ(t) = C(n,i)tⁱ(1-t)ⁿ⁻ⁱ |
| 经过点 | 仅首末控制点 |
| 仿射不变性 | ✓（投影不变性 ✗） |

## De Casteljau 算法

给定 n+1 个控制点，对 t∈[0,1]：
1. 相邻控制点之间做线性插值（比例 t）
2. 得到 n 个新点
3. 递归直到只剩 1 个点 → 曲线上的点

## Bernstein 多项式

$$b(t) = \sum_{i=0}^{n} B_i \cdot b_i^n(t), \quad b_i^n(t) = \binom{n}{i} t^i(1-t)^{n-i}$$

## Bezier 曲线性质

| 性质 | 说明 |
|------|------|
| 端点插值 | 经过 b₀ 和 bₙ |
| 端点切线 | b'(0) = n(b₁-b₀)，b'(1) = n(bₙ-bₙ₋₁) |
| 仿射不变性 | 变换控制点等价于变换曲线 |
| 凸包性质 | 曲线在控制点的凸包内 |

> [!warning] 仿射变换不变性 ✓，投影变换不变性 ✗

## 连续性

- **C0**：位置连续（端点重合）
- **C1**：一阶导连续（aₙ = b₀ 且 aₙ - aₙ₋₁ = b₁ - b₀）
- **G1**：切线方向连续（不要求大小相等）

## 分段 Bezier 曲线

高阶 Bezier 难以控制 → 每 4 个点一段 Cubic Bezier，保证 C1 连续。

## B-spline 简介

具有**局部性**：移动一个控制点只影响附近曲线段。Bezier 不具备此性质。

---

## 考试/测试常见模式
| 场景/关键词 | 答案 |
|-------------|------|
| "经过中间控制点吗" | **不经过** |
| "端点切线方向" | **n(b₁-b₀)** |
| "C1连续条件" | **端点重合 + 一阶导相等** |
| "Bezier vs B-spline" | **B-spline有局部性** |

## 相关笔记
- [[隐式与显式几何表示]]
- [[曲面与网格表示]]
- [[网格细分与简化]]
