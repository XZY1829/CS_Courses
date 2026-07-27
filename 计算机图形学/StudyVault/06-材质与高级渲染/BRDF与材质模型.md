---
source_pdf: GAMES101_Lecture_17.pdf
part: 17
keywords: BRDF, diffuse, specular, Fresnel, microfacet, Snell law
---

# BRDF与材质模型（★★★）

#computer-graphics #materials #brdf #microfacet #concept

## 概览表（一目了然）
| 条目 | 要点 |
|------|------|
| Material = BRDF | 定义表面如何反射光 |
| 完美漫反射 | fr = ρ/π |
| 镜面反射 | 出射方向关于法线对称 |
| Fresnel | 掠射角反射率趋近 1 |
| 微表面 | fr = F·G·D / (4cosθi·cosθo) |

## 完美漫反射 (Lambertian)

$$f_r = \frac{\rho}{\pi}$$

推导：能量守恒 ∫fr·cosθ dω = ρ → fr·π = ρ。ρ 为 albedo ∈[0,1]。

## 镜面反射与折射

**镜面反射**：ωo 关于法线 n 对称于 ωi。

**折射（Snell 定律）**：$n_1\sin\theta_1 = n_2\sin\theta_2$

**全内反射**：当 $(n_1/n_2)^2\sin^2\theta_i > 1$ 时无折射（光密→光疏）。

## Fresnel 项

反射率随入射角变化：掠射时趋近 1（如水面远处像镜子）。

**Schlick 近似**：
$$R(\theta) = R_0 + (1-R_0)(1-\cos\theta)^5, \quad R_0 = \left(\frac{n_1-n_2}{n_1+n_2}\right)^2$$

## 微表面模型 (Microfacet)

远看是材质（flat），近看是几何（rough）。

$$f_r = \frac{F(\mathbf{i},\mathbf{h}) \cdot G(\mathbf{i},\mathbf{o},\mathbf{h}) \cdot D(\mathbf{h})}{4(\mathbf{n}\cdot\mathbf{i})(\mathbf{n}\cdot\mathbf{o})}$$

| 项 | 含义 |
|----|------|
| **D(h)** | 法线分布函数(NDF)：微面元法线在 h 方向的概率 |
| **G(i,o,h)** | 几何遮蔽/阴影：微面元间互相遮挡 |
| **F(i,h)** | Fresnel 项：入射角反射率 |

**粗糙度**：D 函数越宽 → 漫反射越强；越窄 → 镜面越强。

## 各向异性 BRDF

BRDF 随方位角旋转变化（如拉丝金属、尼龙）。

---

## 考试/测试常见模式
| 场景/关键词 | 答案 |
|-------------|------|
| "漫反射 fr" | **ρ/π** |
| "全反射条件" | **(n₁/n₂)²sin²θ > 1** |
| "Fresnel掠射" | **反射率→1** |
| "D 越宽" | **越粗糙/漫反射** |

## 相关笔记
- [[辐射度量学]]
- [[高级光传输方法]]
- [[高级材质与外观]]
