---
source_pdf: GAMES101_Lecture_07.pdf, GAMES101_Lecture_08.pdf
part: 7-8
keywords: Blinn-Phong, diffuse, specular, ambient, shading model
---

# Blinn-Phong反射模型（★★★）

#computer-graphics #shading #blinn-phong #concept

## 概览表（一目了然）
| 条目 | 要点 |
|------|------|
| 漫反射 | Ld = kd(I/r²)max(0,n·l) |
| 高光 | Ls = ks(I/r²)max(0,n·h)ᵖ |
| 环境光 | La = ka·Ia（常数） |
| 半程向量 | h = normalize(v+l) |

## 漫反射 (Diffuse)

光均匀地向各方向反射，与观察方向无关：

$$L_d = k_d \cdot \frac{I}{r^2} \cdot \max(0, \mathbf{n} \cdot \mathbf{l})$$

- **kd**：漫反射系数（颜色/albedo）
- **I/r²**：点光源到着色点的能量（距离平方衰减）
- **max(0, n·l)**：Lambert 余弦定律，背面截断为 0

## 高光 (Specular)

观察方向接近镜面反射方向时看到高光：

$$L_s = k_s \cdot \frac{I}{r^2} \cdot \max(0, \mathbf{n} \cdot \mathbf{h})^p$$

- **h = normalize(v + l)**：半程向量（Blinn 的改进，避免计算反射方向）
- **p**：高光指数，通常 100-200，越大高光越集中

> [!tip] Phong vs Blinn-Phong
> Phong 用反射向量 R 和 v 的夹角；Blinn-Phong 用 n 和 h 的夹角，计算更简单。

## 环境光 (Ambient)

保证场景不全黑的常数近似：

$$L_a = k_a \cdot I_a$$

> [!warning] 这是大胆近似，与真实环境光照无关。

## 总和

$$L = L_a + L_d + L_s = k_aI_a + k_d\frac{I}{r^2}\max(0,n \cdot l) + k_s\frac{I}{r^2}\max(0,n \cdot h)^p$$

---

## 考试/测试常见模式
| 场景/关键词 | 答案 |
|-------------|------|
| "h 怎么算" | **normalize(v+l)** |
| "p 越大" | **高光越小越集中** |
| "漫反射与观察方向" | **无关** |
| "I/r²的含义" | **点光源能量随距离²衰减** |

## 相关笔记
- [[着色频率与图形管线]]
- [[Z-Buffer深度缓冲]]
- [[纹理映射与重心坐标]]
