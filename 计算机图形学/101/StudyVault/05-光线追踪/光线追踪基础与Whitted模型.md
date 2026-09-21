---
source_pdf: GAMES101_Lecture_13.pdf
part: 13
keywords: ray tracing, Whitted, recursive, reflection, refraction, ray casting
---

# 光线追踪基础与Whitted模型（★★★）

#computer-graphics #ray-tracing #whitted #concept

## 概览表（一目了然）
| 条目 | 要点 |
|------|------|
| 动机 | 光栅化难处理全局效果 |
| 光线假设 | 直线/不碰撞/光路可逆 |
| Ray Casting | 每像素发一条光线 |
| Whitted | 递归反射+折射+阴影光线 |

## 为什么需要光线追踪？

光栅化难以处理：软阴影、多次反射/折射、焦散、间接光照。

## 光线三个假设
1. 光沿直线传播
2. 光线不互相碰撞
3. **光路可逆**（从眼睛追踪和从光源追踪等价）

## Ray Casting (Appel 1968)
1. 从眼睛通过每像素发射一条光线
2. 找最近交点
3. 向光源发射 shadow ray 判断阴影
4. 计算该点局部着色

## Whitted 风格递归光线追踪

在每个交点处：
- **反射光线**（镜面反射方向）→ 递归
- **折射光线**（Snell 定律方向）→ 递归
- **阴影光线**（到光源）→ 判断可见性

最终颜色 = 局部着色 + 反射颜色权重 + 折射颜色权重

```
primary ray → 最近交点
    ├── shadow ray → 光源（判断阴影）
    ├── reflected ray → 递归（镜面反射）
    └── refracted ray → 递归（折射/透射）
```

---

## 考试/测试常见模式
| 场景/关键词 | 答案 |
|-------------|------|
| "光路可逆" | **从眼追踪等价于从光追踪** |
| "Whitted 递归终止" | **达到最大深度或能量低于阈值** |
| "shadow ray" | **从交点到光源，判断遮挡** |

## 相关笔记
- [[光线求交与加速结构]]
- [[渲染方程与全局光照]]
