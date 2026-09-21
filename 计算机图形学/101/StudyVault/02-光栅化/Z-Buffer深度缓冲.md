---
source_pdf: GAMES101_Lecture_06.pdf, GAMES101_Lecture_07.pdf
part: 6-7
keywords: z-buffer, depth buffer, visibility, occlusion, painter algorithm
---

# Z-Buffer深度缓冲（★★）

#computer-graphics #rasterization #z-buffer #concept

## 概览表（一目了然）
| 条目 | 要点 |
|------|------|
| 问题 | 确定哪些三角形应该被看到 |
| 画家算法 | 从远到近画，有循环遮挡缺陷 |
| Z-Buffer | 逐像素维护最小深度，O(n) |
| 局限 | 不能处理透明物体 |

## 画家算法

从远到近画，后画覆盖先画。**致命缺陷**：存在无法排序的循环遮挡。

## Z-Buffer 算法

```
初始化 zbuffer[x][y] = +∞
对每个三角形 T:
    对 T 覆盖的每个像素 (x,y):
        z = T 在此处的深度
        if z < zbuffer[x][y]:
            zbuffer[x][y] = z
            framebuffer[x][y] = 颜色
```

| 特性 | 说明 |
|------|------|
| 时间复杂度 | O(n)，与三角形数量线性 |
| 顺序无关 | 处理顺序不影响结果 |
| 可并行 | 每个像素独立处理 |

> [!warning] Z-Buffer 不能处理透明物体
> 透明物体需混合前后颜色。解决：先渲染不透明，再按深度排序渲染透明物体。

---

## 考试/测试常见模式
| 场景/关键词 | 答案 |
|-------------|------|
| "Z-Buffer 初始值" | **+∞（每次取 min）** |
| "Z-Buffer 复杂度" | **O(n)** |
| "画家算法问题" | **循环遮挡无法排序** |
| "不能处理什么" | **透明物体** |

## 相关笔记
- [[视口变换与三角形光栅化]]
- [[反走样与采样理论]]
- [[Blinn-Phong反射模型]]
