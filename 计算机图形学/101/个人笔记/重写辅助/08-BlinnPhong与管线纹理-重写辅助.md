# 第 08 章：Blinn-Phong、着色频率与图形管线、纹理引入

## 资料对齐

- **个人笔记**：`## 08`（无标题）
- **课件**：`Lecture_07` 尾 + `Lecture_08` Shading 2
- **StudyVault**：`Blinn-Phong反射模型.md`、`着色频率与图形管线.md`
- **核心目标**：补全 Blinn-Phong；分清 Flat/Gouraud/Phong；记住实时管线骨架；纹理=表面属性的 2D 参数化

---

## 原笔记逐点完善

### 小点 1：高光；半程向量 h；环境光；Blinn-Phong

#### 课件依据
高光：靠近镜面反射方向更亮；Blinn 用 \(h=\mathrm{normalize}(v+l)\)，看 \(n\cdot h\)。环境光：加常数，假的、近似。总和 Ambient+Diffuse+Specular。

#### StudyVault
\[
L_s=k_s\frac{I}{r^2}\max(0,n\cdot h)^p,\quad
L_a=k_a I_a,\quad
L=L_a+L_d+L_s
\]
\(h=\mathrm{normalize}(v+l)\)（不是 \((v+l)/2\) 未归一化就当单位向量用）。

#### 冲突标注
- 原笔记：「v 和 r 接近 → h 和 n 接近」——正确直觉。
- Vault：h 必须 normalize。
- **采用**：先单位化再点积；p 越大高光越尖（常 100–200，经验）。

#### 手写建议
画反射方向 r 与半程 h 对比；写三项求和。

#### 易错点
\(h=(v+l)/2\) 却不归一化；把 ambient 当成物理正确 GI。

---

### 小点 2：着色频率 Flat / Gouraud / Phong

#### 课件依据
Flat：每面一个法线/一色。Gouraud：顶点着色再插值颜色；顶点法线常取邻面平均。Phong shading：插值法线，每像素完整着色（≠ Blinn-Phong 反射模型同名易混）。模型够密时前两者差异变小。

#### 完善后的学习笔记
| 名称 | 在哪着色 | 插值什么 |
|------|----------|----------|
| Flat | 每三角形 | 无（或面中心一次） |
| Gouraud | 顶点 | 颜色 |
| Phong shading | 像素 | 法线 |

#### 易错点
「Phong shading」与「Blinn-Phong reflectance」混名。

---

### 小点 3：实时图形管线

#### 课件依据
顶点定义 → 哪些点成三角 → 变换/投影 → 光栅化（含 Z）→ 着色（vertex/fragment shader）→ 输出。现代可编程：vertex / fragment 阶段。

#### 完善后的学习笔记
保留你的链条，补「MVP+Viewport」插入点：
**点 → MVP/视口 → 三角形 → 光栅化+Z → 着色 → 帧缓冲**

#### 手写建议
画一条横向管线，标注你已学章节号。

---

### 小点 4：纹理映射引入

#### 课件依据
在表面不同位置定义不同属性；表面参数本质 2D；三角形对应纹理三角形；uv∈[0,1]。

#### 完善后的学习笔记
纹理 = 查表得到 \(k_d\) 等。细节过滤留给第 09 章。

---

## 本章重写版笔记

# 08. 着色（2）：Blinn-Phong、频率、管线、纹理引入

## 1. Blinn-Phong
\[
L=k_a I_a+k_d\frac{I}{r^2}\max(0,n\cdot l)+k_s\frac{I}{r^2}\max(0,n\cdot h)^p
\]
\(h=\mathrm{normalize}(v+l)\)。Ambient 是假的常数补光。

## 2. 着色频率
Flat / Gouraud（插颜色）/ Phong shading（插法线）。注意与 Blinn-Phong 模型别名冲突。

## 3. 实时管线
顶点→图元组装→变换投影→光栅化+深度→片元着色→输出。

## 4. 纹理引入
uv 参数化；纹理存漫反射等属性；下一章插值与过滤。

---

## 本章自测题

1. 写出完整 Blinn-Phong 并解释 h、p。
2. Ambient 为什么说是 fake？
3. Gouraud 与 Phong shading 差在插值对象？
4. 为何「Phong」一词容易混？
5. 画出实时管线主阶段。
6. 纹理 uv 的范围与几何含义？
7. 局部着色 + 纹理能否做出 bump 感？（预告法线贴图）
8. 模型很密时为何 Flat 也可能够用？

## 本章待确认

- 原笔记无标题：本文件标题供重写用；你手写可用「08 Blinn-Phong 与管线」
