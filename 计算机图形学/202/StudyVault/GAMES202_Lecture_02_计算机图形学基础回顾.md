# GAMES202 Lecture 02：计算机图形学基础回顾

> 主题：GPU 硬件管线、OpenGL、渲染方程  
> 课程：GAMES202 高质量实时渲染  
> 依据：`GAMES202_Lecture_02.pdf`，重点对应讲义第 5–20、33–35 页。

## 0. 本节课要建立的总图

实时渲染可以先用三层关系理解：

1. **GPU 管线**：硬件如何把三维顶点变成屏幕像素。
2. **OpenGL**：CPU 端如何配置和调用 GPU 管线。
3. **渲染方程**：一个表面点为什么会呈现出当前的亮度和颜色。

可以把它们串起来：

```text
应用程序准备几何体、相机、材质和光源
                ↓
OpenGL 把数据和状态配置到 GPU
                ↓
GPU 管线：顶点 → 三角形 → fragment → 着色 → framebuffer
                ↓
渲染方程决定每个表面点的光照结果
```

---

## 1. GPU 图形管线

### 1.1 管线总览

讲义中的基本流程是：

```text
Application
    ↓
Vertex Processing
    ↓
Triangle Processing
    ↓
Rasterization
    ↓
Fragment Processing
    ↓
Framebuffer Operations
    ↓
Display
```

各阶段的数据形态大致如下：

| 阶段 | 输入/输出 | 核心工作 |
|---|---|---|
| Application | 模型数据、材质、相机、绘制命令 | CPU 准备场景并提交 GPU |
| Vertex Processing | 顶点流 | 对顶点做模型、视图、投影变换等 |
| Triangle Processing | 顶点流 | 组装三角形，确定三角形的几何关系 |
| Rasterization | 三角形流 | 采样三角形覆盖的屏幕区域，生成 fragment |
| Fragment Processing | fragment 流 | 纹理采样、光照、材质计算 |
| Framebuffer Operations | shaded fragment | 深度/模板测试、混合、写入颜色附件 |
| Display | 图像像素 | 显示最终 framebuffer |

### 1.2 顶点处理：M、V、P

顶点通常从模型自己的局部坐标开始，经过三个主要变换：

```text
局部坐标 --M--> 世界坐标 --V--> 观察/相机坐标 --P--> 裁剪空间
```

常写作：

$$
\mathbf{x}_{clip}=PVM\mathbf{x}_{local}
$$

其中：

- `M`（Model）：把模型放到世界中的正确位置、方向和大小；
- `V`（View）：把世界转换到相机坐标系；
- `P`（Projection）：透视投影或正交投影，产生裁剪空间坐标。

在 GLSL 顶点着色器中常见：

```glsl
gl_Position = uProjection * uView * uModel * vec4(aPosition, 1.0);
```

矩阵乘法的书写顺序要结合向量约定。上面采用列向量约定，最右侧的 `M` 最先作用。

### 1.3 三角形处理与光栅化

GPU 通常把顶点组装成三角形。光栅化阶段会判断三角形覆盖了哪些屏幕采样位置，并为每个覆盖位置生成一个 fragment。

需要区分：

- **vertex**：几何输入中的顶点；
- **primitive**：由顶点组装出来的点、线或三角形；
- **fragment**：primitive 覆盖某个屏幕采样位置后生成的候选片段；
- **pixel**：最终写入图像的像素。fragment 不一定最终成为 pixel，因为它可能被深度测试、模板测试等丢弃。

顶点属性（例如颜色、法线、UV）通常会在三角形内部插值，再传给 fragment shader。

### 1.4 Fragment 处理与 framebuffer 操作

Fragment shader 对每个 fragment 计算颜色或其他输出：

```glsl
vec3 N = normalize(vNormal);
vec3 L = normalize(uLightPosition - vWorldPosition);
float NdotL = max(dot(N, L), 0.0);
outColor = vec4(uAlbedo * NdotL, 1.0);
```

之后还要经过 framebuffer 阶段，典型操作包括：

- **深度测试（z-buffer）**：保留更靠近相机的 fragment；
- **模板测试**：根据 stencil buffer 决定是否通过；
- **混合（blending）**：处理透明或颜色叠加；
- **写入颜色/深度附件**：把通过测试的结果写入 framebuffer。

讲义用 Blinn–Phong 说明 fragment 阶段可组合环境光、漫反射和镜面反射：

```text
ambient + diffuse + specular = Blinn-Phong reflectance model
```

### 1.5 GPU 管线的复习重点

- 顶点 shader 主要处理“每个顶点”；fragment shader 主要处理“每个候选片段”。
- 光栅化不是着色，它负责把连续三角形覆盖关系离散到屏幕采样点。
- 深度测试通常在 fragment shader 之后的 framebuffer 阶段完成；不要把“fragment shader 被调用”误认为“该 fragment 一定会显示”。
- 纹理映射、法线插值、光照计算通常发生在 fragment processing 阶段。

---

## 2. OpenGL：用 CPU 配置 GPU 管线

### 2.1 OpenGL 是什么

OpenGL 是一组 API，用来从 CPU 端调用和配置 GPU 图形管线。它本身不是一种编程语言，因此可以由 C/C++ 等语言调用。

讲义强调：OpenGL 的软件接口步骤，与 GAMES101 中手写的软件光栅化器有一一对应关系，只是实际工作由 GPU 完成。

OpenGL 的特点：

- 跨平台；
- C 风格 API；
- 版本和扩展较多，使用复杂；
- 其他图形 API 包括 DirectX、Vulkan 等。

### 2.2 “油画”类比

讲义把一次 OpenGL 渲染类比成油画：

```text
A. 摆放物体/模型
B. 设置画架的位置（相机）
C. 给画架装画布（framebuffer）
D. 在画布上作画（shader + rasterization）
E. 可以用同一个画架画多张画
F. 可以把之前的画作为参考
```

这正对应实时渲染中的 multiple passes：前一个 pass 的纹理结果可以作为后一个 pass 的输入。

### 2.3 A：放置模型

应用程序需要向 GPU 提供：

- 顶点位置；
- 法线；
- 纹理坐标；
- 索引或其他几何组织信息。

这些数据可以放入 **VBO（Vertex Buffer Object）**。它和 `.obj` 文件中的几何数据在概念上相近，但 VBO 是 GPU 可访问的缓冲区对象。

模型变换可以通过矩阵完成。旧式 OpenGL 中会见到 `glTranslate`、`glMultMatrix` 等函数；现代 OpenGL 通常由应用程序或 GLM 构造矩阵，再作为 uniform 传入 shader。

### 2.4 B：设置相机和 framebuffer

设置相机主要涉及：

- view transformation；
- projection transformation；
- framebuffer 的创建或选择。

旧式示例可以用 `gluPerspective` 设置透视投影：

```cpp
gluPerspective(fovy, aspect, zNear, zFar);
```

现代 OpenGL 中通常自己构造投影矩阵，并通过 uniform 传给顶点 shader。

### 2.5 C：指定渲染目标

一次 rendering pass 至少要明确：

1. 使用哪个 framebuffer；
2. 输出到哪些纹理或附件；
3. 使用哪些 shader；
4. 绘制哪些几何体。

输出不一定是屏幕颜色，还可以是：

- 颜色纹理；
- 深度纹理；
- 法线纹理；
- 位置纹理；
- 阴影贴图；
- 其他中间结果。

这就是后续 shadow mapping、G-buffer、后处理等方法的基础。

### 2.6 D：执行绘制

一次绘制的主要顺序是：

```text
提交顶点
    ↓
vertex shader：变换顶点、准备插值数据
    ↓
primitive assembly + rasterization：生成 fragment
    ↓
fragment shader：计算着色结果
    ↓
深度/模板/混合等 framebuffer 操作
    ↓
写入输出纹理或屏幕
```

讲义强调，用户真正定义的核心工作主要集中在 vertex shader 和 fragment shader；其他大量管线操作由 OpenGL/GPU 封装执行。

### 2.7 F：多次渲染（multiple passes）

当一个 pass 的输出被另一个 pass 读取时，就形成多次渲染：

```text
Pass 1：从某个视角/规则生成中间纹理
Pass 2：把中间纹理作为输入，完成最终着色
```

shadow mapping 的典型例子：

```text
Pass 1：从光源视角渲染深度 → shadow map
Pass 2：从相机视角渲染场景，查询 shadow map → 阴影结果
```

每个 pass 都可以有自己的相机、framebuffer、输入纹理和输出纹理。多 pass 的代价是额外的绘制和带宽，但它能把复杂问题拆成多个 GPU 可执行步骤。

### 2.8 OpenGL 一次 pass 的检查清单

遇到“画面不对”时按以下顺序检查：

- 几何体是否真的提交到了 GPU？
- `M/V/P` 是否正确，矩阵顺序是否一致？
- 当前绑定的 framebuffer 是谁？
- 输出纹理格式、尺寸和附件是否正确？
- 当前使用的 shader program 是否正确？
- vertex attribute 和 uniform 是否绑定到预期位置？
- 深度测试、背面剔除、混合状态是否影响结果？
- fragment shader 输出的值是否超出预期范围？

---

## 3. 渲染方程

### 3.1 方程

渲染方程描述表面点处的光传输：

$$
L_o(p,\omega_o)
=
L_e(p,\omega_o)
+
\int_{\mathcal H^2}
f_r(p,\omega_i\rightarrow\omega_o)
L_i(p,\omega_i)
\cos\theta_i\,d\omega_i
$$

一句话记忆：

```text
出射光 = 自发光 + 对所有入射方向的反射光积分
```

### 3.2 符号解释

| 符号            | 含义                          |
| ------------- | --------------------------- |
| `p`           | 表面上的位置                      |
| `ω_o`         | 从 p 指向外部的出射方向，最终通常指向相机      |
| `L_o(p, ω_o)` | p 沿 ω_o 离开的辐射亮度             |
| `L_e(p, ω_o)` | p 自己发出的辐射亮度                 |
| `ω_i`         | 从 p 指向入射光来源的方向（常用方向约定）      |
| `L_i(p, ω_i)` | 从 ω_i 方向到达 p 的入射辐射亮度        |
| `f_r`         | BRDF，描述表面把入射光反射到出射方向的能力     |
| `cos θ_i`     | 入射角余弦，通常为 `max(n · ω_i, 0)` |
| `dω_i`        | 半球方向上的微小立体角                 |
| `H²`          | 以表面法线为中心的上半球                |

### 3.3 方程的物理含义

#### 自发光项

$$
L_e(p,\omega_o)
$$

表示表面自己发光，例如灯泡、屏幕、火焰。普通非发光物体通常令它为 0。

#### 入射光项

$$
L_i(p,\omega_i)
$$

表示从某个方向来的光。它可以来自：

- 直接光源；
- 天空或环境贴图；
- 其他表面反射的间接光；
- 多次反弹后的光。

#### BRDF

$$
f_r(p,\omega_i\rightarrow\omega_o)
$$

BRDF 是材质属性，不是简单的颜色。它回答：

> 从 `ω_i` 来的光，有多少会被重新分配到 `ω_o`？

漫反射、Phong、Blinn-Phong、Cook–Torrance 都可以看成对 BRDF 的不同建模。

#### 余弦项

$$
\cos\theta_i=n\cdot\omega_i
$$

它表示入射光在表面法线方向上的有效投影面积：

- 光线正对表面：余弦接近 1；
- 光线擦着表面：余弦接近 0。

实现中通常写成：

```glsl
float NoL = max(dot(N, L), 0.0);
```

### 3.4 为什么是对半球积分

表面点可能从上半球的每个方向接收光，因此理论上要把所有方向的贡献加起来：

```text
方向 1 的入射光贡献
+ 方向 2 的入射光贡献
+ ...
+ 上半球所有方向的贡献
```

这是连续方向的积分，而不是简单地只看一盏灯。实时渲染为了速度，通常会把积分近似成有限个光源或有限个采样方向的求和：

$$
L_o\approx
L_e+
\sum_k
f_r(p,\omega_k\rightarrow\omega_o)
L_i(p,\omega_k)
\max(n\cdot\omega_k,0)
\Delta\omega_k
$$

### 3.5 实时渲染中的可见性

讲义第 35 页给出了实时渲染中常见的改写：

$$
L_o(p,\omega_o)
=
\int_{\Omega^+}
L_i(p,\omega_i)
f_r(p,\omega_i,\omega_o)
\cos\theta_i
V(p,\omega_i)\,d\omega_i
$$

其中新增：

$$
V(p,\omega_i)
$$

它是可见性函数，通常取 0 或 1：

- `V = 1`：从该方向来的光没有被遮挡；
- `V = 0`：光线到达 p 之前被其他物体挡住。

这正是 shadow mapping 要解决的部分。shadow map 并没有改变渲染方程，而是在近似其中的可见性项。

### 3.6 从渲染方程到直接光照

只考虑一个点光源时，积分可以近似为单项：

$$
L_o
\approx
f_r(p,\omega_i\rightarrow\omega_o)
L_i(p,\omega_i)
\max(n\cdot\omega_i,0)
V(p,\omega_i)
$$

对应 Shader 结构：

```glsl
vec3 L = normalize(lightPosition - worldPosition);
vec3 V = normalize(cameraPosition - worldPosition);
vec3 N = normalize(worldNormal);

float NoL = max(dot(N, L), 0.0);
float visibility = shadowTest(worldPosition);
vec3 fr = evaluateBRDF(N, L, V, material);
vec3 Lo = fr * incomingLight * NoL * visibility;
```

要注意：

- `fr` 负责材质反射特性；
- `incomingLight` 是入射光；
- `NoL` 是余弦权重；
- `visibility` 是遮挡判断。

### 3.7 渲染方程的递归性

一个点的入射光可能来自另一个表面点，而另一个表面点的出射光又由渲染方程决定。因此全局光照天然具有递归结构：

```text
p 接收 q 的光
q 接收 r 的光
r 接收光源的光
```

直接光照只计算光源到当前点的一次路径；一 bounce、二 bounce 全局光照会继续考虑光在表面之间的反弹。

---

## 4. 三个单元如何串联

### GPU 管线回答“怎么算出来”

它说明数据如何从顶点变成屏幕上的 fragment，以及深度测试和 framebuffer 如何参与最终输出。

### OpenGL 回答“如何配置和组织”

它把几何体、相机、framebuffer、纹理、shader 和绘制命令交给 GPU，并允许通过 multiple passes 组织复杂算法。

### 渲染方程回答“应该算什么”

它定义表面点最终出射光的物理目标：自发光加上半球内所有入射光经过 BRDF 和余弦项加权后的积分。

三者关系可以记成：

```text
渲染方程：目标是什么
OpenGL：如何配置一次或多次 GPU 计算
GPU 管线：硬件如何执行这些计算
```

---

## 5. 复习自测题

1. 为什么顶点 shader 不能直接决定屏幕上每个像素的最终颜色？
2. fragment 和 pixel 有什么区别？
3. `M`、`V`、`P` 分别解决什么问题？
4. 一个 OpenGL rendering pass 需要明确哪些输入、状态和输出？
5. 为什么 shadow mapping 通常需要至少两个 pass？
6. 渲染方程中的 `L_e` 与积分项分别表示什么？
7. BRDF、`cos θ_i` 和可见性函数各自承担什么作用？
8. 为什么实时渲染会把半球积分近似成有限光源求和？
9. shadow map 主要是在近似渲染方程中的哪一项？
10. 如果画面全黑，应该如何区分是几何、矩阵、光照、纹理还是 framebuffer 状态的问题？

## 6. 一页速记

```text
GPU：顶点 → 三角形 → 光栅化 → fragment → 深度/混合 → framebuffer

OpenGL pass：
几何体 + 相机/MVP + framebuffer/纹理 + vertex shader/fragment shader → Render

渲染方程：
Lo = Le + ∫ fr · Li · cosθ · dω

实时渲染常显式加入：
V(p, wi)，表示光源到表面点的可见性

Shadow mapping：
光源 pass 生成深度 → 相机 pass 查询深度 → 判断 V
```

