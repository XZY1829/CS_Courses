# GAMES202 Lecture 03：实时阴影（一）

> 主题：Shadow Mapping、阴影伪影、PCF、PCSS  
> 课程：GAMES202 高质量实时渲染  
> 依据：`GAMES202_Lecture_03.pdf`，共 43 页  
> 本讲边界：重点讲普通 Shadow Mapping、PCF 与 PCSS；VSSM、MIPMAP/SAT 和 Moment Shadow Mapping 只在结尾预告，将在下一讲展开。

## 0. 本节课要建立的总图

Lecture 03 的主线可以概括为：

```text
渲染方程中的可见性 V
        ↓
普通 Shadow Mapping：一次深度比较，得到 0/1 可见性
        ↓
问题：self-occlusion、detached shadow、aliasing
        ↓
PCF：多次深度比较，再平均比较结果
        ↓
问题：固定滤波核只能平滑边缘，不能表达真实半影变化
        ↓
PCSS：搜索 blocker → 估计半影 → 使用可变大小的 PCF
```

三种方法解决的问题不同：

| 方法 | 核心问题 | 核心操作 | 典型结果 |
|---|---|---|---|
| Shadow Mapping | 某个点能否看见光源 | 比较一次光源空间深度 | 硬阴影，结果为 0 或 1 |
| PCF | Shadow Map 离散采样导致的边缘锯齿 | 比较多个深度，再平均比较结果 | 边缘平滑，但软硬程度通常固定 |
| PCSS | 面积光源产生的可变半影 | 搜索遮挡物并动态改变 PCF 核大小 | 接触处硬、远处软 |

---

## 1. 普通 Shadow Mapping

### 1.1 它在解决什么问题

在渲染方程中，阴影由可见性函数决定：

$$
V(p,\omega_i)=
\begin{cases}
1, & \text{点 }p\text{ 沿 }\omega_i\text{ 能看到光源}\\
0, & \text{点 }p\text{ 沿 }\omega_i\text{ 被遮挡}
\end{cases}
$$

Shadow Mapping 的目标，就是用光源视角下的深度图近似这个可见性函数。

核心观察是：

> 如果点 `p` 是光源沿某条视线最先看到的表面，那么它能直接看到光源；如果同一方向上存在比 `p` 更靠近光源的表面，那么 `p` 被遮挡。

### 1.2 两个渲染 Pass

Shadow Mapping 是一个典型的 two-pass 算法。

#### Pass 1：从光源视角渲染

只输出深度，不需要最终颜色：

```text
场景几何体
    ↓
光源的 View / Projection
    ↓
深度测试
    ↓
Shadow Map：每个 texel 保存光源看到的最近深度
```

把 Shadow Map 中坐标 `(u,v)` 的深度记作：

$$
z_{sm}(u,v)
$$

它表示光源沿这个方向遇到的第一个表面。

#### Pass 2：从相机视角渲染

对于相机当前可见的每个 fragment：

1. 取得世界空间位置 `p`；
2. 把 `p` 重新投影到光源空间；
3. 得到 Shadow Map 坐标 `(u,v)` 和当前点的光源空间深度 `z_p`；
4. 用 `z_p` 与 `z_sm(u,v)` 比较。

光源空间变换可以写成：

$$
\mathbf p_{lightClip}=P_LV_L\mathbf p_{world}
$$

透视除法后得到光源 NDC：

$$
\mathbf p_{lightNDC}
=
\frac{\mathbf p_{lightClip}}{w_{lightClip}}
$$

再把 NDC 的 `[-1,1]` 映射到纹理坐标 `[0,1]`，即可查询 Shadow Map。

### 1.3 深度比较

忽略数值误差时，阴影判断为：

$$
V(p)=
\begin{cases}
1, & z_p\le z_{sm}(u,v)\\
0, & z_p>z_{sm}(u,v)
\end{cases}
$$

直观解释：

```text
z_p 与 Shadow Map 深度相同
→ 当前点就是光源最先看到的表面
→ 可见，V = 1

z_p 比 Shadow Map 深度更远
→ 前面存在另一个表面
→ 被遮挡，V = 0
```

### 1.4 Shadow Mapping 的性质

它属于 image-space 算法：

- 不需要显式求光线与三角形的解析交点；
- 可以复用 GPU 的光栅化和深度测试；
- 算法复杂度主要与输出分辨率和绘制规模有关；
- Shadow Map 是离散图像，因此会产生精度、采样和走样问题。

讲义中特别展示了两个阴影带来的视觉结果：

- 被遮挡区域不会出现该直接光源产生的镜面高光；
- 曲面、人物等复杂几何体可以自然地相互投影。

这里的“没有镜面高光”只针对被遮挡的直接光源贡献；环境光、间接光或其他光源仍可能产生亮度。

---

## 2. Shadow Mapping 的典型问题

### 2.1 Self-occlusion：物体错误地遮挡自己

理论上，同一表面在两个 pass 中应满足：

$$
z_p=z_{sm}
$$

但实际中会受到以下因素影响：

- Shadow Map 分辨率有限；
- 一个 texel 覆盖了一小块连续表面；
- 光源 pass 和相机 pass 的光栅化位置不同；
- 深度插值与深度缓冲精度有限；
- 倾斜表面在一个 texel 内存在明显深度变化。

于是可能出现：

$$
z_p=z_{sm}+\varepsilon
$$

严格比较会把这个微小误差当成真实遮挡，产生密集的黑色条纹或噪点，即：

- self-occlusion；
- shadow acne；
- 自遮挡错误或阴影痤疮。

#### 什么时候最严重

当光线接近表面的切线方向，也就是光线几乎擦着表面传播时最严重。

原因是表面越倾斜，同一个 Shadow Map texel 覆盖范围内的深度变化越大：

```text
表面正对光源 → 一个 texel 内的深度变化小
表面侧对光源 → 一个 texel 内的深度变化大
```

### 2.2 Bias：给比较留出误差空间

工程上常把判断改为：

$$
V(p)=
\begin{cases}
1, & z_p\le z_{sm}+b\\
0, & z_p>z_{sm}+b
\end{cases}
$$

其中 `b` 是 bias。

只要数值误差满足：

$$
\varepsilon<b
$$

同一表面就不会被误判为阴影。

由于倾斜表面的误差更大，实际常使用 variable bias 或 slope-scale bias：

```text
bias = 常量项 + 与表面斜率有关的项
```

### 2.3 Detached shadow：bias 太大的副作用

如果 bias 太大，真实存在的浅层遮挡也会被忽略：

```text
本应在阴影中的接触区域
→ 因深度差小于 bias
→ 被错误判断为受光
```

结果是阴影与物体接触处出现空隙，看起来像物体悬浮。这称为：

- detached shadow；
- Peter Panning；
- 阴影脱离或阴影悬浮。

因此 bias 存在基本权衡：

```text
bias 太小 → shadow acne
bias 太大 → Peter Panning
```

### 2.4 Second-depth Shadow Mapping

普通 Shadow Map 只记录光线遇到的第一个深度：

$$
z_1=\text{first depth}
$$

Second-depth Shadow Mapping 还记录第二个交点：

$$
z_2=\text{second depth}
$$

然后使用二者中点作为比较参考：

$$
z_{mid}=\frac{z_1+z_2}{2}
$$

对于封闭物体，一条光线通常先进入物体再离开物体：

```text
光源 → z1（进入）→ 物体内部 → z2（离开）
```

中点位于物体内部，可以比人工常量 bias 更自然地区分前后表面。

它的局限是：

- 要求物体 watertight，即封闭且没有裂缝；
- 薄片、单面模型或开口网格可能没有可靠的第二个交点；
- 第二个深度可能来自其他物体；
- 需要额外存储和渲染开销，工程收益未必值得。

### 2.5 Aliasing：离散 Shadow Map 的走样

Shadow Map 的一个 texel 可能投影到相机画面的多个 pixel：

```text
低分辨率的光源空间深度图
        ↓ 投影
高分辨率的相机画面
        ↓
阴影边缘出现台阶和锯齿
```

提高 Shadow Map 分辨率可以缓解问题，但会增加显存、带宽和渲染成本，也不能消除所有投影不均匀问题。

---

## 3. Shadow Mapping 背后的数学近似

### 3.1 通用积分近似

讲义先给出：

$$
\int_{\Omega}f(x)g(x)\,dx
\approx
\frac{\int_{\Omega}f(x)\,dx}{\int_{\Omega}dx}
\cdot
\int_{\Omega}g(x)\,dx
$$

定义 `f` 在区域上的平均值：

$$
\bar f
=
\frac{\int_{\Omega}f(x)\,dx}{\int_{\Omega}dx}
$$

那么近似式就是：

$$
\int_{\Omega}f(x)g(x)\,dx
\approx
\bar f\int_{\Omega}g(x)\,dx
$$

也就是用 `f` 的平均值代替积分内部随位置变化的 `f(x)`。

从统计角度看，近似误差来自被忽略的相关项：

$$
\int_{\Omega}fg
=
|\Omega|
\left(
\mathbb E[f]\mathbb E[g]
+
\operatorname{Cov}(f,g)
\right)
$$

近似相当于忽略：

$$
|\Omega|\operatorname{Cov}(f,g)
$$

因此，当一个函数近似常量、两个函数相关性较弱，或者有效积分区域很小时，这个近似更准确。

### 3.2 代入带可见性的渲染方程

显式加入可见性的渲染方程为：

$$
L_o(p,\omega_o)
=
\int_{\Omega^+}
L_i(p,\omega_i)
f_r(p,\omega_i,\omega_o)
\cos\theta_i
V(p,\omega_i)
\,d\omega_i
$$

令：

$$
f(\omega_i)=V(p,\omega_i)
$$

$$
g(\omega_i)
=
L_i(p,\omega_i)
f_r(p,\omega_i,\omega_o)
\cos\theta_i
$$

可以得到近似：

$$
L_o(p,\omega_o)
\approx
\frac{
\int_{\Omega^+}V(p,\omega_i)\,d\omega_i
}{
\int_{\Omega^+}d\omega_i
}
\cdot
\int_{\Omega^+}
L_i(p,\omega_i)
f_r(p,\omega_i,\omega_o)
\cos\theta_i
\,d\omega_i
$$

可以记成：

```text
带阴影的光照
≈ 平均可见率 × 不考虑遮挡时的光照
```

其中平均可见率为：

$$
\bar V(p)
=
\frac{
\int_{\Omega^+}V(p,\omega_i)\,d\omega_i
}{
\int_{\Omega^+}d\omega_i
}
$$

如果 `V` 是二值函数，那么：

$$
0\le \bar V\le 1
$$

- `0`：所有有效方向都被遮挡；
- `1`：所有有效方向都可见；
- `(0,1)`：只看见部分光源，对应半影。

### 3.3 近似何时更准确

讲义给出两个条件。

#### Small support

光照只来自很小的有效方向范围，例如点光源或方向光。

这时可以近似只查询一个光源方向：

$$
L_o
\approx
V(p,\omega_l)
L_{unshadowed}
$$

普通 Shadow Mapping 的一次二值深度比较就比较合理。

#### Smooth integrand

以下部分随方向变化比较平滑：

$$
L_i f_r\cos\theta_i
$$

典型情况包括：

- 漫反射 BRDF；
- 近似恒定辐射亮度的面积光源；
- 没有非常尖锐的高光峰值。

如果材质高度镜面化，只有很窄的方向贡献巨大，而且这些方向的遮挡情况与其他方向不同，那么“平均可见率乘总光照”的误差会增大。

---

## 4. 从硬阴影到软阴影

### 4.1 点光源产生硬阴影

理想点光源只有一个发光位置。对于接收点，光源只有两种状态：

```text
看得见点光源 → V = 1
看不见点光源 → V = 0
```

阴影边界从 0 直接跳到 1，因此是硬边界。

### 4.2 面积光源产生软阴影

面积光源具有多个发光位置。一个接收点可能只看到光源的一部分：

```text
光源的一些位置被挡住
光源的另一些位置仍然可见
→ 0 < 平均可见率 < 1
```

因此产生三个区域：

| 区域 | 光源可见情况 | 可见率 |
|---|---|---:|
| Umbra，本影 | 整个光源都被挡住 | `0` |
| Penumbra，半影 | 只看见部分光源 | `(0,1)` |
| 完全受光区 | 整个光源都可见 | `1` |

软阴影不是简单的图像模糊，而是对面积光源上不同采样位置的可见性进行平均。

---

## 5. PCF：Percentage Closer Filtering

### 5.1 PCF 的目标

PCF 主要用于缓解 Shadow Map 阴影边缘的走样。

它的核心不是模糊 Shadow Map 深度，而是：

> 对邻域内的每个深度分别完成阴影比较，再平均这些二值比较结果。

### 5.2 为什么不能先平均深度

假设邻域深度为 `d_i`，错误思路是：

$$
V
=
\operatorname{step}
\left(
z_p,
\frac{1}{N}\sum_i d_i
\right)
$$

无论平均深度是多少，最后只比较一次，所以结果仍然是 0 或 1；而且平均深度不一定对应任何真实表面。

PCF 使用的是：

$$
V_{PCF}(p)
=
\frac{1}{N}
\sum_{i=1}^{N}
\operatorname{step}(z_p,z_{sm,i}+b)
$$

因此：

$$
\operatorname{step}
\left(z_p,\operatorname{avg}(z_{sm,i})\right)
\ne
\operatorname{avg}
\left(\operatorname{step}(z_p,z_{sm,i})\right)
$$

可以简记为：

```text
错误：先过滤深度，再比较
PCF：先分别比较，再过滤比较结果
```

### 5.3 计算示例

对一个 fragment 查询 `3×3` 的 Shadow Map 邻域，得到：

```text
1 0 1
1 0 1
1 1 0
```

共有 6 个样本可见：

$$
V_{PCF}=\frac{6}{9}\approx0.667
$$

最终直接光照可以乘以这个可见率：

$$
L_{direct}^{shadowed}
\approx
V_{PCF}L_{direct}^{unshadowed}
$$

### 5.4 为什么边缘变平滑

```text
阴影内部：邻域几乎全为 0 → 平均值接近 0
受光区域：邻域几乎全为 1 → 平均值接近 1
阴影边缘：邻域同时出现 0 和 1 → 平均值位于 0 和 1 之间
```

因此 PCF 在阴影边缘产生灰度过渡，减少锯齿。

### 5.5 PCF 为什么不等于真实软阴影

固定大小的 PCF 核只是在 Shadow Map 邻域中进行图像空间过滤：

- 没有显式考虑面积光源的大小；
- 没有考虑 blocker 与 receiver 的相对距离；
- 阴影各处通常具有相似的模糊宽度；
- 更像抗锯齿，而不是物理半影模拟。

滤波核越小，阴影越锐利；滤波核越大，阴影越模糊。接下来的问题是：

> 每个位置应该使用多大的滤波核？滤波大小是否应该处处相同？

这正是 PCSS 要解决的问题。

---

## 6. PCSS：Percentage Closer Soft Shadows

### 6.1 核心观察：接触处硬，远处软

现实中的面积光源阴影具有 contact hardening 特征：

```text
blocker 靠近 receiver
→ 不同光源位置投下的边界差异小
→ 半影窄，阴影较硬

blocker 远离 receiver
→ 不同光源位置投下的边界差异大
→ 半影宽，阴影较软
```

因此滤波大小应该与 blocker 和 receiver 的相对距离有关。

### 6.2 半影宽度估计

根据相似三角形，讲义给出：

$$
w_{penumbra}
=
(d_{receiver}-d_{blocker})
\frac{w_{light}}{d_{blocker}}
$$

其中：

- `w_light`：面积光源宽度；
- `d_blocker`：光源到遮挡物的距离；
- `d_receiver`：光源到接收面的距离；
- `w_penumbra`：接收面上的半影宽度。

当 blocker 与 receiver 接触时：

$$
d_{receiver}\approx d_{blocker}
\Rightarrow
w_{penumbra}\approx0
$$

当二者距离增大时，半影宽度随之增大。

实际场景可能有多个 blocker，因此使用邻域中 blocker 的平均投影深度：

$$
\bar d_{blocker}
=
\frac{1}{N_B}
\sum_{i\in B}d_i
$$

### 6.3 PCSS 的三个步骤

#### Step 1：Blocker Search

对于当前 receiver fragment：

1. 计算 Shadow Map 坐标和 receiver 深度 `d_receiver`；
2. 在附近区域采样 Shadow Map；
3. 找出满足以下条件的样本：

$$
d_i+b<d_{receiver}
$$

这些样本比 receiver 更靠近光源，因此被视为 blocker；

4. 对 blocker 深度求平均，得到 `d_blocker_avg`。

如果没有找到 blocker，可以直接认为当前点完全受光：

$$
V=1
$$

#### Step 2：Penumbra Estimation

利用平均 blocker 深度估计半影：

$$
w_{penumbra}
\approx
(d_{receiver}-\bar d_{blocker})
\frac{w_{light}}{\bar d_{blocker}}
$$

再把世界或光源空间中的半影宽度换算为 Shadow Map texel 半径。

#### Step 3：Variable-size PCF

使用估计出的滤波半径执行 PCF：

```text
半影小 → PCF 核小 → 阴影较硬
半影大 → PCF 核大 → 阴影较软
```

### 6.4 Blocker Search 应搜索多大区域

讲义指出，搜索区域可以简单设为固定值，例如 `5×5`，但更合理的启发式会考虑：

- 面积光源尺寸；
- receiver 到光源的距离；
- Shadow Map 的投影和分辨率。

直观上：

```text
光源越大 → 可能影响 receiver 的 blocker 范围越大
receiver 离光源越远 → 投影搜索范围通常也需要扩大
```

搜索区域太小可能漏掉 blocker；搜索区域太大可能混入不相关几何，增加开销并错误扩大半影。

### 6.5 PCSS 伪代码

```glsl
float pcss(vec2 shadowUV, float receiverDepth)
{
    BlockerResult blockers = searchBlockers(
        shadowUV,
        receiverDepth,
        blockerSearchRadius
    );

    if (blockers.count == 0)
        return 1.0;

    float penumbra =
        (receiverDepth - blockers.averageDepth)
        * lightSize
        / blockers.averageDepth;

    float filterRadius = projectToShadowTexels(penumbra);

    return percentageCloserFilter(
        shadowUV,
        receiverDepth,
        filterRadius
    );
}
```

这段伪代码表达算法关系，不代表某个 API 的直接实现；实际还需处理深度空间是否线性、透视投影、采样分布、bias 和纹理边界。

### 6.6 PCSS 的局限

- 每个 fragment 需要 blocker search 和 PCF 两组采样，成本高于普通 PCF；
- 平均 blocker 深度只是对复杂遮挡关系的近似；
- 搜索核、采样数量不足时可能闪烁、抖动或出现噪点；
- 仍依赖 Shadow Map，因此仍需处理 bias、分辨率和投影走样；
- 它近似面积光源软阴影，但不是完整的物理面积光积分。

---

## 7. 三种方法的完整对比

| 维度 | Shadow Mapping | PCF | PCSS |
|---|---|---|---|
| 每个 fragment 的核心操作 | 一次深度比较 | 多次深度比较并平均 | blocker search + 半影估计 + 可变 PCF |
| 可见性输出 | 通常为 0 或 1 | 0 到 1 | 0 到 1 |
| 主要目标 | 产生阴影 | 阴影边缘抗锯齿 | 模拟距离相关的软阴影 |
| 滤波核 | 无 | 通常固定 | 动态变化 |
| 是否表现接触硬化 | 否 | 通常否 | 是 |
| 是否仍需 bias | 是 | 是 | 是 |
| 成本 | 低 | 中 | 高 |

三者不是互斥关系，而是逐步扩展：

```text
PCF 建立在 Shadow Mapping 的深度比较上
PCSS 又使用 PCF 作为最后一步
```

---

## 8. 实现 Shadow Mapping 时的检查顺序

如果阴影结果异常，可以按以下顺序排查。

### 8.1 Shadow Map 生成是否正确

- 光源 View / Projection 是否正确；
- Shadow Map framebuffer 是否完整；
- 深度附件格式和分辨率是否正确；
- 是否真正绘制了所有投射阴影的物体；
- near/far 范围是否合理。

### 8.2 相机 Pass 的重投影是否正确

- 世界坐标是否正确变换到光源裁剪空间；
- 是否执行了透视除法；
- 是否正确从 NDC 映射到纹理坐标；
- `(u,v)` 是否落在 Shadow Map 范围内；
- `z_p` 与 Shadow Map 深度是否位于同一坐标约定和深度范围。

### 8.3 深度比较是否正确

- 比较方向是否写反；
- bias 符号是否正确；
- 是否出现 self-occlusion；
- 是否因 bias 太大产生 Peter Panning；
- 背面剔除和正面剔除是否影响 Shadow Map。

### 8.4 PCF / PCSS 是否正确

- 平均的是比较结果，而不是原始深度；
- texel size 是否使用 `1 / shadowMapResolution`；
- PCF 核半径是否处于合理范围；
- PCSS 是否正确区分 blocker 和 receiver；
- 没有 blocker 时是否直接返回完全可见；
- blocker 深度与 receiver 深度是否经过一致的线性化或投影处理。

---

## 9. 常见误区

### 误区一：Shadow Map 中存的是阴影颜色

不是。普通 Shadow Map 存的是从光源视角看到的最近深度。

### 误区二：把 Shadow Map 模糊就等于 PCF

不是。PCF 是先进行多次深度比较，再平均 0/1 比较结果。

### 误区三：PCF 就是物理软阴影

固定核 PCF 主要解决锯齿，只是在视觉上产生模糊边缘，不一定符合真实半影几何。

### 误区四：PCSS 不需要 PCF

PCSS 的第三步就是可变大小的 PCF。PCSS 负责决定滤波核应该多大。

### 误区五：bias 越大越安全

bias 过大会使真实接触阴影消失，产生 Peter Panning。

### 误区六：PCSS 能彻底解决 Shadow Mapping 的所有问题

不能。它仍然依赖有限分辨率的 Shadow Map，也仍有 bias、走样、采样成本和复杂遮挡近似问题。

---

## 10. 复习自测题

1. 为什么 Shadow Mapping 通常需要两个 pass？
2. Shadow Map 中保存的是什么数据？
3. 为什么相机可见点还要重新投影到光源空间？
4. `z_p > z_sm` 在几何上代表什么？
5. 为什么倾斜表面更容易出现 shadow acne？
6. bias 太小和太大分别会产生什么问题？
7. Second-depth Shadow Mapping 为什么要求模型 watertight？
8. 为什么“先平均深度再比较”不能替代 PCF？
9. PCF 得到 `0.667` 可见率代表什么？
10. 为什么固定大小 PCF 不属于严格意义上的软阴影算法？
11. PCSS 为什么需要先搜索 blocker？
12. blocker 与 receiver 越远，PCSS 的滤波核为什么越大？
13. PCSS 的三个步骤分别是什么？
14. 为什么 PCSS 仍然需要 bias？
15. 普通 Shadow Mapping、PCF、PCSS 分别解决什么问题？

---

## 11. 一页速记

```text
普通 Shadow Mapping：
Pass 1：从光源渲染最近深度 → Shadow Map
Pass 2：从相机渲染，把 fragment 投回光源空间
比较：z_receiver 与 z_shadowMap

典型问题：
shadow acne       ← 深度误差、自遮挡
Peter Panning     ← bias 过大、阴影脱离
aliasing          ← Shadow Map 分辨率和投影采样不足

数学解释：
带阴影光照 ≈ 平均可见率 × 未遮挡光照

PCF：
多个 Shadow Map 样本分别做深度比较
→ 平均 0/1 比较结果
→ 平滑阴影边缘

PCSS：
1. Blocker Search
2. Penumbra Estimation
3. Variable-size PCF

半影关系：
w_penumbra = (d_receiver - d_blocker) * w_light / d_blocker

最终区别：
Shadow Mapping：一次比较，硬阴影
PCF：固定核多次比较，边缘抗锯齿
PCSS：根据 blocker 距离改变核大小，接触处硬、远处软
```
