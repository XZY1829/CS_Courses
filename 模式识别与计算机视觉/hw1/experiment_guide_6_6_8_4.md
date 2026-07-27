# 习题 6.6 与 8.4 实验操作指南

本文档用于从 0 到 1 完成《模式识别》课程作业中的两个实验题：

- 习题 6.6：PCA + FLD 人脸识别，比较 Eigenfaces 与 Fisherfaces。
- 习题 8.4：使用 Matlab `ksdensity` 对对数正态分布做 KDE。

建议先完成实验，再把实验截图、识别率、带宽数值和现象总结写入 `prcv_hw1_solution.tex`。

---

## 一、习题 6.6：PCA + FLD 人脸识别

### 1. 实验目标

本实验需要完成以下事情：

1. 下载 ORL/AT&T 人脸数据集。
2. 使用 OpenCV 读取人脸图像。
3. 分别训练 Eigenfaces 和 Fisherfaces 模型。
4. 比较两种方法的识别准确率。
5. 观察 Eigenfaces 的人脸重构效果。
6. 总结 PCA 与 FLD 在人脸识别中的差异。

---

### 2. 安装 Python 与依赖

建议使用 Python 3.10 或更高版本。

在 PowerShell 中进入作业目录：

```powershell
cd "f:\documents_personal\模式识别与计算机视觉\hw1"
```

创建虚拟环境：

```powershell
python -m venv .venv
```

激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 提示禁止执行脚本，先运行：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

然后重新激活虚拟环境。

安装依赖：

```powershell
pip install opencv-contrib-python numpy matplotlib scikit-learn
```

注意：必须安装 `opencv-contrib-python`，不能只安装 `opencv-python`，因为 EigenFaceRecognizer 和 FisherFaceRecognizer 在 contrib 模块里。

验证安装：

```powershell
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2.face, 'EigenFaceRecognizer_create')); print(hasattr(cv2.face, 'FisherFaceRecognizer_create'))"
```

如果最后两项输出都是 `True`，说明环境正确。

---

### 3. 下载 ORL/AT&T 数据集

ORL 数据集也叫 AT&T Database of Faces。常见下载地址：

```text
https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
```

下载 `att_faces.zip` 后解压到作业目录，例如：

```text
f:\documents_personal\模式识别与计算机视觉\hw1\att_faces
```

解压后的目录结构通常是：

```text
att_faces
├── s1
│   ├── 1.pgm
│   ├── 2.pgm
│   └── ...
├── s2
│   ├── 1.pgm
│   └── ...
...
└── s40
    ├── 1.pgm
    └── ...
```

其中：

- `s1` 到 `s40` 表示 40 个人。
- 每个人有 10 张图。
- 每张图大小通常是 `92 x 112`。
- 图像格式是 `.pgm`。

---

### 4. 新建实验脚本

在作业目录中新建文件：

```text
face_experiment.py
```

写入以下代码：

```python
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


DATA_DIR = Path("att_faces")
RANDOM_STATE = 0


def load_orl_faces(data_dir: Path):
    images = []
    labels = []

    subject_dirs = sorted(
        [p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("s")],
        key=lambda p: int(p.name[1:]),
    )

    for subject_dir in subject_dirs:
        label = int(subject_dir.name[1:]) - 1
        image_paths = sorted(subject_dir.glob("*.pgm"), key=lambda p: int(p.stem))

        for image_path in image_paths:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels, dtype=np.int32)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.train(list(X_train), y_train)

    correct = 0
    predictions = []
    confidences = []

    for img, label in zip(X_test, y_test):
        pred, confidence = model.predict(img)
        predictions.append(pred)
        confidences.append(confidence)
        if pred == label:
            correct += 1

    accuracy = correct / len(y_test)
    return accuracy, np.array(predictions), np.array(confidences)


def save_mean_face_and_eigenfaces(images):
    flat = images.reshape(images.shape[0], -1).astype(np.float64)
    mean = flat.mean(axis=0)
    centered = flat - mean

    # SVD gives principal directions. Rows of vh are eigenfaces.
    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    h, w = images.shape[1], images.shape[2]
    mean_face = mean.reshape(h, w)

    plt.figure(figsize=(3, 4))
    plt.imshow(mean_face, cmap="gray")
    plt.title("Mean Face")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("mean_face.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    for i in range(10):
        face = vh[i].reshape(h, w)
        plt.subplot(2, 5, i + 1)
        plt.imshow(face, cmap="gray")
        plt.title(f"Eigenface {i+1}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("top10_eigenfaces.png", dpi=150)
    plt.close()

    return mean, vh


def reconstruct_examples(images, mean, eigenfaces, indices=(0, 1, 2), components=(5, 10, 30, 80)):
    h, w = images.shape[1], images.shape[2]

    for idx in indices:
        x = images[idx].reshape(-1).astype(np.float64)
        centered = x - mean

        plt.figure(figsize=(2 * (len(components) + 1), 3))
        plt.subplot(1, len(components) + 1, 1)
        plt.imshow(images[idx], cmap="gray")
        plt.title("Original")
        plt.axis("off")

        for j, r in enumerate(components, start=2):
            basis = eigenfaces[:r]
            coeffs = basis @ centered
            recon = mean + coeffs @ basis
            recon = recon.reshape(h, w)

            plt.subplot(1, len(components) + 1, j)
            plt.imshow(recon, cmap="gray")
            plt.title(f"r={r}")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"reconstruction_{idx}.png", dpi=150)
        plt.close()


def main():
    images, labels = load_orl_faces(DATA_DIR)
    print(f"Loaded images: {images.shape}")
    print(f"Loaded labels: {labels.shape}")
    print(f"Number of classes: {len(np.unique(labels))}")

    X_train, X_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.4,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    eigen_model = cv2.face.EigenFaceRecognizer_create(num_components=80)
    eigen_acc, eigen_pred, eigen_conf = evaluate_model(
        eigen_model, X_train, y_train, X_test, y_test
    )

    # Fisherfaces has at most C - 1 effective components.
    fisher_model = cv2.face.FisherFaceRecognizer_create(num_components=39)
    fisher_acc, fisher_pred, fisher_conf = evaluate_model(
        fisher_model, X_train, y_train, X_test, y_test
    )

    print(f"Eigenfaces accuracy: {eigen_acc:.4f}")
    print(f"Fisherfaces accuracy: {fisher_acc:.4f}")

    mean, eigenfaces = save_mean_face_and_eigenfaces(images)
    reconstruct_examples(images, mean, eigenfaces)

    print("Saved figures:")
    print("  mean_face.png")
    print("  top10_eigenfaces.png")
    print("  reconstruction_0.png, reconstruction_1.png, reconstruction_2.png")


if __name__ == "__main__":
    main()
```

---

### 5. 运行实验

确保目录中有：

```text
att_faces
face_experiment.py
```

运行：

```powershell
python face_experiment.py
```

你会看到类似输出：

```text
Loaded images: (400, 112, 92)
Loaded labels: (400,)
Number of classes: 40
Train size: 240
Test size: 160
Eigenfaces accuracy: ...
Fisherfaces accuracy: ...
Saved figures:
  mean_face.png
  top10_eigenfaces.png
  reconstruction_0.png, reconstruction_1.png, reconstruction_2.png
```

把终端里的两个准确率记录下来：

```text
Eigenfaces accuracy = ______
Fisherfaces accuracy = ______
```

同时查看生成的图片：

```text
mean_face.png
top10_eigenfaces.png
reconstruction_0.png
reconstruction_1.png
reconstruction_2.png
```

---

### 6. 应该观察什么

#### 6.1 平均脸

`mean_face.png` 应该是一张模糊的人脸，保留了大致脸型、眼睛、鼻子和嘴的位置。

可写入报告的观察：

```text
平均脸呈现出 ORL 数据集中所有人脸的整体轮廓，但个体差异被平均掉，因此图像较为模糊。
```

#### 6.2 Eigenfaces

`top10_eigenfaces.png` 中每张图不是正常人脸，而是一些明暗变化模式。前几个 eigenfaces 往往反映光照、脸部轮廓、姿态等主要变化。

可写入报告的观察：

```text
前几个 eigenfaces 主要描述训练集中方差最大的变化方向，包括整体亮度、脸部轮廓和局部阴影变化。这说明 PCA 保留的是样本总体变化，而不是直接针对类别判别优化。
```

#### 6.3 重构图像

查看 `reconstruction_0.png` 等图片。一般会看到：

- `r=5`：只保留大致脸型，很模糊。
- `r=10`：五官位置更清楚。
- `r=30`：面部细节明显增加。
- `r=80`：更接近原图。

可写入报告的观察：

```text
随着使用的 eigenfaces 数量增加，重构图像逐渐接近原始图像。较少主成分只能保留低频结构和整体轮廓，较多主成分可以恢复更多五官细节和个体差异。
```

#### 6.4 Eigenfaces 与 Fisherfaces 准确率

一般来说，Fisherfaces 可能高于 Eigenfaces，但具体结果取决于训练测试划分、组件数量和 OpenCV 版本。

可写入报告的句式：

```text
本实验中 Eigenfaces 的识别准确率为 ______，Fisherfaces 的识别准确率为 ______。
从结果看，______ 的准确率更高。Eigenfaces 使用 PCA 最大化总体方差，并未直接利用类别标记；Fisherfaces 使用 FLD，使类间散度增大、类内散度减小，因此更适合分类任务。
```

如果你的结果 Fisherfaces 更高，就填：

```text
Fisherfaces
```

如果你的结果 Eigenfaces 更高，也不要慌，可以写：

```text
Eigenfaces
```

并解释：

```text
这可能与训练测试划分、样本数量较少以及参数设置有关。理论上 FLD 更关注类别可分性，但实际表现会受到数据划分和模型参数影响。
```

---

### 7. 6.6 报告模板

你可以把下面这段改成自己的准确率后放进 LaTeX：

```text
本实验使用 ORL 人脸数据集，共 40 个类别，每个类别 10 张图像。实验中将每张灰度图像展开为向量，并划分训练集和测试集。Eigenfaces 使用 PCA 提取主成分特征，Fisherfaces 先通过 PCA 降维，再使用 FLD 提取判别特征。

实验得到 Eigenfaces 的识别准确率为 ______，Fisherfaces 的识别准确率为 ______。从结果看，______ 的识别效果更好。原因是 Eigenfaces 主要保留样本总体方差较大的方向，这些方向可能反映光照、姿态等变化；Fisherfaces 使用类别标记，目标是增大类间散度并减小类内散度，因此通常更适合人脸身份识别。

在人脸重构实验中，使用较少 eigenfaces 时，重构图像只保留大致脸型和低频结构，五官细节较模糊；随着 eigenfaces 数量增加，重构图像逐渐接近原始图像。这说明 PCA 的主成分数量越多，保留的信息越多，重构误差越小。
```

---

## 二、习题 8.4：Matlab KDE 实验

### 1. 实验目标

本实验需要完成：

1. 从参数为 $\mu=2,\sigma=0.5$ 的对数正态分布生成样本。
2. 画出真实概率密度函数。
3. 使用 Matlab `ksdensity` 做核密度估计。
4. 比较不同带宽下 KDE 的差异。
5. 比较样本数为 `1000`、`10000`、`100000` 时自动带宽的变化。

---

### 2. 新建 Matlab 脚本

在作业目录中新建文件：

```text
kde_experiment.m
```

写入以下代码：

```matlab
clear; clc; close all;

rng(0);

mu = 2;
sigma = 0.5;

%% Part (a): generate 1000 samples and plot true pdf
n = 1000;
x = lognrnd(mu, sigma, n, 1);

xgrid = linspace(0.001, max(x) * 1.2, 1000);
true_pdf = lognpdf(xgrid, mu, sigma);

figure;
histogram(x, 40, 'Normalization', 'pdf');
hold on;
plot(xgrid, true_pdf, 'r-', 'LineWidth', 2);
title('Lognormal Samples and True PDF, n = 1000');
legend('Sample histogram', 'True PDF');
xlabel('x');
ylabel('Density');
grid on;
saveas(gcf, 'kde_part_a_true_pdf.png');

fprintf('Part (a)\n');
fprintf('Sample mean = %.6f\n', mean(x));
fprintf('Sample variance = %.6f\n', var(x));
fprintf('Theoretical mean = %.6f\n', exp(mu + sigma^2 / 2));
fprintf('Theoretical variance = %.6f\n\n', (exp(sigma^2) - 1) * exp(2 * mu + sigma^2));

%% Part (b): ksdensity with automatic bandwidth
[f_auto, xi_auto, bw_auto] = ksdensity(x);

figure;
plot(xgrid, true_pdf, 'k-', 'LineWidth', 2);
hold on;
plot(xi_auto, f_auto, 'r--', 'LineWidth', 2);
title('KDE with Automatic Bandwidth, n = 1000');
legend('True PDF', 'KDE');
xlabel('x');
ylabel('Density');
grid on;
saveas(gcf, 'kde_part_b_auto_bandwidth.png');

fprintf('Part (b)\n');
fprintf('Automatic bandwidth for n = 1000: %.6f\n\n', bw_auto);

%% Part (c): bandwidth 0.2 and 5
[f_h02, xi_h02] = ksdensity(x, 'Bandwidth', 0.2);
[f_h5, xi_h5] = ksdensity(x, 'Bandwidth', 5);

figure;
plot(xgrid, true_pdf, 'k-', 'LineWidth', 2);
hold on;
plot(xi_h02, f_h02, 'b--', 'LineWidth', 1.5);
plot(xi_h5, f_h5, 'r-.', 'LineWidth', 1.5);
title('KDE with Different Bandwidths');
legend('True PDF', 'Bandwidth = 0.2', 'Bandwidth = 5');
xlabel('x');
ylabel('Density');
grid on;
saveas(gcf, 'kde_part_c_bandwidth_compare.png');

fprintf('Part (c)\n');
fprintf('Bandwidth 0.2: curve is usually rough and sensitive to samples.\n');
fprintf('Bandwidth 5: curve is usually over-smoothed.\n\n');

%% Part (d): automatic bandwidth for larger sample sizes
sample_sizes = [1000, 10000, 100000];
bws = zeros(size(sample_sizes));

figure;
plot(xgrid, true_pdf, 'k-', 'LineWidth', 2);
hold on;

colors = {'r--', 'b-.', 'g:'};

for i = 1:length(sample_sizes)
    ni = sample_sizes(i);
    xi = lognrnd(mu, sigma, ni, 1);
    [fi, xii, bwi] = ksdensity(xi);
    bws(i) = bwi;
    plot(xii, fi, colors{i}, 'LineWidth', 1.5);
end

title('KDE with Automatic Bandwidth for Different Sample Sizes');
legend('True PDF', 'n=1000', 'n=10000', 'n=100000');
xlabel('x');
ylabel('Density');
grid on;
saveas(gcf, 'kde_part_d_sample_size_compare.png');

fprintf('Part (d)\n');
for i = 1:length(sample_sizes)
    fprintf('n = %d, automatic bandwidth = %.6f\n', sample_sizes(i), bws(i));
end
```

---

### 3. 运行 Matlab 实验

打开 Matlab，切换到作业目录：

```matlab
cd('f:\documents_personal\模式识别与计算机视觉\hw1')
```

运行：

```matlab
kde_experiment
```

运行后会生成四张图：

```text
kde_part_a_true_pdf.png
kde_part_b_auto_bandwidth.png
kde_part_c_bandwidth_compare.png
kde_part_d_sample_size_compare.png
```

命令行会输出：

```text
Sample mean = ...
Sample variance = ...
Theoretical mean = ...
Theoretical variance = ...
Automatic bandwidth for n = 1000: ...
n = 1000, automatic bandwidth = ...
n = 10000, automatic bandwidth = ...
n = 100000, automatic bandwidth = ...
```

把这些数值记录下来。

---

### 4. 每个小问应该怎么写

#### 4.1 Part (a)

你需要写：

- 样本来自对数正态分布。
- 样本全部为正。
- 直方图右偏。
- 理论密度是单峰长尾。
- 样本均值和方差接近理论均值和方差。

报告模板：

```text
生成 1000 个对数正态样本后，样本全部位于正半轴，直方图呈明显右偏分布。真实 p.d.f. 为单峰曲线，峰值位于正数区域，右侧有较长尾部。实验中样本均值为 ______，样本方差为 ______；理论均值为 8.3729，理论方差为 19.9117。样本统计量与理论值基本一致。
```

#### 4.2 Part (b)

你需要写：

- KDE 与真实密度整体接近。
- 主体区域拟合较好。
- 右尾区域波动较大。
- 自动带宽是多少。

报告模板：

```text
使用 ksdensity 的自动带宽进行 KDE，得到带宽为 ______。KDE 曲线整体上能够跟随真实 p.d.f. 的单峰和右偏形状，在样本密集的主体区域拟合较好；在右尾样本较稀疏的区域，KDE 与真实曲线存在一定偏差。
```

#### 4.3 Part (c)

你需要写：

- `Bandwidth = 0.2`：曲线更抖、局部峰多、方差大。
- `Bandwidth = 5`：曲线过度平滑、峰值被压低、偏差大。

报告模板：

```text
当带宽设为 0.2 时，KDE 曲线较为尖锐，局部波动明显，说明带宽过小会导致估计方差较大，容易受到样本随机波动影响。当带宽设为 5 时，KDE 曲线非常平滑，峰值明显降低，分布主体被拉宽，说明带宽过大会带来较大偏差。由此可见，带宽控制 KDE 的偏差和方差折中。
```

#### 4.4 Part (d)

你需要写：

- 记录三个样本数下的自动带宽。
- 说明样本数变大，自动带宽通常变小。
- KDE 曲线更平滑且更接近真实密度。

报告模板：

```text
当样本数分别为 1000、10000 和 100000 时，ksdensity 自动选择的带宽分别为 ______、______ 和 ______。可以观察到，随着样本数增加，自动带宽整体呈下降趋势。这是因为样本数越多，局部区域内可用于估计的样本越充分，因此可以使用更小的带宽来提高分辨率，同时仍保持较低方差。实验曲线也显示，样本数越大，KDE 越接近真实 p.d.f.。
```

---

## 三、最终提交前检查

### 1. 习题 6.6 需要保留的材料

建议保存：

```text
mean_face.png
top10_eigenfaces.png
reconstruction_0.png
Eigenfaces accuracy
Fisherfaces accuracy
```

### 2. 习题 8.4 需要保留的材料

建议保存：

```text
kde_part_a_true_pdf.png
kde_part_b_auto_bandwidth.png
kde_part_c_bandwidth_compare.png
kde_part_d_sample_size_compare.png
自动带宽数值
样本均值和方差
```

### 3. 写报告时不要只贴代码

每个实验至少写三类内容：

1. 实验设置：数据、方法、参数。
2. 实验结果：准确率、带宽、图像现象。
3. 结果解释：为什么会出现这种现象。

---

## 四、常见问题

### 1. OpenCV 报错：`module 'cv2' has no attribute 'face'`

原因：安装的是 `opencv-python`，不是 `opencv-contrib-python`。

解决：

```powershell
pip uninstall opencv-python opencv-contrib-python -y
pip install opencv-contrib-python
```

### 2. 找不到 ORL 数据集

确认目录结构是：

```text
hw1
├── face_experiment.py
└── att_faces
    ├── s1
    ├── s2
    └── ...
```

### 3. Fisherfaces 报错

Fisherfaces 要求训练集中每个类别至少有样本。代码使用了 `stratify=labels`，正常不会出现某类完全缺失。如果你改了划分方式，记得保证每个人都有训练样本。

### 4. Matlab 找不到 `lognrnd`

`lognrnd` 需要 Statistics and Machine Learning Toolbox。如果没有该工具箱，可以用：

```matlab
x = exp(mu + sigma * randn(n, 1));
```

这与从对数正态分布采样等价。

### 5. Matlab 找不到 `ksdensity`

`ksdensity` 也需要 Statistics and Machine Learning Toolbox。如果没有该工具箱，需要安装对应工具箱，或改用 Python 的 `scipy.stats.gaussian_kde` 完成类似实验。