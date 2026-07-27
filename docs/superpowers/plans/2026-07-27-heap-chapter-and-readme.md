# 堆章节与仓库 README 完善 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将《堆》完善为兼顾课程考试与算法面试的中文笔记，并为 `CS_Courses` 独立仓库添加根目录 README 后同步 GitHub 与父仓库。

**Architecture:** 《堆》章节负责完整讲解概念、操作、实现和应用；根 README 只负责仓库导航、克隆同步流程及资料使用边界。验证分为 Markdown 结构检查、C++ 示例编译运行、相对链接检查和 Git 远端一致性检查。

**Tech Stack:** Markdown、Obsidian Wiki Link、C++17、Git、Git Submodule、PowerShell

## Global Constraints

- 默认文档使用中文。
- 保持 StudyVault 现有的 YAML 元数据、标签、概览表和 Obsidian 双向链接风格。
- 使用 0 基数组描述堆，不扩展到索引堆、二项堆和斐波那契堆。
- 不覆盖 `design_patterns/Cpp-Design-Patterns` 等嵌套仓库的本地改动。
- 不把 `.ovpn`、重复导出文本或其他已忽略文件加入版本控制。
- 根 README 不替第三方课程资料声明许可证，只明确提醒使用者核查原始来源和版权要求。

---

### Task 1: 完善《堆》章节

**Files:**
- Modify: `数据结构与算法/StudyVault/05-树与二叉树/堆.md`

**Interfaces:**
- Consumes: `数据结构与算法/StudyVault/03-栈和队列/队列与优先级队列.md` 与 `数据结构与算法/StudyVault/09-排序/排序算法综合比较.md` 的现有 Wiki Link 名称。
- Produces: 一篇包含最小堆完整 C++17 示例、复杂度证明、典型应用和自测题的独立笔记。

- [ ] **Step 1: 运行内容基线检查并确认旧章节不满足要求**

```powershell
$text = Get-Content -Raw '数据结构与算法/StudyVault/05-树与二叉树/堆.md'
$required = @('局部有序', '为什么 Floyd 建堆是 O(n)', 'std::priority_queue', 'Top K', '自测题')
$missing = @($required | Where-Object { -not $text.Contains($_) })
"MISSING=$($missing.Count)"
$missing
```

Expected: `MISSING` 大于 `0`，并列出缺失主题。

- [ ] **Step 2: 重写基础概念和统一示例**

补充以下内容：

- 完全二叉树是结构条件，父子满足堆序是顺序条件。
- 最小堆只保证祖先不大于后代，不保证同层或左右子树整体有序。
- 使用数组 `[9, 17, 23, 45, 78, 65, 87, 53]` 解释下标映射。
- 插入 `11` 时通过 `siftUp` 沿 `末尾 → 父结点 45 → 父结点 17` 上浮，最终停在下标 `1`；不得写成越过根结点 `9`。
- 删除堆顶时使用末尾元素补根，再通过 `siftDown` 持续与较小孩子交换。

- [ ] **Step 3: 增加操作、复杂度证明与完整 C++17 实现**

文档中的 `MinHeap` 必须提供以下公开接口：

```cpp
class MinHeap {
public:
    MinHeap() = default;
    explicit MinHeap(std::vector<int> values);
    bool empty() const noexcept;
    std::size_t size() const noexcept;
    int top() const;
    void push(int value);
    void pop();
};
```

实现使用 `std::vector<int>`，空堆 `top()` 和 `pop()` 抛出 `std::out_of_range`。构造函数从 `size / 2` 递减执行 `siftDown`，避免无符号下标下溢。正文用各高度结点数量的加权和说明 Floyd 建堆是 `O(n)`，并对比逐个插入的 `O(n log n)`。

- [ ] **Step 4: 增加标准库、应用、堆排序和易错点**

必须包含以下标准库写法：

```cpp
std::priority_queue<int> maxHeap;
std::priority_queue<int, std::vector<int>, std::greater<int>> minHeap;
```

应用覆盖：

- 求前 K 大元素：维护容量为 K 的最小堆，复杂度 `O(n log k)`；
- 求第 K 大元素：扫描结束后的堆顶即答案；
- 合并 K 个有序序列：堆中保存每个序列的当前候选；
- 任务调度：根据优先级或下一次执行时间取堆顶。

堆排序明确使用最大堆得到升序结果，时间 `O(n log n)`、额外空间 `O(1)`、不稳定。

- [ ] **Step 5: 提取并编译运行文档中的 C++ 示例**

将文档中标记为“完整实现”的 C++ 代码块通过标准输入交给编译器：

```powershell
$markdown = Get-Content -Raw '数据结构与算法/StudyVault/05-树与二叉树/堆.md'
$match = [regex]::Match($markdown, '(?s)<!-- heap-example:start -->\s*```cpp\s*(.*?)\s*```\s*<!-- heap-example:end -->')
if (-not $match.Success) { throw '未找到完整实现代码块' }
$match.Groups[1].Value | g++ -std=c++17 -Wall -Wextra -pedantic -x c++ - -o "$env:TEMP\heap_note_test.exe"
& "$env:TEMP\heap_note_test.exe"
```

Expected: 编译退出码为 `0`，程序输出 `heap example: PASS`。

- [ ] **Step 6: 提交《堆》章节**

```powershell
git add -- '数据结构与算法/StudyVault/05-树与二叉树/堆.md'
git commit -m 'docs: 完善堆章节'
```

Expected: 只提交目标章节。

---

### Task 2: 添加独立仓库根 README

**Files:**
- Create: `README.md`
- Modify: `docs/superpowers/plans/2026-07-27-heap-chapter-and-readme.md`

**Interfaces:**
- Consumes: 当前顶层课程目录和 `.gitmodules` 中的四个嵌套子模块。
- Produces: GitHub 仓库首页、递归克隆说明、日常同步命令和资料使用提醒。

- [ ] **Step 1: 确认根 README 尚不存在**

```powershell
Test-Path -LiteralPath 'README.md'
```

Expected: `False`。

- [ ] **Step 2: 创建中文 README**

README 包含：

- 仓库定位：计算机课程、算法练习与学习笔记集合；
- 顶层目录导航：学习计划、数据结构与算法、操作系统、网络、C++、设计模式、NLP、AI Agent、强化学习、视觉和图形学；
- 推荐入口：`两个月学习计划/README.md` 和数据结构 StudyVault；
- 递归克隆命令：

```bash
git clone --recurse-submodules git@github.com:XZY1829/CS_Courses.git
```

- 已有克隆初始化命令：

```bash
git submodule update --init --recursive
```

- 日常拉取和推送说明；
- 资料边界：个人学习整理，第三方内容遵循各自原始许可证和版权声明，根仓库暂无统一许可证。

- [ ] **Step 3: 检查 README 相对链接**

```powershell
$links = @(
  '两个月学习计划/README.md',
  '数据结构与算法/StudyVault/00-Dashboard/MOC - 数据结构与算法.md'
)
$missing = @($links | Where-Object { -not (Test-Path -LiteralPath $_) })
"BROKEN=$($missing.Count)"
$missing
```

Expected: `BROKEN=0`。如果实际总索引路径不同，应在写入 README 前使用现有路径替换，不创建虚假入口。

- [ ] **Step 4: 提交 README 和实施计划**

```powershell
git add -- README.md docs/superpowers/plans/2026-07-27-heap-chapter-and-readme.md
git commit -m 'docs: 添加 CS_Courses 仓库说明'
```

Expected: 本次提交只包含 README 和实施计划。

---

### Task 3: 验证并同步独立仓库与父仓库

**Files:**
- Verify: `数据结构与算法/StudyVault/05-树与二叉树/堆.md`
- Verify: `README.md`
- Modify in parent repository: `CS_Courses` gitlink

**Interfaces:**
- Consumes: Task 1 和 Task 2 的两个子仓库提交。
- Produces: GitHub `main` 最新提交，以及父仓库 Gitee `master` 对应的子模块指针。

- [ ] **Step 1: 运行最终文档检查**

```powershell
git diff --check HEAD~2..HEAD
rg -n '插入 11.*与 09 交换|TO[D]O|TB[D]' '数据结构与算法/StudyVault/05-树与二叉树/堆.md' README.md
git status --short
```

Expected: `git diff --check` 无输出；不存在错误示例或临时标记；除已知嵌套子模块本地改动外没有未提交文件。

- [ ] **Step 2: 推送独立仓库**

```powershell
git push origin main
$local = git rev-parse HEAD
$remote = (git ls-remote origin refs/heads/main).Split("`t")[0]
if ($local -ne $remote) { throw 'GitHub main 与本地 HEAD 不一致' }
```

Expected: GitHub `main` 与本地 `HEAD` 完全一致。

- [ ] **Step 3: 更新父仓库子模块指针**

```powershell
Set-Location ..
git add -- CS_Courses
git commit -m 'chore: 更新 CS_Courses 子模块'
git push origin master
```

Expected: 父仓库只记录 `CS_Courses` gitlink 的新提交，不包含 `.agents/skills/skill-c-cleaner/reports/`。

- [ ] **Step 4: 验证两个远端和本地保留项**

```powershell
$parentLocal = git rev-parse HEAD
$parentRemote = (git ls-remote origin refs/heads/master).Split("`t")[0]
$childLocal = git -C CS_Courses rev-parse HEAD
$childRemote = (git -C CS_Courses ls-remote origin refs/heads/main).Split("`t")[0]
"PARENT_MATCH=$($parentLocal -eq $parentRemote)"
"CHILD_MATCH=$($childLocal -eq $childRemote)"
git status --short
git -C CS_Courses status --short
```

Expected: `PARENT_MATCH=True`、`CHILD_MATCH=True`；原有嵌套仓库本地改动和父仓库未跟踪报告仍然存在且未进入提交。
