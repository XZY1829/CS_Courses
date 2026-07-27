# CS_Courses

面向计算机基础、算法训练与 AI 方向的个人学习仓库。内容包括课程资料、结构化笔记、练习记录、课程项目和部分外部学习项目。

这个仓库更适合作为学习索引与长期知识库使用，而不是一套从零开始、严格按单一教学大纲编排的教材。

## 推荐入口

- [两个月学习计划](两个月学习计划/README.md)：按阶段组织的数据结构、算法、C++、操作系统、计算机网络与设计模式学习路线。
- [数据结构与算法知识地图](数据结构与算法/StudyVault/00-Dashboard/MOC%20-%20数据结构与算法.md)：Obsidian StudyVault 的课程导航、重点笔记与配套练习。

## 仓库内容

| 目录 | 内容 |
|------|------|
| [`两个月学习计划/`](两个月学习计划/) | 分阶段学习安排、每日任务和验收清单 |
| [`数据结构与算法/`](数据结构与算法/) | 数据结构课程资料、StudyVault 笔记和章节练习 |
| [`leetcode-hot100/`](leetcode-hot100/) | LeetCode Hot 100 训练内容 |
| [`cpp/`](cpp/) | C++ 语言与工程实践 |
| [`design_patterns/`](design_patterns/) | 设计模式资料及外部示例仓库 |
| [`OS/`](OS/) | 操作系统课程资料 |
| [`中科大郑老师网络（2025）/`](中科大郑老师网络（2025）/) | 计算机网络课程资料 |
| [`NLP/`](NLP/) | 自然语言处理课程、实验和项目 |
| [`AIAgent/`](AIAgent/) | AI Agent 学习资料与实践项目 |
| [`多智能体系统与强化学习/`](多智能体系统与强化学习/) | 多智能体系统、强化学习笔记和作业 |
| [`模式识别与计算机视觉/`](模式识别与计算机视觉/) | 模式识别和计算机视觉课程内容 |
| [`计算机图形学/`](计算机图形学/) | 计算机图形学课程与作业 |
| [`程设/`](程设/) | 程序设计课程内容 |

## 克隆仓库

仓库包含嵌套 Git 子模块，首次克隆建议直接递归初始化：

```bash
git clone --recurse-submodules git@github.com:XZY1829/CS_Courses.git
cd CS_Courses
```

如果已经完成普通克隆，再执行：

```bash
git submodule update --init --recursive
```

查看所有子模块状态：

```bash
git submodule status --recursive
```

## 日常同步

### 拉取独立仓库记录的版本

```bash
git pull --ff-only origin main
git submodule update --init --recursive
```

第一条命令更新 `CS_Courses` 本身，第二条命令让嵌套子模块回到当前提交记录的版本。嵌套子模块存在未提交修改时，应先提交或暂存自己的改动，再更新对应子模块。

### 提交 CS_Courses 内容

```bash
git add <修改的文件>
git commit -m "docs: 更新学习笔记"
git push origin main
```

提交前建议使用 `git status --short` 确认范围，避免把本地配置、生成文件或嵌套仓库中的实验改动意外加入提交。

### 作为 Study-2026 子模块使用

如果本仓库位于父仓库的 `CS_Courses` 目录中，需要先提交并推送本仓库，再让父仓库记录新的子模块提交：

```bash
# 在 CS_Courses 中
git add <修改的文件>
git commit -m "docs: 更新课程内容"
git push origin main

# 回到 Study-2026
cd ..
git add CS_Courses
git commit -m "chore: 更新 CS_Courses 子模块"
git push origin master
```

父仓库只保存一个子模块提交指针，不会重复保存 `CS_Courses` 内的全部文件。

## 关于 StudyVault

`数据结构与算法/StudyVault` 使用 Obsidian Wiki Link 组织知识关系。直接在 GitHub 阅读 Markdown 不影响主要正文，但 `[[笔记名称]]` 形式的双向链接在 Obsidian 中体验更完整。

## 资料与版权说明

本仓库用于个人学习、复习与课程实践，其中部分目录包含第三方课程资料、示例项目或 Git 子模块：

- 原创笔记与代码之外的内容，版权归原作者或原课程提供方所有；
- 外部项目应遵循其各自仓库中的许可证和使用条款；
- 课程讲义、作业要求和数据集不因被收录在本仓库中而改变原有版权状态；
- 本仓库根目录目前没有统一的 `LICENSE`，不要假定所有内容都采用同一种开源许可证。

如计划引用、修改或再分发某项内容，请先检查对应文件、子目录或上游项目的来源与授权说明。
