# UE 开发知识库

这个目录用于系统整理 Unreal Engine 与 D5 Render 开发知识。目标不是收藏零散链接，而是形成一套能够学习、复习、查源码和验证掌握程度的长期知识库。

## 推荐入口

- [从零掌握 UE 与 D5 开发](00-学习路线/从零掌握UE与D5开发.md)：完整学习顺序、阶段目标和验收方式。
- [UE 对象模型](01-UE-C++基础/01-UE对象模型.md)：从 `UObject`、`AActor` 和 `Component` 建立第一套 UE 心智模型。

## 知识结构

| 模块 | 主要内容 | 当前状态 |
|---|---|---|
| `00-学习路线` | 学习地图、阶段计划、掌握标准 | 已建立 |
| `01-UE-C++基础` | UE 类型、容器、委托、反射、UObject、CDO、GC | 编写中 |
| `02-Actor与场景` | World、Level、Actor、Component、生命周期、Transform | 待补充 |
| `03-三维与渲染` | 坐标、Camera、材质、模型、Spline、图形学基础 | 待补充 |
| `04-D5业务专题` | 素材库、保存加载、资源列表、模型、植被、相机、视频 | 待补充 |
| `05-工程与底层` | Qt+UE、Windows 桌面端、模块通信、线程、构建和 UE 底层 | 待补充 |
| `06-源码阅读与实战` | D5 调用链、真实需求、故障定位、验证记录 | 待补充 |

目录会随着笔记实际产生逐步创建，避免保留没有内容的空目录。

## 每篇笔记的统一结构

后续笔记尽量按以下结构编写：

1. **一句话结论**：这个概念解决什么问题。
2. **核心心智模型**：它与其他 UE 类型、系统的关系。
3. **最小示例**：只保留理解概念所需的代码。
4. **D5 对应位置**：它在真实业务中的使用方式。
5. **常见错误**：生命周期、线程、引用、时序和性能风险。
6. **掌握检查**：能否解释、实现、定位和验证。

## 掌握标准

一个主题只有同时通过下面五层，才标记为“已掌握”：

- 能用自己的话解释；
- 能读懂最小示例；
- 能独立写出基本实现；
- 能定位至少一种典型错误；
- 能在 D5 源码或真实功能中找到对应链路。

## 参考入口

- [Unreal Engine 官方文档](https://dev.epicgames.com/documentation/zh-cn/unreal-engine/)
- [Programming with C++](https://dev.epicgames.com/documentation/en-us/unreal-engine/programming-with-cplusplus-in-unreal-engine)
- [Unreal Engine C++ API Reference](https://dev.epicgames.com/documentation/en-us/unreal-engine/API)

外部资料只负责提供线索。涉及 D5 项目事实时，应以当前源码、构建结果和运行验证为准。
