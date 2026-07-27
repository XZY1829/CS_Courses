# AI Agent 主线 + UE 掌握路线（2-3个月，可落地）

> 目标定位：**侧重 AI Agent 应用开发**，UE 达到可独立开发/调试/讲解架构的掌握水平  
> 你的优势：已有客户端功能开发和 MCP Server 暴露经验，适合做“AI 调度 + UE 执行”方向

---

## 1. 你这轮的核心目标

### 主目标（AI）
- 建立完整 Agent 闭环：`用户指令 -> 任务规划 -> 工具调用 -> UE执行 -> 结果验证 -> 回复`
- 能设计可控工具系统：参数校验、白名单、超时取消、失败重试、回滚
- 能拿指标说话：任务成功率、平均执行时延、异常恢复率

### 次目标（UE 掌握）
- 能独立完成 UE 玩法原型和基础 AI（不是只会拼蓝图）
- 能把关键逻辑从蓝图迁移到 C++ 并解释设计原因
- 能对行为树、EQS、输入系统、状态同步做调试

---

## 2. 技术栈优先级（AI 优先）

## P0（前 2-3 周，必须拿下）
- Agent 基础：Function Calling / Tool Calling / JSON Schema
- 工具层：MCP Server、工具描述、输入输出协议、错误码设计
- UE 执行层：`AIController` + `Behavior Tree` + `Blackboard` + `NavMesh`
- UE 基础：`GameMode` / `PlayerController` / `Character` / `Enhanced Input`

## P1（第 4-7 周，形成工程能力）
- Agent 编排：多步任务规划、状态机、人工确认节点（HITL）
- 可观测性：日志链路、调用追踪、指标上报
- UE 进阶：C++ 组件化、`AIPerception`、`EQS`、DataAsset
- 稳定性：重试策略、幂等设计、任务中断与恢复

## P2（第 8-10 周，冲简历亮点）
- Lyra 架构理解（模块化和标签驱动）
- UE 与 Agent 深度集成（编辑器工具化、批量自动化、场景操作）
- Demo 包装：演示视频 + 技术文档 + 指标报告

---

## 3. 10 周执行计划（AI 主线）

## 第 1-2 周：打通最小闭环
- 任务：做出 `NL -> Tool -> UE Action -> 结果返回` 的最小链路
- UE 最小动作：生成 Actor、移动对象、切换天气、触发渲染
- 交付：`MVP v0.1`（可录屏）

## 第 3-4 周：工具系统可控化
- 任务：加参数校验、白名单、权限分层、失败处理
- UE 任务扩展：敌人巡逻/追击 + 场景对象批量放置
- 交付：`Tool Runtime v0.2`（有错误注入测试）

## 第 5-6 周：Agent 编排升级
- 任务：支持多步骤指令（如“放置掩体 -> 生成敌人 -> 渲染截图”）
- 增加：Planner/Executor/Verifier 三阶段执行
- 交付：`Agent Workflow v0.3`

## 第 7-8 周：UE 掌握强化
- 任务：把 2-3 个关键模块迁到 C++，保留蓝图做高层配置
- 增加：`AIPerception` 或 `EQS`，提升 AI 决策质量
- 交付：`Hybrid (C++ + BP) v0.4`

## 第 9-10 周：面试化与简历化
- 任务：指标统计、故障复盘、Demo 视频和文档整理
- 交付：可投递项目包（仓库 + README + 架构图 + 2个演示视频）

---

## 4. 核心 Demo（必做，直接写简历）

## Demo 名称
**UE Agent Director：自然语言驱动关卡搭建与渲染助手**

## 一句话描述
用户输入自然语言，Agent 自动拆解任务并调用工具，在 UE 中完成对象放置、敌人生成、参数调整和渲染出图。

## 最小功能（MVP）
- 输入指令：`在前方生成3个掩体，右侧生成2个巡逻敌人，并渲染一张4K预览图`
- Agent 能力：解析意图 -> 规划步骤 -> 调用多个工具 -> 汇总结果
- UE 执行：放置 Actor、设置 AI 巡逻点、触发渲染、返回结果路径

## 工程架构（建议）
- `agent-core`：对话管理、任务规划、工具调度
- `tool-gateway`：MCP/Function Calling 适配层、参数校验、权限控制
- `ue-runtime`：UE 执行模块（蓝图 + C++）
- `metrics`：成功率、时延、失败分类和日志

## Demo 验收标准
- 指令成功执行率 >= 80%
- 工具调用平均时延 <= 2s（不含长时间渲染）
- 出错时能给出明确原因并支持重试
- 至少 2 段演示视频：
  - 正常流程视频（1-2 分钟）
  - 异常恢复视频（30-60 秒）

## 简历写法模板（可直接改）
- 设计并实现 UE Agent Director，构建“自然语言 -> Agent 规划 -> 工具调用 -> UE 执行”的闭环系统，支持对象放置、敌人生成与渲染出图自动化。  
- 搭建 MCP/Function Calling 工具网关，加入参数校验、权限白名单和失败回滚机制，将复杂场景指令一次完成率提升至 `XX%`。  
- 建立调用链路监控与指标体系（成功率/时延/失败分类），将故障定位时间缩短 `XX%`。

---

## 5. 学习资源（可直接点开）

## AI Agent / MCP（主线优先）
- [2025 Agent + MCP 实战教程](https://www.bilibili.com/video/BV1wte4zUENd/)
- [2025版 MCP+Agent 从入门到实战](https://www.bilibili.com/video/BV1UAeRzBEqi/)
- [MCP+Agent 实战讲解](https://www.bilibili.com/video/BV1gvsmzwE9w/)
- [OpenAI Function Calling 官方文档](https://developers.openai.com/api/docs/guides/function-calling)

## UE 基础与掌握
- [UE5零基础入门（5.7）](https://www.bilibili.com/video/BV1qYSvBHELW/)
- [UE5蓝图视觉脚本入门（2025）](https://www.bilibili.com/video/BV189YnzkEy3/)
- [UE5 C++ 和蓝图入门](https://www.bilibili.com/video/BV1LzDHYWEJj/)
- [Tom Looman UE C++ 专业课程（AI中字）](https://www.bilibili.com/video/BV1Rt421V7r2/)

## UE AI / 架构进阶
- [UE5 AI基础：行为树起步](https://www.bilibili.com/video/BV18g411C7sc/)
- [Lyra流程讲解 + GAS 核心解读](https://www.bilibili.com/video/BV1dJ4m1w7pX/)
- [UE 行为树官方文档（中文）](https://dev.epicgames.com/documentation/zh-cn/unreal-engine/behavior-tree-in-unreal-engine---user-guide)
- [Editor Utility Blueprints 官方教程](https://dev.epicgames.com/community/learning/tutorials/owYv/unreal-engine-getting-started-with-editor-utility-blueprints)
- [GASDocumentation](https://github.com/tranek/GASDocumentation)

---

## 6. 明天上班执行清单（AI 版）

- [ ] 定义 5 个可调用工具（如 `spawn_actor`、`set_patrol`、`set_weather`、`render_shot`、`query_scene`）
- [ ] 写清每个工具的输入 Schema 和错误码
- [ ] 打通第一条自然语言指令到 UE 执行闭环
- [ ] 录制 1 分钟 MVP 演示
- [ ] 在 `README` 记录首批指标：成功率、时延、失败原因 Top3

---

## 7. 学习策略（避免跑偏）

- AI 主线不等于只调模型，核心是**系统工程能力**（工具、安全、可观测、可恢复）
- UE 掌握不等于全都 C++ 重写，正确做法是 **C++ 做核心，蓝图做配置和迭代**
- 每周必须有“可演示产物”，否则学习节奏会虚高
- 面试时优先讲“你怎么保证系统可靠”，而不只是“你接了哪些模型”
