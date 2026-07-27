# LLM 中的强化学习

LLM 可看作强化学习智能体：状态是 prompt 与已生成上下文，动作是下一个 token，奖励评价回答质量。目标仍是最大化累计奖励。

## RLHF

SFT 只模仿标准答案，缺少反馈。RLHF 把人类偏好转化为奖励模型，经典三阶段：SFT、Reward Model、PPO。

LLM-PPO 常维护 Actor、Reference Model、Reward Model、Critic。优势估计可写成：

$$
\hat A_t=(R_{score}-\beta KL(\pi,\pi_{ref})+\gamma V(s_{t+1}))-V(s_t)
$$

痛点包括显存占用高、训练成本高、工程复杂、稳定性要求高。

## RLAIF

RLAIF 用 AI 反馈替代或减少人工反馈。宪法 AI 给 Teacher 模型原则清单，让其按规则打分。优点是成本低、扩展快、容易注入安全目标；风险是 AI 偏见可能被放大。

## DPO

DPO 不显式训练 Reward Model 和 Critic，而是直接使用 chosen/rejected 偏好数据，让 chosen 概率上升、rejected 概率下降。它降低工程复杂度，但仍依赖高质量偏好数据。

## RLVR

RLVR 使用可验证奖励，奖励来自环境、规则或测试。适合代码沙箱、数学符号验证、格式与执行检查。DeepSeek-R1 类模型展示了 RLVR 对推理链条和自我纠错的强化作用。

## GRPO

GRPO 不依赖 Critic，也不一定需要成对偏好数据，而是对同一问题生成一组回答，在组内计算优势：

$$
A_i=\frac{r_i-\operatorname{mean}(r_{group})}{\operatorname{std}(r_{group})+\epsilon}
$$

它保留 clip 和 KL 约束，减少 Reward Model/Critic 开销；但需要对同一输入采样多个候选，采样成本较高，且依赖初始模型能产生有区分度的答案。

## 三类反馈比较

| 范式 | 反馈来源 | 代表技术 | 适用场景 |
|---|---|---|---|
| RLHF | 人类偏好 | InstructGPT/PPO | 创意写作、安全性、闲聊 |
| RLAIF | AI 打分器 | Constitutional AI/DPO | 大规模通用对齐 |
| RLVR | 规则/环境 | DeepSeek-R1/GRPO | 数学、编程、逻辑推理 |
