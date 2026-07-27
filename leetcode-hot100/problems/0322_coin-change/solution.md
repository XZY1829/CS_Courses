# 322. 零钱兑换 - 题解
## 思路
完全背包：`dp[i] = min(dp[i - coin] + 1)`。初始化 `dp[0]=0`，其余为 `amount+1`（不可达标志）。
## 复杂度
- **时间**：O(amount × |coins|)　**空间**：O(amount)
