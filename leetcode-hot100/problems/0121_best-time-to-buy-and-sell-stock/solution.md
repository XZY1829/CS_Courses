# 121. 买卖股票的最佳时机 - 题解

## 思路
维护历史最低价，对每个价格计算利润 = 当前价 - 历史最低价。

## 解法
```cpp
int maxProfit(vector<int>& prices) {
    int minPrice = INT_MAX, maxProfit = 0;
    for (int p : prices) { minPrice = min(minPrice, p); maxProfit = max(maxProfit, p - minPrice); }
    return maxProfit;
}
```
## 复杂度
- **时间**：O(n)　**空间**：O(1)
