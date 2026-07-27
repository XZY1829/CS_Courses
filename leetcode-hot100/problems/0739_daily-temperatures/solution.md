# 739. 每日温度 - 题解

## 思路

**单调栈**：维护一个递减栈（存下标）。新温度高于栈顶时，弹出并计算天数差。

## 解法

```cpp
vector<int> dailyTemperatures(vector<int>& t) {
    int n = t.size();
    vector<int> ans(n, 0);
    stack<int> stk;
    for (int i = 0; i < n; i++) {
        while (!stk.empty() && t[i] > t[stk.top()]) { int prev = stk.top(); stk.pop(); ans[prev] = i - prev; }
        stk.push(i);
    }
    return ans;
}
```

## 复杂度

- **时间**：O(n)　　**空间**：O(n)

## 关键点

1. 单调栈经典应用：找"下一个更大元素"
2. 栈中存下标而非值，方便计算距离
