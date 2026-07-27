# 155. 最小栈 - 题解

## 思路

辅助栈 `minStk` 同步记录每个状态的最小值。push 时入两个栈，pop 时出两个栈。

## 解法

```cpp
class MinStack {
    stack<int> stk, minStk;
public:
    void push(int val) { stk.push(val); minStk.push(minStk.empty() ? val : min(val, minStk.top())); }
    void pop() { stk.pop(); minStk.pop(); }
    int top() { return stk.top(); }
    int getMin() { return minStk.top(); }
};
```

## 复杂度

- 所有操作 O(1)
- **空间**：O(n)
