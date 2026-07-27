# 22. 括号生成 - 题解

## 思路

回溯：维护已放置的左括号数 `open` 和右括号数 `close`：
- `open < n`：可以放左括号
- `close < open`：可以放右括号（保证合法性）

## 解法

```cpp
class Solution {
    vector<string> result;
    void backtrack(string& s, int open, int close, int n) {
        if (s.size() == 2 * n) { result.push_back(s); return; }
        if (open < n) { s += '('; backtrack(s, open+1, close, n); s.pop_back(); }
        if (close < open) { s += ')'; backtrack(s, open, close+1, n); s.pop_back(); }
    }
public:
    vector<string> generateParenthesis(int n) {
        result.clear(); string s; backtrack(s, 0, 0, n); return result;
    }
};
```

## 复杂度

- **时间**：O(4^n / √n)（第 n 个卡特兰数）
- **空间**：O(n)

## 关键点

1. `close < open` 是保证合法性的关键约束——任何前缀中右括号数不超过左括号数
2. n=3 时有 5 种：`((())), (()()),  (())(), ()(()),  ()()()`
