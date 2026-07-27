# 131. 分割回文串 - 题解

## 思路

回溯：从位置 `start` 开始，尝试每个可能的分割点 `end`。如果 `s[start..end]` 是回文，将其加入路径并递归处理剩余部分。

## 解法

```cpp
class Solution {
    vector<vector<string>> result;
    vector<string> path;
    bool isPalin(const string& s, int l, int r) {
        while (l < r) if (s[l++] != s[r--]) return false;
        return true;
    }
    void backtrack(const string& s, int start) {
        if (start == s.size()) { result.push_back(path); return; }
        for (int end = start; end < s.size(); end++) {
            if (isPalin(s, start, end)) {
                path.push_back(s.substr(start, end-start+1));
                backtrack(s, end+1);
                path.pop_back();
            }
        }
    }
};
```

## 复杂度

- **时间**：O(n × 2^n)
- **空间**：O(n)

## 关键点

1. 可以预处理 DP 回文判断表优化 isPalin 到 O(1)
2. 只有当前段是回文时才继续递归，起到剪枝效果
