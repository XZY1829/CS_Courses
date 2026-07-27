# 39. 组合总和 - 题解

## 思路

回溯：每个候选数可以无限次使用。排序后利用 `candidates[i] > target` 提前剪枝。

递归时传 `i`（不是 `i+1`）允许重复选同一个数。

## 解法

```cpp
class Solution {
    vector<vector<int>> result;
    vector<int> path;
    void backtrack(vector<int>& c, int target, int start) {
        if (target == 0) { result.push_back(path); return; }
        for (int i = start; i < c.size() && c[i] <= target; i++) {
            path.push_back(c[i]);
            backtrack(c, target - c[i], i); // 可重复选
            path.pop_back();
        }
    }
public:
    vector<vector<int>> combinationSum(vector<int>& c, int target) {
        sort(c.begin(), c.end()); result.clear(); path.clear();
        backtrack(c, target, 0); return result;
    }
};
```

## 复杂度

- **时间**：取决于解的数量和搜索树大小
- **空间**：O(target / min(candidates))

## 关键点

1. 排序 + `c[i] <= target` 剪枝大幅减少搜索量
2. `start` 参数避免重复组合（[2,3] 和 [3,2] 只出现一次）
