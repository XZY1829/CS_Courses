# 78. 子集 - 题解

## 思路

回溯：每个元素有"选"或"不选"两种选择。从 `start` 开始枚举，每到一个节点就记录当前路径。

## 解法

```cpp
class Solution {
    vector<vector<int>> result;
    vector<int> path;
    void backtrack(vector<int>& nums, int start) {
        result.push_back(path);
        for (int i = start; i < nums.size(); i++) {
            path.push_back(nums[i]);
            backtrack(nums, i + 1);
            path.pop_back();
        }
    }
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        result.clear(); path.clear(); backtrack(nums, 0); return result;
    }
};
```

## 复杂度

- **时间**：O(n × 2^n)
- **空间**：O(n)

## 关键点

1. 与排列的区别：子集用 `start` 防止重复选择；排列不限制
2. 总共 2^n 个子集，因为每个元素选或不选
