# 46. 全排列 - 题解

## 思路

回溯：在每个位置 `start`，尝试将后面的每个元素交换到当前位置，然后递归处理 `start+1`。

## 解法

```cpp
class Solution {
    vector<vector<int>> result;
    void backtrack(vector<int>& nums, int start) {
        if (start == nums.size()) { result.push_back(nums); return; }
        for (int i = start; i < nums.size(); i++) {
            swap(nums[start], nums[i]);
            backtrack(nums, start + 1);
            swap(nums[start], nums[i]); // 回溯
        }
    }
public:
    vector<vector<int>> permute(vector<int>& nums) {
        result.clear(); backtrack(nums, 0); return result;
    }
};
```

## 复杂度

- **时间**：O(n × n!)
- **空间**：O(n)（递归栈）

## 关键点

1. swap 交换法不需要 visited 数组，比"选择-撤销"写法更简洁
2. 有重复元素的排列（47 题）需要额外排序 + 去重
