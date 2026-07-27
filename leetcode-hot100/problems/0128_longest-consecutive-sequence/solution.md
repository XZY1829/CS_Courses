# 128. 最长连续序列 - 题解

## 思路

题目要求 O(n) 时间，排序是 O(n log n) 不满足。

**核心观察**：对于一个连续序列 `[1,2,3,4]`，只需从**序列起点**（即 `num-1` 不在集合中的数）开始向后计数。

**算法**：
1. 将所有数放入 HashSet（去重 + O(1) 查找）
2. 遍历 set 中每个数 n：
   - 如果 `n-1` 在 set 中，说明 n 不是起点，**跳过**
   - 如果 `n-1` 不在 set 中，n 是起点，从 n 开始向后数连续的数
3. 维护全局最长长度

跳过非起点是保证 O(n) 的关键——每个元素最多被作为起点访问一次、被后续计数访问一次。

## 解法

```cpp
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> s(nums.begin(), nums.end());
        int best = 0;
        for (int n : s) {
            if (!s.count(n - 1)) {  // n 是序列起点
                int cur = n, len = 1;
                while (s.count(cur + 1)) {
                    cur++;
                    len++;
                }
                best = max(best, len);
            }
        }
        return best;
    }
};
```

## 复杂度

- **时间**：O(n)，虽然有嵌套循环，但 while 循环总共最多执行 n 次（每个元素最多被 while 访问一次）
- **空间**：O(n)，HashSet 存储所有元素

## 关键点

1. **只从起点开始计数**（`n-1` 不在集合中），这是 O(n) 的保证，否则退化为 O(n²)
2. HashSet 自动处理重复元素，如 `[1,0,1,2]` 中重复的 1
3. 空数组返回 0，单个元素返回 1
