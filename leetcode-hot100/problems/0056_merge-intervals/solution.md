# 56. 合并区间 - 题解

## 思路

1. 按区间**起点排序**
2. 遍历区间，与已合并的最后一个区间比较：
   - 有重叠（当前起点 ≤ 上一个终点）：合并，取两个终点的较大值
   - 无重叠：直接加入结果

## 解法

```cpp
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end());
        vector<vector<int>> merged;
        for (auto& intv : intervals) {
            if (merged.empty() || merged.back()[1] < intv[0])
                merged.push_back(intv);
            else
                merged.back()[1] = max(merged.back()[1], intv[1]);
        }
        return merged;
    }
};
```

## 复杂度

- **时间**：O(n log n)，排序主导
- **空间**：O(log n)（排序栈空间，不计输出）

## 关键点

1. 排序后只需一次遍历，因为排序保证了"如果两个区间不重叠，后面的也不可能与前面的重叠"
2. 合并时取 `max(终点)`，不能直接用后一个的终点（可能被前一个完全包含）
3. 边界相等（`[1,4]` 和 `[4,5]`）算重叠
