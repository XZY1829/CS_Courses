# 239. 滑动窗口最大值 - 题解

## 思路

暴力每个窗口取 max 是 O(nk)。用**单调递减双端队列（deque）**可以做到 O(n)。

**单调队列**维护窗口内元素的下标，保证队列中对应的值**从头到尾单调递减**：
1. **入队**：新元素入队前，弹出队尾所有 ≤ 新元素的值（它们不可能成为任何后续窗口的最大值）
2. **出队**：如果队头下标已不在窗口内（`front ≤ i - k`），弹出队头
3. **取值**：队头始终是当前窗口的最大值

## 解法

```cpp
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        deque<int> dq;  // 存下标，维护单调递减
        vector<int> result;
        for (int i = 0; i < (int)nums.size(); i++) {
            if (!dq.empty() && dq.front() <= i - k)
                dq.pop_front();
            while (!dq.empty() && nums[dq.back()] <= nums[i])
                dq.pop_back();
            dq.push_back(i);
            if (i >= k - 1)
                result.push_back(nums[dq.front()]);
        }
        return result;
    }
};
```

## 复杂度

- **时间**：O(n)，每个元素最多入队出队各一次
- **空间**：O(k)，deque 中最多 k 个元素

## 关键点

1. 队列中存**下标**而非值，方便判断是否超出窗口范围
2. 单调队列的"弹出不可能成为最大值的元素"是核心——使队头始终是窗口最大值
3. 单调队列是处理"滑动窗口极值"问题的通用工具
