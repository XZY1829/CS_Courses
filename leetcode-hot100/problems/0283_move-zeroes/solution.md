# 283. 移动零 - 题解

## 思路

经典的**快慢双指针**原地分区问题。

- `slow` 指向下一个要放非零元素的位置
- `fast` 遍历整个数组，遇到非零元素就和 `slow` 位置交换，然后 `slow++`

这样 `[0, slow)` 区间内全是非零元素且保持原始相对顺序，`[slow, n)` 区间全是零。

## 解法

```cpp
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int slow = 0;
        for (int fast = 0; fast < (int)nums.size(); fast++) {
            if (nums[fast] != 0) {
                swap(nums[slow], nums[fast]);
                slow++;
            }
        }
    }
};
```

## 复杂度

- **时间**：O(n)，一次遍历
- **空间**：O(1)，原地操作

## 关键点

1. 用 `swap` 而非赋值，可以同时完成"非零前移"和"零后移"，无需第二轮填零
2. 当 `slow == fast` 时 swap 是自交换，不影响正确性但可加 `if (slow != fast)` 优化
3. 此模式可推广到"将满足某条件的元素移到前面"等一系列分区问题
