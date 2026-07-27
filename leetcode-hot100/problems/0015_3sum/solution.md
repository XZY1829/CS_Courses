# 15. 三数之和 - 题解

## 思路

暴力 O(n³) 不可接受。核心思路：**排序 + 固定一个数 + 双指针找另外两个数**。

1. 先将数组排序
2. 枚举第一个数 `nums[i]`（`i` 从 0 到 n-3）
3. 对剩余部分 `[i+1, n-1]` 用左右双指针找两数之和为 `-nums[i]` 的对

**去重**是本题的难点：
- 第一层：如果 `nums[i] == nums[i-1]`，跳过（避免重复固定同一个值）
- 第二层：找到一组解后，左右指针分别跳过相同值

## 解法

```cpp
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> result;
        int n = nums.size();
        for (int i = 0; i < n - 2; i++) {
            if (nums[i] > 0) break;
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int left = i + 1, right = n - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (sum < 0) left++;
                else if (sum > 0) right--;
                else {
                    result.push_back({nums[i], nums[left], nums[right]});
                    while (left < right && nums[left] == nums[left + 1]) left++;
                    while (left < right && nums[right] == nums[right - 1]) right--;
                    left++;
                    right--;
                }
            }
        }
        return result;
    }
};
```

## 复杂度

- **时间**：O(n²)，外层 O(n) × 内层双指针 O(n)，排序 O(n log n)
- **空间**：O(log n)（排序栈空间，不计输出）

## 关键点

1. **排序是前提**：只有有序数组才能用双指针 + 跳重复
2. **剪枝**：`nums[i] > 0` 时直接 break（排序后后面都 > 0，不可能凑零）
3. **三层去重**：i 层跳重复 + 找到解后 left/right 各跳重复
4. 此模式可推广到 4Sum、kSum
