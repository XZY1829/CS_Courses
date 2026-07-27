# 560. 和为 K 的子数组 - 题解

## 思路

**前缀和 + 哈希表**。

子数组 `[i+1, j]` 的和 = `prefix[j] - prefix[i]`。如果这个差等于 k，则 `prefix[i] = prefix[j] - k`。

因此问题转化为：对于每个 j，有多少个之前的前缀和等于 `prefix[j] - k`？用哈希表计数即可。

**注意**：因为数组含负数，不能用滑动窗口（窗口和不单调）。

## 解法

```cpp
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> prefixCount;
        prefixCount[0] = 1;  // 空前缀的前缀和为 0
        int sum = 0, count = 0;
        for (int n : nums) {
            sum += n;
            if (prefixCount.count(sum - k))
                count += prefixCount[sum - k];
            prefixCount[sum]++;
        }
        return count;
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(n)，哈希表存前缀和

## 关键点

1. **初始化 `prefixCount[0] = 1`** 是关键——表示空前缀，处理从下标 0 开始的子数组
2. 含负数，所以不能用滑动窗口，只能用前缀和
3. 这是"两数之和"的变体：从"两数差为 k"→"两前缀和差为 k"
