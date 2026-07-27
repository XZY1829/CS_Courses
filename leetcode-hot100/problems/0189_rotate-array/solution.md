# 189. 轮转数组 - 题解

## 思路

**三次翻转法**（O(1) 空间）：

以 `[1,2,3,4,5,6,7], k=3` 为例：
1. 整体翻转：`[7,6,5,4,3,2,1]`
2. 翻转前 k 个：`[5,6,7,4,3,2,1]`
3. 翻转后 n-k 个：`[5,6,7,1,2,3,4]`

直觉：右移 k 位 = 把后 k 个元素搬到前面。三次翻转正好实现这个效果。

## 解法

```cpp
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        int n = nums.size();
        k %= n;
        reverse(nums.begin(), nums.end());
        reverse(nums.begin(), nums.begin() + k);
        reverse(nums.begin() + k, nums.end());
    }
};
```

## 复杂度

- **时间**：O(n)，三次翻转共访问 2n 次
- **空间**：O(1)

## 关键点

1. **`k %= n`** 是必须的——k 可能大于数组长度
2. 三次翻转法也用于字符串的循环移位问题
3. 其他方法：额外数组 O(n) 空间、环状替换 O(1) 空间但实现复杂
