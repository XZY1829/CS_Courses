# 238. 除了自身以外数组的乘积 - 题解

## 思路

不能用除法，所以 `answer[i] = 左边所有元素的乘积 × 右边所有元素的乘积`。

**两次遍历**：
1. 从左到右扫一遍，`answer[i]` 存左边的前缀乘积
2. 从右到左扫一遍，`answer[i]` 乘上右边的后缀乘积

用 `answer` 数组本身存前缀乘积，再用一个变量 `right` 累积后缀乘积，实现 O(1) 额外空间。

## 解法

```cpp
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int n = nums.size();
        vector<int> answer(n, 1);
        int left = 1;
        for (int i = 0; i < n; i++) {
            answer[i] = left;
            left *= nums[i];
        }
        int right = 1;
        for (int i = n - 1; i >= 0; i--) {
            answer[i] *= right;
            right *= nums[i];
        }
        return answer;
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(1)（输出数组不计）

## 关键点

1. 不用除法——面试中这个约束很常见（处理含零的情况，除法会导致除零错误）
2. 两次遍历的思想：前缀乘积 × 后缀乘积
3. 含 0 的情况无需特殊处理，乘积自然会是 0
