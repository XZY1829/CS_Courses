# 108. 将有序数组转换为二叉搜索树 - 题解

## 思路

有序数组的中间元素作为根，可以保证左右子树高度差 ≤ 1（高度平衡）。递归对左半和右半构建子树。

## 解法

```cpp
class Solution {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return build(nums, 0, (int)nums.size() - 1);
    }
private:
    TreeNode* build(vector<int>& nums, int lo, int hi) {
        if (lo > hi) return nullptr;
        int mid = lo + (hi - lo) / 2;
        TreeNode* node = new TreeNode(nums[mid]);
        node->left = build(nums, lo, mid - 1);
        node->right = build(nums, mid + 1, hi);
        return node;
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(log n)，递归栈

## 关键点

1. 选中间元素保证平衡；选偏左或偏右的中间元素都可以（答案不唯一）
2. 这本质是二分思想的应用
