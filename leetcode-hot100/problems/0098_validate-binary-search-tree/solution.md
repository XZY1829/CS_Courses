# 98. 验证二叉搜索树 - 题解

## 思路

BST 要求每个节点的值**严格在一个范围内**：左子树所有值 < 当前值 < 右子树所有值。

递归传递上下界：`validate(node, lower_bound, upper_bound)`。

## 解法

```cpp
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        return validate(root, LONG_MIN, LONG_MAX);
    }
private:
    bool validate(TreeNode* node, long lo, long hi) {
        if (!node) return true;
        if (node->val <= lo || node->val >= hi) return false;
        return validate(node->left, lo, node->val)
            && validate(node->right, node->val, hi);
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(h)

## 关键点

1. 用 `long` 避免 INT_MIN/INT_MAX 边界问题
2. 不能只检查"左孩子 < 根 < 右孩子"——需要检查**整个子树**都满足范围
3. 也可以用中序遍历检查是否严格递增
