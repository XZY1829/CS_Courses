# 226. 翻转二叉树 - 题解

## 思路

递归交换每个节点的左右子树。前序、后序、层序都可以。

## 解法

```cpp
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (!root) return nullptr;
        swap(root->left, root->right);
        invertTree(root->left);
        invertTree(root->right);
        return root;
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(h)

## 关键点

1. 前序（先交换再递归）和后序（先递归再交换）都正确，但中序不行（会重复交换）
