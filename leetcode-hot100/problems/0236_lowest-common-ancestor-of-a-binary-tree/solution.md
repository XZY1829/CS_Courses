# 236. 二叉树的最近公共祖先 - 题解

## 思路

后序遍历，递归返回"在这棵子树中是否找到了 p 或 q"。

- 如果当前节点是 p 或 q，直接返回
- 递归左右子树
- 如果左右都非空，说明 p 和 q 分别在两侧 → 当前节点就是 LCA
- 否则返回非空的那个

## 解法

```cpp
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root || root == p || root == q) return root;
        TreeNode* left = lowestCommonAncestor(root->left, p, q);
        TreeNode* right = lowestCommonAncestor(root->right, p, q);
        if (left && right) return root;
        return left ? left : right;
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(h)

## 关键点

1. 代码极其简洁但理解需要仔细：返回值的含义是"在子树中找到的 p/q 或它们的 LCA"
2. 保证 p 和 q 都存在于树中，否则需要额外验证
