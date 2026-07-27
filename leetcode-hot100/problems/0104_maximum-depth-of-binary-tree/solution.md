# 104. 二叉树的最大深度 - 题解

## 思路

递归一行解决：树的最大深度 = `1 + max(左子树深度, 右子树深度)`。

## 解法

```cpp
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (!root) return 0;
        return 1 + max(maxDepth(root->left), maxDepth(root->right));
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(h)，递归栈

## 关键点

1. 后序遍历的思路：先算左右子树深度，再汇总
2. 也可以用 BFS 层序遍历，数层数
