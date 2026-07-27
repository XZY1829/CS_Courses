# 543. 二叉树的直径 - 题解

## 思路

直径 = 任意两节点之间最长路径的**边数**。最长路径一定经过某个节点，且 = 该节点左子树深度 + 右子树深度。

在求深度的过程中，顺便更新全局最大直径。

## 解法

```cpp
class Solution {
    int ans = 0;
    int depth(TreeNode* node) {
        if (!node) return 0;
        int l = depth(node->left);
        int r = depth(node->right);
        ans = max(ans, l + r);
        return 1 + max(l, r);
    }
public:
    int diameterOfBinaryTree(TreeNode* root) {
        depth(root);
        return ans;
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(h)

## 关键点

1. 直径是**边数**不是节点数，所以 `l + r` 而不是 `l + r + 1`
2. 最长路径不一定经过根节点，所以需要在每个节点都检查
3. 与第 124 题（最大路径和）思路完全一样，只是一个统计边数、一个统计值的和
