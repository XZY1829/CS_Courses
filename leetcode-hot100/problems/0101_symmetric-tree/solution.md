# 101. 对称二叉树 - 题解

## 思路

对称 = 左子树和右子树互为镜像。递归检查：左的左 vs 右的右，左的右 vs 右的左。

## 解法

```cpp
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        return !root || check(root->left, root->right);
    }
private:
    bool check(TreeNode* l, TreeNode* r) {
        if (!l && !r) return true;
        if (!l || !r) return false;
        return l->val == r->val && check(l->left, r->right) && check(l->right, r->left);
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(h)

## 关键点

1. 镜像比较：`check(l->left, r->right)` 和 `check(l->right, r->left)`
2. 也可以用 BFS + 队列实现迭代版本
