# 230. 二叉搜索树中第 K 小的元素 - 题解

## 思路

BST 中序遍历是有序的。中序遍历到第 k 个元素即可提前返回。

## 解法

```cpp
class Solution {
public:
    int kthSmallest(TreeNode* root, int k) {
        stack<TreeNode*> stk;
        TreeNode* cur = root;
        while (cur || !stk.empty()) {
            while (cur) { stk.push(cur); cur = cur->left; }
            cur = stk.top(); stk.pop();
            if (--k == 0) return cur->val;
            cur = cur->right;
        }
        return -1;
    }
};
```

## 复杂度

- **时间**：O(h + k)，最坏 O(n)
- **空间**：O(h)

## 关键点

1. 利用 BST 的中序有序性
2. 不需要遍历完整棵树，找到第 k 个就返回
