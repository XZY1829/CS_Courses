# 114. 二叉树展开为链表 - 题解

## 思路

将二叉树按前序遍历展开为链表（全部挂在 right 上，left 置 null）。

**O(1) 空间的迭代法**：对每个节点，如果有左子树：
1. 找左子树的最右节点（前序遍历中，它是左子树最后一个被访问的）
2. 把当前节点的右子树挂到这个最右节点的 right
3. 把左子树移到 right，left 置 null

## 解法

```cpp
class Solution {
public:
    void flatten(TreeNode* root) {
        TreeNode* cur = root;
        while (cur) {
            if (cur->left) {
                TreeNode* pre = cur->left;
                while (pre->right) pre = pre->right;
                pre->right = cur->right;
                cur->right = cur->left;
                cur->left = nullptr;
            }
            cur = cur->right;
        }
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(1)

## 关键点

1. 这实际上是 Morris 遍历的变体
2. 展开后的顺序是前序遍历
