# 94. 二叉树的中序遍历 - 题解

## 思路

中序遍历：左 → 根 → 右。迭代法用栈模拟递归：

1. 一路向左压栈
2. 弹出栈顶，记录值
3. 转向右子树

## 解法

```cpp
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> result;
        stack<TreeNode*> stk;
        TreeNode* cur = root;
        while (cur || !stk.empty()) {
            while (cur) { stk.push(cur); cur = cur->left; }
            cur = stk.top(); stk.pop();
            result.push_back(cur->val);
            cur = cur->right;
        }
        return result;
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(h)，h 为树高

## 关键点

1. 迭代法的核心是"一路向左入栈"，模拟递归调用栈
2. 前序遍历只需调整 `push_back` 的位置；后序遍历用"根右左"再反转
3. Morris 遍历可以做到 O(1) 空间
