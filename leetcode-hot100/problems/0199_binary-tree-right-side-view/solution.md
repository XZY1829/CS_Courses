# 199. 二叉树的右视图 - 题解

## 思路

层序遍历，每层取**最后一个**节点的值。

## 解法

```cpp
class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
        vector<int> result;
        if (!root) return result;
        queue<TreeNode*> q;
        q.push(root);
        while (!q.empty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode* node = q.front(); q.pop();
                if (i == size - 1) result.push_back(node->val);
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
        }
        return result;
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(n)

## 关键点

1. 就是层序遍历（102 题）的变体，只取每层最后一个
2. DFS 也可以：按「根→右→左」顺序遍历，每层第一个遇到的就是右视图节点
