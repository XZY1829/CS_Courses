# 102. 二叉树的层序遍历 - 题解

## 思路

BFS + 按层分组：每次处理队列中的所有节点（即一层），然后再处理下一层。

关键是用 `size = q.size()` 记录当前层的节点数，内层 for 循环恰好处理一层。

## 解法

```cpp
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> result;
        if (!root) return result;
        queue<TreeNode*> q;
        q.push(root);
        while (!q.empty()) {
            int size = q.size();
            vector<int> level;
            for (int i = 0; i < size; i++) {
                TreeNode* node = q.front(); q.pop();
                level.push_back(node->val);
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            result.push_back(level);
        }
        return result;
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(n)（队列最多一层的节点数）

## 关键点

1. `size` 必须在内层循环开始前获取，因为循环中队列大小会变
2. BFS 层序遍历是树、图问题的基础模板
