# 437. 路径总和 III - 题解

## 思路

**前缀和 + DFS**——第 560 题（和为 K 的子数组）的树上版本。

从根到当前节点的路径和 `curSum`，如果存在某个祖先节点的前缀和为 `curSum - target`，那么从那个祖先到当前节点的路径和恰好等于 target。

DFS 时维护前缀和的出现次数哈希表，离开节点时**回溯**（减掉计数）。

## 解法

```cpp
class Solution {
    int count = 0;
    unordered_map<long long, int> prefixCount;
    void dfs(TreeNode* node, long long curSum, int targetSum) {
        if (!node) return;
        curSum += node->val;
        if (prefixCount.count(curSum - targetSum))
            count += prefixCount[curSum - targetSum];
        prefixCount[curSum]++;
        dfs(node->left, curSum, targetSum);
        dfs(node->right, curSum, targetSum);
        prefixCount[curSum]--;
    }
public:
    int pathSum(TreeNode* root, int targetSum) {
        count = 0; prefixCount.clear();
        prefixCount[0] = 1;
        dfs(root, 0, targetSum);
        return count;
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(h)

## 关键点

1. 用 `long long` 避免前缀和溢出
2. **回溯**是必须的——离开当前路径后，前缀和不应影响其他分支
3. `prefixCount[0] = 1` 处理从根到当前节点恰好和为 target 的情况
