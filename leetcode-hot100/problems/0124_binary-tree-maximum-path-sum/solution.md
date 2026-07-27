# 124. 二叉树中的最大路径和 - 题解

## 思路

和第 543 题（直径）思路一样：在后序遍历中，对每个节点计算"经过该节点的最大路径和"。

对每个节点：
- **左贡献** = `max(0, leftGain)`（负贡献不如不选）
- **右贡献** = `max(0, rightGain)`
- **经过该节点的路径和** = `val + left + right` → 更新全局 ans
- **向上返回** = `val + max(left, right)` → 路径只能选一侧

## 解法

```cpp
class Solution {
    int ans = INT_MIN;
    int maxGain(TreeNode* node) {
        if (!node) return 0;
        int left = max(0, maxGain(node->left));
        int right = max(0, maxGain(node->right));
        ans = max(ans, node->val + left + right);
        return node->val + max(left, right);
    }
public:
    int maxPathSum(TreeNode* root) {
        ans = INT_MIN;
        maxGain(root);
        return ans;
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(h)

## 关键点

1. `max(0, gain)` 截断负贡献——如果子树贡献为负，不如不选
2. 向上返回只能选**一侧**（路径不能分叉），但更新答案时可以用**两侧**
3. `ans` 初始化为 `INT_MIN` 而非 0，因为所有节点可能都是负数
