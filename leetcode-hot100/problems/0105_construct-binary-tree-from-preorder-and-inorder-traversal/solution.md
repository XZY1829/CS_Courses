# 105. 从前序与中序遍历序列构造二叉树 - 题解

## 思路

- 前序遍历的第一个元素是根
- 在中序遍历中找到根的位置，左边是左子树，右边是右子树
- 递归构建

用哈希表预存中序遍历的索引，O(1) 定位根在中序中的位置。

## 解法

```cpp
class Solution {
    unordered_map<int, int> inIdx;
    int prePos = 0;
    TreeNode* build(vector<int>& preorder, int inLo, int inHi) {
        if (inLo > inHi) return nullptr;
        int rootVal = preorder[prePos++];
        TreeNode* root = new TreeNode(rootVal);
        int mid = inIdx[rootVal];
        root->left = build(preorder, inLo, mid - 1);
        root->right = build(preorder, mid + 1, inHi);
        return root;
    }
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        prePos = 0; inIdx.clear();
        for (int i = 0; i < (int)inorder.size(); i++) inIdx[inorder[i]] = i;
        return build(preorder, 0, (int)inorder.size() - 1);
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(n)

## 关键点

1. `prePos` 是全局递增指针，按前序顺序取根；不需要单独传递 preorder 的范围
2. 必须先构建**左子树**再构建右子树（和前序遍历顺序一致）
3. 题目保证值不重复，否则无法唯一确定树
