// 105. 从前序与中序遍历序列构造二叉树
// https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/

#include <iostream>
#include <vector>
#include <unordered_map>
#include <cassert>
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;
    // preorder=[3,9,20,15,7], inorder=[9,3,15,20,7] → 根=3
    vector<int> pre = {3, 9, 20, 15, 7}, in = {9, 3, 15, 20, 7};
    TreeNode* root = sol.buildTree(pre, in);
    assert(root->val == 3);
    assert(root->left->val == 9);
    assert(root->right->val == 20);
    cout << "All tests passed!" << endl;
    return 0;
}
