// 226. 翻转二叉树
// https://leetcode.cn/problems/invert-binary-tree/

#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        // TODO: 在此实现
    }

};

int main() {
    // [4,2,7,1,3,6,9] → [4,7,2,9,6,3,1]
    TreeNode* r = new TreeNode(4);
    r->left = new TreeNode(2); r->right = new TreeNode(7);
    r->left->left = new TreeNode(1); r->left->right = new TreeNode(3);
    r->right->left = new TreeNode(6); r->right->right = new TreeNode(9);
    Solution().invertTree(r);
    assert(r->left->val == 7 && r->right->val == 2);
    assert(r->left->left->val == 9 && r->right->right->val == 1);
    cout << "All tests passed!" << endl;
    return 0;
}
