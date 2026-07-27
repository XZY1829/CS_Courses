// 543. 二叉树的直径
// https://leetcode.cn/problems/diameter-of-binary-tree/

#include <iostream>
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
    int diameterOfBinaryTree(TreeNode* root) {
        // TODO: 在此实现
    }

};

int main() {
    // [1,2,3,4,5] → 3 (路径 4→2→1→3 或 5→2→1→3)
    TreeNode* r = new TreeNode(1);
    r->left = new TreeNode(2); r->right = new TreeNode(3);
    r->left->left = new TreeNode(4); r->left->right = new TreeNode(5);
    assert(Solution().diameterOfBinaryTree(r) == 3);

    // [1,2] → 1
    TreeNode* r2 = new TreeNode(1);
    r2->left = new TreeNode(2);
    assert(Solution().diameterOfBinaryTree(r2) == 1);

    cout << "All tests passed!" << endl;
    return 0;
}
