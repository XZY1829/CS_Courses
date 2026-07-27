// 236. 二叉树的最近公共祖先
// https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/

#include <iostream>
#include <cassert>
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        // TODO: 在此实现
    }

};

int main() {
    // [3,5,1,6,2,0,8,null,null,7,4], p=5, q=1 → 3
    TreeNode *n3 = new TreeNode(3), *n5 = new TreeNode(5), *n1 = new TreeNode(1);
    TreeNode *n6 = new TreeNode(6), *n2 = new TreeNode(2), *n0 = new TreeNode(0);
    TreeNode *n8 = new TreeNode(8), *n7 = new TreeNode(7), *n4 = new TreeNode(4);
    n3->left = n5; n3->right = n1;
    n5->left = n6; n5->right = n2;
    n1->left = n0; n1->right = n8;
    n2->left = n7; n2->right = n4;

    Solution sol;
    assert(sol.lowestCommonAncestor(n3, n5, n1) == n3);
    assert(sol.lowestCommonAncestor(n3, n5, n4) == n5);

    cout << "All tests passed!" << endl;
    return 0;
}
