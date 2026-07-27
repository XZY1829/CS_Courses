// 101. 对称二叉树
// https://leetcode.cn/problems/symmetric-tree/

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
    bool isSymmetric(TreeNode* root) {
        // TODO: 在此实现
    }

};

int main() {
    // [1,2,2,3,4,4,3] → true
    TreeNode* r1 = new TreeNode(1);
    r1->left = new TreeNode(2); r1->right = new TreeNode(2);
    r1->left->left = new TreeNode(3); r1->left->right = new TreeNode(4);
    r1->right->left = new TreeNode(4); r1->right->right = new TreeNode(3);
    assert(Solution().isSymmetric(r1) == true);

    // [1,2,2,null,3,null,3] → false
    TreeNode* r2 = new TreeNode(1);
    r2->left = new TreeNode(2); r2->right = new TreeNode(2);
    r2->left->right = new TreeNode(3); r2->right->right = new TreeNode(3);
    assert(Solution().isSymmetric(r2) == false);

    cout << "All tests passed!" << endl;
    return 0;
}
