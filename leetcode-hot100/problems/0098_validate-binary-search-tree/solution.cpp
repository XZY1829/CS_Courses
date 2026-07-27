// 98. 验证二叉搜索树
// https://leetcode.cn/problems/validate-binary-search-tree/

#include <iostream>
#include <climits>
#include <cassert>
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    bool isValidBST(TreeNode* root) {
        // TODO: 在此实现
    }

};

int main() {
    // [2,1,3] → true
    TreeNode* r1 = new TreeNode(2);
    r1->left = new TreeNode(1); r1->right = new TreeNode(3);
    assert(Solution().isValidBST(r1) == true);

    // [5,1,4,null,null,3,6] → false
    TreeNode* r2 = new TreeNode(5);
    r2->left = new TreeNode(1); r2->right = new TreeNode(4);
    r2->right->left = new TreeNode(3); r2->right->right = new TreeNode(6);
    assert(Solution().isValidBST(r2) == false);

    cout << "All tests passed!" << endl;
    return 0;
}
