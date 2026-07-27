// 104. 二叉树的最大深度
// https://leetcode.cn/problems/maximum-depth-of-binary-tree/

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
    int maxDepth(TreeNode* root) {
        // TODO: 在此实现
    }

};

int main() {
    // [3,9,20,null,null,15,7] → 3
    TreeNode* r = new TreeNode(3);
    r->left = new TreeNode(9);
    r->right = new TreeNode(20);
    r->right->left = new TreeNode(15);
    r->right->right = new TreeNode(7);
    assert(Solution().maxDepth(r) == 3);
    assert(Solution().maxDepth(nullptr) == 0);
    cout << "All tests passed!" << endl;
    return 0;
}
