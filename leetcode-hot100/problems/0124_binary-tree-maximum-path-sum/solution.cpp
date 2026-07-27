// 124. 二叉树中的最大路径和
// https://leetcode.cn/problems/binary-tree-maximum-path-sum/

#include <iostream>
#include <algorithm>
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
    int maxPathSum(TreeNode* root) {
        // TODO: 在此实现
    }

};

int main() {
    // [1,2,3] → 6 (路径 2→1→3)
    TreeNode* r1 = new TreeNode(1);
    r1->left = new TreeNode(2); r1->right = new TreeNode(3);
    assert(Solution().maxPathSum(r1) == 6);

    // [-10,9,20,null,null,15,7] → 42 (路径 15→20→7)
    TreeNode* r2 = new TreeNode(-10);
    r2->left = new TreeNode(9); r2->right = new TreeNode(20);
    r2->right->left = new TreeNode(15); r2->right->right = new TreeNode(7);
    assert(Solution().maxPathSum(r2) == 42);

    cout << "All tests passed!" << endl;
    return 0;
}
