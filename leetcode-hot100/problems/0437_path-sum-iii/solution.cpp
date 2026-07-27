// 437. 路径总和 III
// https://leetcode.cn/problems/path-sum-iii/

#include <iostream>
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
    int pathSum(TreeNode* root, int targetSum) {
        // TODO: 在此实现
    }

};

int main() {
    // [10,5,-3,3,2,null,11,3,-2,null,1], targetSum=8 → 3
    TreeNode* r = new TreeNode(10);
    r->left = new TreeNode(5); r->right = new TreeNode(-3);
    r->left->left = new TreeNode(3); r->left->right = new TreeNode(2);
    r->right->right = new TreeNode(11);
    r->left->left->left = new TreeNode(3); r->left->left->right = new TreeNode(-2);
    r->left->right->right = new TreeNode(1);
    assert(Solution().pathSum(r, 8) == 3);
    cout << "All tests passed!" << endl;
    return 0;
}
