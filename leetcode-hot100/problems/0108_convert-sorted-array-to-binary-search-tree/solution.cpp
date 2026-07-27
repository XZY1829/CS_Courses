// 108. 将有序数组转换为二叉搜索树
// https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/

#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;
    // [-10,-3,0,5,9] → 高度平衡 BST，根为 0
    vector<int> nums = {-10, -3, 0, 5, 9};
    TreeNode* root = sol.sortedArrayToBST(nums);
    assert(root->val == 0);
    assert(root->left->val == -3 || root->left->val == -10);
    cout << "All tests passed!" << endl;
    return 0;
}
