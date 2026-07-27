// 199. 二叉树的右视图
// https://leetcode.cn/problems/binary-tree-right-side-view/

#include <iostream>
#include <vector>
#include <queue>
#include <cassert>
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
        // TODO: 在此实现
    }

};

int main() {
    // [1,2,3,null,5,null,4] → [1,3,4]
    TreeNode* r = new TreeNode(1);
    r->left = new TreeNode(2); r->right = new TreeNode(3);
    r->left->right = new TreeNode(5); r->right->right = new TreeNode(4);
    auto res = Solution().rightSideView(r);
    assert((res == vector<int>{1, 3, 4}));
    cout << "All tests passed!" << endl;
    return 0;
}
