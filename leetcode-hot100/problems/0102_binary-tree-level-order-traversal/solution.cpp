// 102. 二叉树的层序遍历
// https://leetcode.cn/problems/binary-tree-level-order-traversal/

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
    vector<vector<int>> levelOrder(TreeNode* root) {
        // TODO: 在此实现
    }

};

int main() {
    // [3,9,20,null,null,15,7] → [[3],[9,20],[15,7]]
    TreeNode* r = new TreeNode(3);
    r->left = new TreeNode(9); r->right = new TreeNode(20);
    r->right->left = new TreeNode(15); r->right->right = new TreeNode(7);
    auto res = Solution().levelOrder(r);
    assert(res.size() == 3);
    assert((res[0] == vector<int>{3}));
    assert((res[1] == vector<int>{9, 20}));
    assert((res[2] == vector<int>{15, 7}));
    cout << "All tests passed!" << endl;
    return 0;
}
