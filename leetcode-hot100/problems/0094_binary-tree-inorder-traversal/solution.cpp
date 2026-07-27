// 94. 二叉树的中序遍历
// https://leetcode.cn/problems/binary-tree-inorder-traversal/

#include <iostream>
#include <vector>
#include <stack>
#include <cassert>
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        // TODO: 在此实现
    }

};

int main() {
    // 示例 1: [1,null,2,3] → [1,3,2]
    TreeNode* r1 = new TreeNode(1);
    r1->right = new TreeNode(2);
    r1->right->left = new TreeNode(3);
    assert((Solution().inorderTraversal(r1) == vector<int>{1,3,2}));

    // 示例 2: [] → []
    assert(Solution().inorderTraversal(nullptr).empty());

    // 示例 3: [1] → [1]
    assert((Solution().inorderTraversal(new TreeNode(1)) == vector<int>{1}));

    cout << "All tests passed!" << endl;
    return 0;
}
