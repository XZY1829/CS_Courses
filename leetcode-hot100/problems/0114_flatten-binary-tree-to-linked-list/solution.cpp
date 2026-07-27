// 114. 二叉树展开为链表
// https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/

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
    void flatten(TreeNode* root) {
        // TODO: 在此实现
    }

};

int main() {
    // [1,2,5,3,4,null,6] → [1,null,2,null,3,null,4,null,5,null,6]
    TreeNode* r = new TreeNode(1);
    r->left = new TreeNode(2); r->right = new TreeNode(5);
    r->left->left = new TreeNode(3); r->left->right = new TreeNode(4);
    r->right->right = new TreeNode(6);

    Solution().flatten(r);
    int expected[] = {1, 2, 3, 4, 5, 6};
    TreeNode* cur = r;
    for (int i = 0; i < 6; i++) {
        assert(cur->val == expected[i]);
        assert(cur->left == nullptr);
        cur = cur->right;
    }
    cout << "All tests passed!" << endl;
    return 0;
}
