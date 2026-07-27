// 230. 二叉搜索树中第 K 小的元素
// https://leetcode.cn/problems/kth-smallest-element-in-a-bst/

#include <iostream>
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
    int kthSmallest(TreeNode* root, int k) {
        // TODO: 在此实现
    }

};

int main() {
    // [3,1,4,null,2], k=1 → 1
    TreeNode* r = new TreeNode(3);
    r->left = new TreeNode(1); r->right = new TreeNode(4);
    r->left->right = new TreeNode(2);
    assert(Solution().kthSmallest(r, 1) == 1);
    assert(Solution().kthSmallest(r, 3) == 3);
    cout << "All tests passed!" << endl;
    return 0;
}
