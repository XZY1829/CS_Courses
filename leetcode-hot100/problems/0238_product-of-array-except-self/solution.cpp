// 238. 除了自身以外数组的乘积
// https://leetcode.cn/problems/product-of-array-except-self/

#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [1,2,3,4] → [24,12,8,6]
    vector<int> n1 = {1, 2, 3, 4};
    auto r1 = sol.productExceptSelf(n1);
    assert((r1 == vector<int>{24, 12, 8, 6}));

    // 示例 2: [-1,1,0,-3,3] → [0,0,9,0,0]
    vector<int> n2 = {-1, 1, 0, -3, 3};
    auto r2 = sol.productExceptSelf(n2);
    assert((r2 == vector<int>{0, 0, 9, 0, 0}));

    cout << "All tests passed!" << endl;
    return 0;
}
