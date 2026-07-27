// 283. 移动零
// https://leetcode.cn/problems/move-zeroes/

#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [0,1,0,3,12] → [1,3,12,0,0]
    vector<int> n1 = {0, 1, 0, 3, 12};
    sol.moveZeroes(n1);
    assert((n1 == vector<int>{1, 3, 12, 0, 0}));

    // 示例 2: [0] → [0]
    vector<int> n2 = {0};
    sol.moveZeroes(n2);
    assert((n2 == vector<int>{0}));

    // 补充: 全零
    vector<int> n3 = {0, 0, 0};
    sol.moveZeroes(n3);
    assert((n3 == vector<int>{0, 0, 0}));

    // 补充: 无零
    vector<int> n4 = {1, 2, 3};
    sol.moveZeroes(n4);
    assert((n4 == vector<int>{1, 2, 3}));

    cout << "All tests passed!" << endl;
    return 0;
}
