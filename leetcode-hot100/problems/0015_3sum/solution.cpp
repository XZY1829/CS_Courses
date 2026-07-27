// 15. 三数之和
// https://leetcode.cn/problems/3sum/

#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
using namespace std;

class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [-1,0,1,2,-1,-4] → [[-1,-1,2],[-1,0,1]]
    vector<int> n1 = {-1, 0, 1, 2, -1, -4};
    auto r1 = sol.threeSum(n1);
    assert(r1.size() == 2);

    // 示例 2: [0,1,1] → []
    vector<int> n2 = {0, 1, 1};
    auto r2 = sol.threeSum(n2);
    assert(r2.empty());

    // 示例 3: [0,0,0] → [[0,0,0]]
    vector<int> n3 = {0, 0, 0};
    auto r3 = sol.threeSum(n3);
    assert(r3.size() == 1);
    assert((r3[0] == vector<int>{0, 0, 0}));

    cout << "All tests passed!" << endl;
    return 0;
}
