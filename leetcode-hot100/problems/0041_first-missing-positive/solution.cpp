// 41. 缺失的第一个正数
// https://leetcode.cn/problems/first-missing-positive/

#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [1,2,0] → 3
    vector<int> n1 = {1, 2, 0};
    assert(sol.firstMissingPositive(n1) == 3);

    // 示例 2: [3,4,-1,1] → 2
    vector<int> n2 = {3, 4, -1, 1};
    assert(sol.firstMissingPositive(n2) == 2);

    // 示例 3: [7,8,9,11,12] → 1
    vector<int> n3 = {7, 8, 9, 11, 12};
    assert(sol.firstMissingPositive(n3) == 1);

    // 连续 [1,2,3] → 4
    vector<int> n4 = {1, 2, 3};
    assert(sol.firstMissingPositive(n4) == 4);

    cout << "All tests passed!" << endl;
    return 0;
}
