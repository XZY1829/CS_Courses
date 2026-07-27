// 53. 最大子数组和
// https://leetcode.cn/problems/maximum-subarray/

#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <cassert>
using namespace std;

class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [-2,1,-3,4,-1,2,1,-5,4] → 6
    vector<int> n1 = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    assert(sol.maxSubArray(n1) == 6);

    // 示例 2: [1] → 1
    vector<int> n2 = {1};
    assert(sol.maxSubArray(n2) == 1);

    // 示例 3: [5,4,-1,7,8] → 23
    vector<int> n3 = {5, 4, -1, 7, 8};
    assert(sol.maxSubArray(n3) == 23);

    // 全负数
    vector<int> n4 = {-3, -2, -1};
    assert(sol.maxSubArray(n4) == -1);

    cout << "All tests passed!" << endl;
    return 0;
}
