// 42. 接雨水
// https://leetcode.cn/problems/trapping-rain-water/

#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
using namespace std;

class Solution {
public:
    int trap(vector<int>& height) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [0,1,0,2,1,0,1,3,2,1,2,1] → 6
    vector<int> h1 = {0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1};
    assert(sol.trap(h1) == 6);

    // 示例 2: [4,2,0,3,2,5] → 9
    vector<int> h2 = {4, 2, 0, 3, 2, 5};
    assert(sol.trap(h2) == 9);

    // 边界: 单个 / 两个
    vector<int> h3 = {3};
    assert(sol.trap(h3) == 0);
    vector<int> h4 = {3, 1};
    assert(sol.trap(h4) == 0);

    cout << "All tests passed!" << endl;
    return 0;
}
