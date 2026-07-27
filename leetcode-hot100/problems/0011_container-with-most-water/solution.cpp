// 11. 盛最多水的容器
// https://leetcode.cn/problems/container-with-most-water/

#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
using namespace std;

class Solution {
public:
    int maxArea(vector<int>& height) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [1,8,6,2,5,4,8,3,7] → 49
    vector<int> h1 = {1, 8, 6, 2, 5, 4, 8, 3, 7};
    assert(sol.maxArea(h1) == 49);

    // 示例 2: [1,1] → 1
    vector<int> h2 = {1, 1};
    assert(sol.maxArea(h2) == 1);

    cout << "All tests passed!" << endl;
    return 0;
}
