// 239. 滑动窗口最大值
// https://leetcode.cn/problems/sliding-window-maximum/

#include <iostream>
#include <vector>
#include <deque>
#include <cassert>
using namespace std;

class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [1,3,-1,-3,5,3,6,7], k=3 → [3,3,5,5,6,7]
    vector<int> n1 = {1, 3, -1, -3, 5, 3, 6, 7};
    auto r1 = sol.maxSlidingWindow(n1, 3);
    assert((r1 == vector<int>{3, 3, 5, 5, 6, 7}));

    // 示例 2: [1], k=1 → [1]
    vector<int> n2 = {1};
    auto r2 = sol.maxSlidingWindow(n2, 1);
    assert((r2 == vector<int>{1}));

    cout << "All tests passed!" << endl;
    return 0;
}
