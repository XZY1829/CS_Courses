// 128. 最长连续序列
// https://leetcode.cn/problems/longest-consecutive-sequence/

#include <iostream>
#include <vector>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <cassert>
using namespace std;

class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [100,4,200,1,3,2] → 4 (序列 [1,2,3,4])
    vector<int> n1 = {100, 4, 200, 1, 3, 2};
    assert(sol.longestConsecutive(n1) == 4);

    // 示例 2: [0,3,7,2,5,8,4,6,0,1] → 9 (序列 [0,1,2,3,4,5,6,7,8])
    vector<int> n2 = {0, 3, 7, 2, 5, 8, 4, 6, 0, 1};
    assert(sol.longestConsecutive(n2) == 9);

    // 示例 3: [1,0,1,2] → 3 (序列 [0,1,2])
    vector<int> n3 = {1, 0, 1, 2};
    assert(sol.longestConsecutive(n3) == 3);

    // 边界: 空数组
    vector<int> n4 = {};
    assert(sol.longestConsecutive(n4) == 0);

    cout << "All tests passed!" << endl;
    return 0;
}
