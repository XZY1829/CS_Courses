// 39. 组合总和
// https://leetcode.cn/problems/combination-sum/

#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
using namespace std;

class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;
    vector<int> c1 = {2,3,6,7};
    auto r1 = sol.combinationSum(c1, 7);
    assert(r1.size() == 2);  // [2,2,3] 和 [7]
    vector<int> c2 = {2,3,5};
    auto r2 = sol.combinationSum(c2, 8);
    assert(r2.size() == 3);  // [2,2,2,2], [2,3,3], [3,5]
    cout << "All tests passed!" << endl;
    return 0;
}
