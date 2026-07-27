// 56. 合并区间
// https://leetcode.cn/problems/merge-intervals/

#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
using namespace std;

class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [[1,3],[2,6],[8,10],[15,18]] → [[1,6],[8,10],[15,18]]
    vector<vector<int>> i1 = {{1,3},{2,6},{8,10},{15,18}};
    auto r1 = sol.merge(i1);
    assert((r1 == vector<vector<int>>{{1,6},{8,10},{15,18}}));

    // 示例 2: [[1,4],[4,5]] → [[1,5]]
    vector<vector<int>> i2 = {{1,4},{4,5}};
    auto r2 = sol.merge(i2);
    assert((r2 == vector<vector<int>>{{1,5}}));

    // 示例 3: [[4,7],[1,4]] → [[1,7]]
    vector<vector<int>> i3 = {{4,7},{1,4}};
    auto r3 = sol.merge(i3);
    assert((r3 == vector<vector<int>>{{1,7}}));

    cout << "All tests passed!" << endl;
    return 0;
}
