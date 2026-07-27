// 54. 螺旋矩阵
// https://leetcode.cn/problems/spiral-matrix/

#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [[1,2,3],[4,5,6],[7,8,9]] → [1,2,3,6,9,8,7,4,5]
    vector<vector<int>> m1 = {{1,2,3},{4,5,6},{7,8,9}};
    auto r1 = sol.spiralOrder(m1);
    assert((r1 == vector<int>{1,2,3,6,9,8,7,4,5}));

    // 示例 2: [[1,2,3,4],[5,6,7,8],[9,10,11,12]] → [1,2,3,4,8,12,11,10,9,5,6,7]
    vector<vector<int>> m2 = {{1,2,3,4},{5,6,7,8},{9,10,11,12}};
    auto r2 = sol.spiralOrder(m2);
    assert((r2 == vector<int>{1,2,3,4,8,12,11,10,9,5,6,7}));

    cout << "All tests passed!" << endl;
    return 0;
}
