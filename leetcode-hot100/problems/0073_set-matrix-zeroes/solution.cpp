// 73. 矩阵置零
// https://leetcode.cn/problems/set-matrix-zeroes/

#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1
    vector<vector<int>> m1 = {{1,1,1},{1,0,1},{1,1,1}};
    sol.setZeroes(m1);
    assert((m1 == vector<vector<int>>{{1,0,1},{0,0,0},{1,0,1}}));

    // 示例 2
    vector<vector<int>> m2 = {{0,1,2,0},{3,4,5,2},{1,3,1,5}};
    sol.setZeroes(m2);
    assert((m2 == vector<vector<int>>{{0,0,0,0},{0,4,5,0},{0,3,1,0}}));

    cout << "All tests passed!" << endl;
    return 0;
}
