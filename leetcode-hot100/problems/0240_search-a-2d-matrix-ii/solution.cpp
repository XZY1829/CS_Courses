// 240. 搜索二维矩阵 II
// https://leetcode.cn/problems/search-a-2d-matrix-ii/

#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    vector<vector<int>> matrix = {
        {1,4,7,11,15},
        {2,5,8,12,19},
        {3,6,9,16,22},
        {10,13,14,17,24},
        {18,21,23,26,30}
    };

    // 示例 1: target=5 → true
    assert(sol.searchMatrix(matrix, 5) == true);

    // 示例 2: target=20 → false
    assert(sol.searchMatrix(matrix, 20) == false);

    cout << "All tests passed!" << endl;
    return 0;
}
