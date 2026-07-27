// 48. 旋转图像
// https://leetcode.cn/problems/rotate-image/

#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
using namespace std;

class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [[1,2,3],[4,5,6],[7,8,9]] → [[7,4,1],[8,5,2],[9,6,3]]
    vector<vector<int>> m1 = {{1,2,3},{4,5,6},{7,8,9}};
    sol.rotate(m1);
    assert((m1 == vector<vector<int>>{{7,4,1},{8,5,2},{9,6,3}}));

    // 示例 2
    vector<vector<int>> m2 = {{5,1,9,11},{2,4,8,10},{13,3,6,7},{15,14,12,16}};
    sol.rotate(m2);
    assert((m2 == vector<vector<int>>{{15,13,2,5},{14,3,4,1},{12,6,8,9},{16,7,10,11}}));

    cout << "All tests passed!" << endl;
    return 0;
}
