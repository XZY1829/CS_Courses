// 51. N 皇后
// https://leetcode.cn/problems/n-queens/

#include <iostream>
#include <vector>
#include <string>
#include <cassert>
using namespace std;

class Solution {
public:
    vector<vector<string>> solveNQueens(int n) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;
    assert(sol.solveNQueens(4).size() == 2);
    assert(sol.solveNQueens(1).size() == 1);
    cout << "All tests passed!" << endl;
    return 0;
}
