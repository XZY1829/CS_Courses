// 200. 岛屿数量
// https://leetcode.cn/problems/number-of-islands/

#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;
    vector<vector<char>> g1 = {{'1','1','1','1','0'},{'1','1','0','1','0'},{'1','1','0','0','0'},{'0','0','0','0','0'}};
    assert(sol.numIslands(g1) == 1);
    vector<vector<char>> g2 = {{'1','1','0','0','0'},{'1','1','0','0','0'},{'0','0','1','0','0'},{'0','0','0','1','1'}};
    assert(sol.numIslands(g2) == 3);
    cout << "All tests passed!" << endl;
    return 0;
}
