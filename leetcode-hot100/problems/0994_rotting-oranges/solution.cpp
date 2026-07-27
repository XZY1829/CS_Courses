// 994. 腐烂的橘子
// https://leetcode.cn/problems/rotting-oranges/

#include <iostream>
#include <vector>
#include <queue>
#include <cassert>
using namespace std;

class Solution {
public:
    int orangesRotting(vector<vector<int>>& grid) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;
    // [[2,1,1],[1,1,0],[0,1,1]] → 4
    vector<vector<int>> g1 = {{2,1,1},{1,1,0},{0,1,1}};
    assert(sol.orangesRotting(g1) == 4);
    // [[2,1,1],[0,1,1],[1,0,1]] → -1
    vector<vector<int>> g2 = {{2,1,1},{0,1,1},{1,0,1}};
    assert(sol.orangesRotting(g2) == -1);
    // [[0,2]] → 0
    vector<vector<int>> g3 = {{0,2}};
    assert(sol.orangesRotting(g3) == 0);
    cout << "All tests passed!" << endl;
    return 0;
}
