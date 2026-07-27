// 64. 最小路径和
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
using namespace std;
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    vector<vector<int>> g1 = {{1,3,1},{1,5,1},{4,2,1}};
    assert(sol.minPathSum(g1) == 7);
    vector<vector<int>> g2 = {{1,2,3},{4,5,6}};
    assert(sol.minPathSum(g2) == 12);
    cout << "All tests passed!" << endl;
}
