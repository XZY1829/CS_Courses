// 74. 搜索二维矩阵
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
    vector<vector<int>> m = {{1,3,5,7},{10,11,16,20},{23,30,34,60}};
    assert(sol.searchMatrix(m, 3) == true);
    assert(sol.searchMatrix(m, 13) == false);
    cout << "All tests passed!" << endl;
}
