// 35. 搜索插入位置
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    vector<int> n = {1,3,5,6};
    assert(sol.searchInsert(n, 5) == 2);
    assert(sol.searchInsert(n, 2) == 1);
    assert(sol.searchInsert(n, 7) == 4);
    assert(sol.searchInsert(n, 0) == 0);
    cout << "All tests passed!" << endl;
}
