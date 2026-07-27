// 33. 搜索旋转排序数组
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;
class Solution {
public:
    int search(vector<int>& nums, int target) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    vector<int> n1 = {4,5,6,7,0,1,2};
    assert(sol.search(n1, 0) == 4);
    assert(sol.search(n1, 3) == -1);
    vector<int> n2 = {1};
    assert(sol.search(n2, 0) == -1);
    cout << "All tests passed!" << endl;
}
