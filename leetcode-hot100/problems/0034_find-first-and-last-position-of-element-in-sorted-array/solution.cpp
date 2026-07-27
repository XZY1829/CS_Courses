// 34. 在排序数组中查找元素的第一个和最后一个位置
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    vector<int> n1 = {5,7,7,8,8,10};
    assert((sol.searchRange(n1,8) == vector<int>{3,4}));
    assert((sol.searchRange(n1,6) == vector<int>{-1,-1}));
    vector<int> n2 = {};
    assert((sol.searchRange(n2,0) == vector<int>{-1,-1}));
    cout << "All tests passed!" << endl;
}
