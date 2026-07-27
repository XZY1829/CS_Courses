// 153. 寻找旋转排序数组中的最小值
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;
class Solution {
public:
    int findMin(vector<int>& nums) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    vector<int> n1 = {3,4,5,1,2};
    assert(sol.findMin(n1) == 1);
    vector<int> n2 = {4,5,6,7,0,1,2};
    assert(sol.findMin(n2) == 0);
    vector<int> n3 = {11,13,15,17};
    assert(sol.findMin(n3) == 11);
    cout << "All tests passed!" << endl;
}
