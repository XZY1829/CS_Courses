// 198. 打家劫舍
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
using namespace std;
class Solution {
public:
    int rob(vector<int>& nums) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    vector<int> n1={1,2,3,1}; assert(sol.rob(n1)==4);
    vector<int> n2={2,7,9,3,1}; assert(sol.rob(n2)==12);
    cout << "All tests passed!" << endl;
}
