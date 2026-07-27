// 416. 分割等和子集
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
using namespace std;
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    vector<int> n1={1,5,11,5}; assert(sol.canPartition(n1)==true);
    vector<int> n2={1,2,3,5}; assert(sol.canPartition(n2)==false);
    cout << "All tests passed!" << endl;
}
