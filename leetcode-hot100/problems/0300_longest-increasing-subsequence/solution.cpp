// 300. 最长递增子序列
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
using namespace std;
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    vector<int> n1={10,9,2,5,3,7,101,18}; assert(sol.lengthOfLIS(n1)==4);
    vector<int> n2={0,1,0,3,2,3}; assert(sol.lengthOfLIS(n2)==4);
    vector<int> n3={7,7,7,7}; assert(sol.lengthOfLIS(n3)==1);
    cout << "All tests passed!" << endl;
}
