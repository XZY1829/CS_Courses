// 152. 乘积最大子数组
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
using namespace std;
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    vector<int> n1={2,3,-2,4}; assert(sol.maxProduct(n1)==6);
    vector<int> n2={-2,0,-1}; assert(sol.maxProduct(n2)==0);
    cout << "All tests passed!" << endl;
}
