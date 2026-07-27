// 287. 寻找重复数
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    vector<int> n1={1,3,4,2,2}; assert(sol.findDuplicate(n1)==2);
    vector<int> n2={3,1,3,4,2}; assert(sol.findDuplicate(n2)==3);
    vector<int> n3={3,3,3,3,3}; assert(sol.findDuplicate(n3)==3);
    cout << "All tests passed!" << endl;
}
