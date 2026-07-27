// 55. 跳跃游戏
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
using namespace std;
class Solution {
public:
    bool canJump(vector<int>& nums) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    vector<int> n1 = {2,3,1,1,4}; assert(sol.canJump(n1) == true);
    vector<int> n2 = {3,2,1,0,4}; assert(sol.canJump(n2) == false);
    cout << "All tests passed!" << endl;
}
