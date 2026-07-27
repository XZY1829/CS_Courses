// 46. 全排列
// https://leetcode.cn/problems/permutations/

#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;
    vector<int> n1 = {1,2,3};
    assert(sol.permute(n1).size() == 6);
    vector<int> n2 = {0,1};
    assert(sol.permute(n2).size() == 2);
    vector<int> n3 = {1};
    assert(sol.permute(n3).size() == 1);
    cout << "All tests passed!" << endl;
    return 0;
}
