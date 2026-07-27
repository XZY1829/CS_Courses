// 78. 子集
// https://leetcode.cn/problems/subsets/

#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;
    vector<int> n1 = {1,2,3};
    assert(sol.subsets(n1).size() == 8);  // 2^3
    vector<int> n2 = {0};
    assert(sol.subsets(n2).size() == 2);
    cout << "All tests passed!" << endl;
    return 0;
}
