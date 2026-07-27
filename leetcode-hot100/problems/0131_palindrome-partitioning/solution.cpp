// 131. 分割回文串
// https://leetcode.cn/problems/palindrome-partitioning/

#include <iostream>
#include <vector>
#include <string>
#include <cassert>
using namespace std;

class Solution {
public:
    vector<vector<string>> partition(string s) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;
    auto r1 = sol.partition("aab");
    assert(r1.size() == 2);  // [["a","a","b"],["aa","b"]]
    auto r2 = sol.partition("a");
    assert(r2.size() == 1);
    cout << "All tests passed!" << endl;
    return 0;
}
