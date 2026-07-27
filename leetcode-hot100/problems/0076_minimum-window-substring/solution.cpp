// 76. 最小覆盖子串
// https://leetcode.cn/problems/minimum-window-substring/

#include <iostream>
#include <string>
#include <unordered_map>
#include <climits>
#include <cassert>
using namespace std;

class Solution {
public:
    string minWindow(string s, string t) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: s="ADOBECODEBANC", t="ABC" → "BANC"
    assert(sol.minWindow("ADOBECODEBANC", "ABC") == "BANC");

    // 示例 2: s="a", t="a" → "a"
    assert(sol.minWindow("a", "a") == "a");

    // 示例 3: s="a", t="aa" → ""
    assert(sol.minWindow("a", "aa") == "");

    cout << "All tests passed!" << endl;
    return 0;
}
