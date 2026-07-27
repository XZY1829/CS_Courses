// 438. 找到字符串中所有字母异位词
// https://leetcode.cn/problems/find-all-anagrams-in-a-string/

#include <iostream>
#include <vector>
#include <string>
#include <cassert>
using namespace std;

class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: s="cbaebabacd", p="abc" → [0,6]
    auto r1 = sol.findAnagrams("cbaebabacd", "abc");
    assert((r1 == vector<int>{0, 6}));

    // 示例 2: s="abab", p="ab" → [0,1,2]
    auto r2 = sol.findAnagrams("abab", "ab");
    assert((r2 == vector<int>{0, 1, 2}));

    // 边界: p 比 s 长
    auto r3 = sol.findAnagrams("a", "ab");
    assert(r3.empty());

    cout << "All tests passed!" << endl;
    return 0;
}
