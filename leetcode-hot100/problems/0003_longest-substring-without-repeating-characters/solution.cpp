// 3. 无重复字符的最长子串
// https://leetcode.cn/problems/longest-substring-without-repeating-characters/

#include <cassert>
#include <iostream>
#include <string>
#include <unordered_set>

using namespace std;

class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        // TODO: 在此实现
        unordered_set<char> window;
        int l = 0, ans = 0;
        for (int r = 0; r < s.length(); r++) {
            while (window.count(s[r])) {
                window.erase(s[l]);
                l++;
            }
            window.insert(s[r]);
            ans = max(ans, r - l + 1);
        }
        return ans;
    }
};

int main() {
    Solution sol;

    // 示例 1: "abcabcbb" → 3
    assert(sol.lengthOfLongestSubstring("abcabcbb") == 3);

    // 示例 2: "bbbbb" → 1
    assert(sol.lengthOfLongestSubstring("bbbbb") == 1);

    // 示例 3: "pwwkew" → 3
    assert(sol.lengthOfLongestSubstring("pwwkew") == 3);

    // 边界: 空串
    assert(sol.lengthOfLongestSubstring("") == 0);

    cout << "All tests passed!" << endl;
    return 0;
}
