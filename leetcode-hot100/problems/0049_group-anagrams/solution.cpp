// 49. 字母异位词分组
// https://leetcode.cn/problems/group-anagrams/

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cassert>
using namespace std;

class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: ["eat","tea","tan","ate","nat","bat"]
    //       → [["bat"],["nat","tan"],["ate","eat","tea"]]
    vector<string> s1 = {"eat", "tea", "tan", "ate", "nat", "bat"};
    auto r1 = sol.groupAnagrams(s1);
    assert(r1.size() == 3);
    // 验证每组内排序后一致
    for (auto& group : r1) {
        string key = group[0];
        sort(key.begin(), key.end());
        for (auto& w : group) {
            string k = w;
            sort(k.begin(), k.end());
            assert(k == key);
        }
    }

    // 示例 2: [""] → [[""]]
    vector<string> s2 = {""};
    auto r2 = sol.groupAnagrams(s2);
    assert(r2.size() == 1 && r2[0].size() == 1 && r2[0][0] == "");

    // 示例 3: ["a"] → [["a"]]
    vector<string> s3 = {"a"};
    auto r3 = sol.groupAnagrams(s3);
    assert(r3.size() == 1 && r3[0].size() == 1 && r3[0][0] == "a");

    cout << "All tests passed!" << endl;
    return 0;
}
