// 17. 电话号码的字母组合
// https://leetcode.cn/problems/letter-combinations-of-a-phone-number/

#include <iostream>
#include <vector>
#include <string>
#include <cassert>
using namespace std;

class Solution {
public:
    vector<string> letterCombinations(string digits) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;
    auto r1 = sol.letterCombinations("23");
    assert(r1.size() == 9);
    auto r2 = sol.letterCombinations("");
    assert(r2.empty());
    auto r3 = sol.letterCombinations("2");
    assert(r3.size() == 3);
    cout << "All tests passed!" << endl;
    return 0;
}
