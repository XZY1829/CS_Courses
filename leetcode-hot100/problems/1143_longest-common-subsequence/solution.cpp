// 1143. 最长公共子序列
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>
using namespace std;
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    assert(sol.longestCommonSubsequence("abcde","ace") == 3);
    assert(sol.longestCommonSubsequence("abc","abc") == 3);
    assert(sol.longestCommonSubsequence("abc","def") == 0);
    cout << "All tests passed!" << endl;
}
