// 5. 最长回文子串
#include <iostream>
#include <string>
#include <cassert>
using namespace std;
class Solution {
public:
    string longestPalindrome(string s) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    string r1 = sol.longestPalindrome("babad");
    assert(r1 == "bab" || r1 == "aba");
    assert(sol.longestPalindrome("cbbd") == "bb");
    cout << "All tests passed!" << endl;
}
