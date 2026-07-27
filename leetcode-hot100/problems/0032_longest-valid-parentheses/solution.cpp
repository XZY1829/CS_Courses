// 32. 最长有效括号
#include <iostream>
#include <string>
#include <stack>
#include <algorithm>
#include <cassert>
using namespace std;
class Solution {
public:
    int longestValidParentheses(string s) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    assert(sol.longestValidParentheses("(()") == 2);
    assert(sol.longestValidParentheses(")()())") == 4);
    assert(sol.longestValidParentheses("") == 0);
    cout << "All tests passed!" << endl;
}
