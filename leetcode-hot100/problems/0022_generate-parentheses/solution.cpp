// 22. 括号生成
// https://leetcode.cn/problems/generate-parentheses/

#include <iostream>
#include <vector>
#include <string>
#include <cassert>
using namespace std;

class Solution {
public:
    vector<string> generateParenthesis(int n) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;
    assert(sol.generateParenthesis(3).size() == 5);
    assert(sol.generateParenthesis(1).size() == 1);
    cout << "All tests passed!" << endl;
    return 0;
}
