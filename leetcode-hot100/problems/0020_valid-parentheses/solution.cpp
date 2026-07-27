// 20. 有效的括号
#include <iostream>
#include <string>
#include <stack>
#include <cassert>
using namespace std;
class Solution {
public:
    bool isValid(string s) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    assert(sol.isValid("()") == true);
    assert(sol.isValid("()[]{}") == true);
    assert(sol.isValid("(]") == false);
    assert(sol.isValid("([])") == true);
    cout << "All tests passed!" << endl;
}
