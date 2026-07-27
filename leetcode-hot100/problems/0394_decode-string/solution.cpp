// 394. 字符串解码
#include <iostream>
#include <string>
#include <stack>
#include <cassert>
using namespace std;
class Solution {
public:
    string decodeString(string s) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    assert(sol.decodeString("3[a]2[bc]") == "aaabcbc");
    assert(sol.decodeString("3[a2[c]]") == "accaccacc");
    assert(sol.decodeString("2[abc]3[cd]ef") == "abcabccdcdcdef");
    cout << "All tests passed!" << endl;
}
