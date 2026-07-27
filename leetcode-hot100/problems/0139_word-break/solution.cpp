// 139. 单词拆分
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
#include <cassert>
using namespace std;
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    vector<string> d1={"leet","code"}; assert(sol.wordBreak("leetcode",d1)==true);
    vector<string> d2={"apple","pen"}; assert(sol.wordBreak("applepenapple",d2)==true);
    vector<string> d3={"cats","dog","sand","and","cat"}; assert(sol.wordBreak("catsandog",d3)==false);
    cout << "All tests passed!" << endl;
}
