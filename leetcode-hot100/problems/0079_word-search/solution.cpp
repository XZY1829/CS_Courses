// 79. 单词搜索
// https://leetcode.cn/problems/word-search/

#include <iostream>
#include <vector>
#include <string>
#include <cassert>
using namespace std;

class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;
    vector<vector<char>> b = {{'A','B','C','E'},{'S','F','C','S'},{'A','D','E','E'}};
    assert(sol.exist(b, "ABCCED") == true);
    b = {{'A','B','C','E'},{'S','F','C','S'},{'A','D','E','E'}};
    assert(sol.exist(b, "SEE") == true);
    b = {{'A','B','C','E'},{'S','F','C','S'},{'A','D','E','E'}};
    assert(sol.exist(b, "ABCB") == false);
    cout << "All tests passed!" << endl;
    return 0;
}
