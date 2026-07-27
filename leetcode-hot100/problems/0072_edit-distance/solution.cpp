// 72. 编辑距离
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>
using namespace std;
class Solution {
public:
    int minDistance(string word1, string word2) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    assert(sol.minDistance("horse","ros") == 3);
    assert(sol.minDistance("intention","execution") == 5);
    cout << "All tests passed!" << endl;
}
