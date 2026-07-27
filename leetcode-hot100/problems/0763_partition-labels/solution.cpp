// 763. 划分字母区间
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>
using namespace std;
class Solution {
public:
    vector<int> partitionLabels(string s) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    auto r1 = sol.partitionLabels("ababcbacadefegdehijhklij");
    assert((r1 == vector<int>{9,7,8}));
    auto r2 = sol.partitionLabels("eccbbbbdec");
    assert((r2 == vector<int>{10}));
    cout << "All tests passed!" << endl;
}
