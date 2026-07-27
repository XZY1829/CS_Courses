// 347. 前 K 个高频元素
#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <cassert>
#include <algorithm>
using namespace std;
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    vector<int> n1 = {1,1,1,2,2,3};
    auto r1 = sol.topKFrequent(n1, 2);
    sort(r1.begin(), r1.end());
    assert((r1 == vector<int>{1, 2}));
    vector<int> n2 = {1};
    assert(sol.topKFrequent(n2, 1).size() == 1);
    cout << "All tests passed!" << endl;
}
