// 121. 买卖股票的最佳时机
#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <cassert>
using namespace std;
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    vector<int> p1 = {7,1,5,3,6,4};
    assert(sol.maxProfit(p1) == 5);
    vector<int> p2 = {7,6,4,3,1};
    assert(sol.maxProfit(p2) == 0);
    cout << "All tests passed!" << endl;
}
