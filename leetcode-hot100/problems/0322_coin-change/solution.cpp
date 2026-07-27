// 322. 零钱兑换
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
using namespace std;
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    vector<int> c1={1,2,5}; assert(sol.coinChange(c1,11)==3);
    vector<int> c2={2}; assert(sol.coinChange(c2,3)==-1);
    vector<int> c3={1}; assert(sol.coinChange(c3,0)==0);
    cout << "All tests passed!" << endl;
}
