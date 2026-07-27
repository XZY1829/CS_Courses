// 739. 每日温度
#include <iostream>
#include <vector>
#include <stack>
#include <cassert>
using namespace std;
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    vector<int> t1 = {73,74,75,71,69,72,76,73};
    assert((sol.dailyTemperatures(t1) == vector<int>{1,1,4,2,1,1,0,0}));
    vector<int> t2 = {30,40,50,60};
    assert((sol.dailyTemperatures(t2) == vector<int>{1,1,1,0}));
    cout << "All tests passed!" << endl;
}
