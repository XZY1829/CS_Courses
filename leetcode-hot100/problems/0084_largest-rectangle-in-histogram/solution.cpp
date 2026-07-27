// 84. 柱状图中最大的矩形
#include <iostream>
#include <vector>
#include <stack>
#include <cassert>
using namespace std;
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    vector<int> h1 = {2,1,5,6,2,3};
    assert(sol.largestRectangleArea(h1) == 10);
    vector<int> h2 = {2,4};
    assert(sol.largestRectangleArea(h2) == 4);
    cout << "All tests passed!" << endl;
}
