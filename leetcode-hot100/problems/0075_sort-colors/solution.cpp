// 75. 颜色分类
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;
class Solution {
public:
    void sortColors(vector<int>& nums) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    vector<int> n1={2,0,2,1,1,0}; sol.sortColors(n1); assert((n1==vector<int>{0,0,1,1,2,2}));
    vector<int> n2={2,0,1}; sol.sortColors(n2); assert((n2==vector<int>{0,1,2}));
    cout << "All tests passed!" << endl;
}
