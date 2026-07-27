// 31. 下一个排列
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
using namespace std;
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    vector<int> n1={1,2,3}; sol.nextPermutation(n1); assert((n1==vector<int>{1,3,2}));
    vector<int> n2={3,2,1}; sol.nextPermutation(n2); assert((n2==vector<int>{1,2,3}));
    vector<int> n3={1,1,5}; sol.nextPermutation(n3); assert((n3==vector<int>{1,5,1}));
    cout << "All tests passed!" << endl;
}
