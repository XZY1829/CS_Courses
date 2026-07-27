// 215. 数组中的第 K 个最大元素
#include <iostream>
#include <vector>
#include <queue>
#include <cassert>
using namespace std;
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        // TODO: 在此实现
    }

};
int main() {
    Solution sol;
    vector<int> n1 = {3,2,1,5,6,4};
    assert(sol.findKthLargest(n1, 2) == 5);
    vector<int> n2 = {3,2,3,1,2,4,5,5,6};
    assert(sol.findKthLargest(n2, 4) == 4);
    cout << "All tests passed!" << endl;
}
