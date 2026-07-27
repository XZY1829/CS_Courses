// 189. 轮转数组
// https://leetcode.cn/problems/rotate-array/

#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
using namespace std;

class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [1,2,3,4,5,6,7], k=3 → [5,6,7,1,2,3,4]
    vector<int> n1 = {1, 2, 3, 4, 5, 6, 7};
    sol.rotate(n1, 3);
    assert((n1 == vector<int>{5, 6, 7, 1, 2, 3, 4}));

    // 示例 2: [-1,-100,3,99], k=2 → [3,99,-1,-100]
    vector<int> n2 = {-1, -100, 3, 99};
    sol.rotate(n2, 2);
    assert((n2 == vector<int>{3, 99, -1, -100}));

    // k > n
    vector<int> n3 = {1, 2};
    sol.rotate(n3, 3);  // 等价于 k=1
    assert((n3 == vector<int>{2, 1}));

    cout << "All tests passed!" << endl;
    return 0;
}
