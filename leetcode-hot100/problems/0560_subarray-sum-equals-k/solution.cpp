// 560. 和为 K 的子数组
// https://leetcode.cn/problems/subarray-sum-equals-k/

#include <iostream>
#include <vector>
#include <unordered_map>
#include <cassert>
using namespace std;

class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [1,1,1], k=2 → 2
    vector<int> n1 = {1, 1, 1};
    assert(sol.subarraySum(n1, 2) == 2);

    // 示例 2: [1,2,3], k=3 → 2 (子数组 [1,2] 和 [3])
    vector<int> n2 = {1, 2, 3};
    assert(sol.subarraySum(n2, 3) == 2);

    // 含负数
    vector<int> n3 = {1, -1, 0};
    assert(sol.subarraySum(n3, 0) == 3);

    cout << "All tests passed!" << endl;
    return 0;
}
