// 1. 两数之和
// https://leetcode.cn/problems/two-sum/

#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        // TODO: 在此实现
        unordered_map<int, int> seen;
        for (int i = 0; i < nums.size(); i++) {
            int rest = target - nums[i];
            if (seen.count(rest)) {
                return {seen[rest], i};
            }
            seen[nums[i]] = i;
        }
        return {};
    }
};

int main() {
    Solution sol;

    // 示例 1: nums = [2,7,11,15], target = 9 → [0,1]
    vector<int> n1 = {2, 7, 11, 15};
    auto r1 = sol.twoSum(n1, 9);
    assert(r1[0] == 0 && r1[1] == 1);

    // 示例 2: nums = [3,2,4], target = 6 → [1,2]
    vector<int> n2 = {3, 2, 4};
    auto r2 = sol.twoSum(n2, 6);
    assert(r2[0] == 1 && r2[1] == 2);

    // 示例 3: nums = [3,3], target = 6 → [0,1]
    vector<int> n3 = {3, 3};
    auto r3 = sol.twoSum(n3, 6);
    assert(r3[0] == 0 && r3[1] == 1);

    cout << "All tests passed!" << endl;
    return 0;
}
