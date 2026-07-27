// 4. 寻找两个正序数组的中位数
#include <algorithm>
#include <cassert>
#include <climits>
#include <iostream>
#include <vector>

using namespace std;
class Solution {
public:
    // double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    //     vector<int> nums;
    //     nums.reserve(nums1.size() + nums2.size());

    //     nums.insert(nums.end(), nums1.begin(), nums1.end());
    //     nums.insert(nums.end(), nums2.begin(), nums2.end());

    //     sort(nums.begin(), nums.end());

    //     int n = nums.size();

    //     if (n % 2 == 1) {
    //         return nums[n / 2];
    //     } else {
    //         return (nums[n / 2 - 1] + nums[n / 2]) / 2.0;
    //     }
    // }
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        // TODO: 在此实现
        
    }
};
int main() {
    Solution sol;
    vector<int> a1 = {1, 3}, b1 = {2};
    assert(sol.findMedianSortedArrays(a1, b1) == 2.0);
    vector<int> a2 = {1, 2}, b2 = {3, 4};
    assert(sol.findMedianSortedArrays(a2, b2) == 2.50000);
    cout << "All tests passed!" << endl;
}
