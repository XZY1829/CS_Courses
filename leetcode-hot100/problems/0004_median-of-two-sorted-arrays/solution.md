# 4. 寻找两个正序数组的中位数 - 题解

## 思路

在较短数组上二分，找到一个分割点 `i`，使得两个数组的左半部分恰好有 `(m+n+1)/2` 个元素，且满足 `left1 <= right2` 和 `left2 <= right1`。

## 解法

```cpp
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    if (nums1.size() > nums2.size()) swap(nums1, nums2);
    int m = nums1.size(), n = nums2.size();
    int lo = 0, hi = m;
    while (lo <= hi) {
        int i = lo + (hi - lo) / 2;
        int j = (m + n + 1) / 2 - i;
        int left1 = i==0 ? INT_MIN : nums1[i-1];
        int right1 = i==m ? INT_MAX : nums1[i];
        int left2 = j==0 ? INT_MIN : nums2[j-1];
        int right2 = j==n ? INT_MAX : nums2[j];
        if (left1<=right2 && left2<=right1) {
            if ((m+n)%2==1) return max(left1,left2);
            return (max(left1,left2)+min(right1,right2))/2.0;
        }
        else if (left1>right2) hi = i-1;
        else lo = i+1;
    }
    return 0;
}
```

## 复杂度

- **时间**：O(log min(m,n))
- **空间**：O(1)

## 关键点

1. 在较短数组上二分，确保 `j` 不越界
2. 边界用 `INT_MIN/INT_MAX` 处理分割点在端点的情况
3. 这是 Hot 100 中最难的题之一，理解"将两个数组同时分成左右两半"是核心
