# 215. 数组中的第 K 个最大元素 - 题解

## 思路

维护一个大小为 k 的**小顶堆**，遍历完后堆顶就是第 k 大。

也可以用快速选择（QuickSelect）达到平均 O(n)。

## 解法

```cpp
int findKthLargest(vector<int>& nums, int k) {
    priority_queue<int, vector<int>, greater<int>> minHeap;
    for (int n : nums) {
        minHeap.push(n);
        if (minHeap.size() > k) minHeap.pop();
    }
    return minHeap.top();
}
```

## 复杂度

- **时间**：O(n log k)　　**空间**：O(k)

## 关键点

1. 小顶堆维护最大的 k 个元素，堆顶是 k 个中最小的 = 第 k 大
2. QuickSelect 平均 O(n) 但最坏 O(n²)
