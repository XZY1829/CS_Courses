# 295. 数据流的中位数 - 题解

## 思路

**对顶堆**：大顶堆存较小的一半，小顶堆存较大的一半。保持 `maxHeap.size() >= minHeap.size()`。

- addNum：先入大顶堆 → 大顶堆最大值转入小顶堆 → 如果小顶堆更多则转回
- findMedian：大顶堆多一个则堆顶，否则两堆顶的平均

## 解法

```cpp
class MedianFinder {
    priority_queue<int> maxHeap;
    priority_queue<int, vector<int>, greater<int>> minHeap;
public:
    void addNum(int num) {
        maxHeap.push(num);
        minHeap.push(maxHeap.top()); maxHeap.pop();
        if (minHeap.size() > maxHeap.size()) { maxHeap.push(minHeap.top()); minHeap.pop(); }
    }
    double findMedian() {
        if (maxHeap.size() > minHeap.size()) return maxHeap.top();
        return (maxHeap.top() + minHeap.top()) / 2.0;
    }
};
```

## 复杂度

- addNum：O(log n)　　findMedian：O(1)
- **空间**：O(n)
