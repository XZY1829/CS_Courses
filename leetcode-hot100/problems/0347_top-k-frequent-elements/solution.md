# 347. 前 K 个高频元素 - 题解

## 思路

1. 统计频率（哈希表）
2. 用大小为 k 的小顶堆维护频率最高的 k 个元素

也可以用桶排序 O(n) 解决。

## 解法

```cpp
vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int, int> freq;
    for (int n : nums) freq[n]++;
    // 小顶堆，按频率排序
    auto cmp = [](pair<int,int>& a, pair<int,int>& b) { return a.second > b.second; };
    priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(cmp)> pq(cmp);
    for (auto& [val, cnt] : freq) { pq.push({val,cnt}); if (pq.size()>k) pq.pop(); }
    vector<int> result;
    while (!pq.empty()) { result.push_back(pq.top().first); pq.pop(); }
    return result;
}
```

## 复杂度

- **时间**：O(n log k)　　**空间**：O(n)
