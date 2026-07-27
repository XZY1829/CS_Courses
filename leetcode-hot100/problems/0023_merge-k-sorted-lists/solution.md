# 23. 合并 K 个升序链表 - 题解

## 思路

**最小堆（优先队列）法**：

1. 把 k 个链表的头节点都放入最小堆
2. 每次弹出堆顶（最小值），接到结果链表
3. 如果弹出节点有 next，把 next 放入堆
4. 堆为空时结束

## 解法

```cpp
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        auto cmp = [](ListNode* a, ListNode* b) { return a->val > b->val; };
        priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> pq(cmp);
        for (auto* l : lists) if (l) pq.push(l);
        ListNode dummy(0);
        ListNode* tail = &dummy;
        while (!pq.empty()) {
            auto* node = pq.top(); pq.pop();
            tail->next = node;
            tail = tail->next;
            if (node->next) pq.push(node->next);
        }
        return dummy.next;
    }
};
```

## 复杂度

- **时间**：O(N log k)，N 是所有节点总数，每次堆操作 O(log k)
- **空间**：O(k)，堆中最多 k 个节点

## 其他解法

| 方法 | 时间 | 空间 |
|------|------|------|
| 最小堆 | O(N log k) | O(k) |
| 分治合并 | O(N log k) | O(log k) |
| 逐一合并 | O(Nk) | O(1) |

## 关键点

1. C++ 的 `priority_queue` 默认是最大堆，自定义比较器 `a->val > b->val` 变成最小堆
2. 分治合并也很常用：递归地两两合并，类似归并排序
