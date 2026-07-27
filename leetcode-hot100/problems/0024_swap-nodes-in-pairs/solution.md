# 24. 两两交换链表中的节点 - 题解

## 思路

迭代法：维护 `prev` 指向当前待交换对的前一个节点。

每次交换 `a = prev->next` 和 `b = a->next`：
- `a->next = b->next`
- `b->next = a`
- `prev->next = b`
- `prev = a`（a 变成了后面节点的前驱）

## 解法

```cpp
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        ListNode dummy(0);
        dummy.next = head;
        ListNode* prev = &dummy;
        while (prev->next && prev->next->next) {
            ListNode* a = prev->next;
            ListNode* b = a->next;
            a->next = b->next;
            b->next = a;
            prev->next = b;
            prev = a;
        }
        return dummy.next;
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(1)

## 关键点

1. 是 K 个一组翻转链表（第 25 题）在 K=2 时的特例
2. 奇数长度链表，最后一个节点不动
