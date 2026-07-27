# 21. 合并两个有序链表 - 题解

## 思路

经典的**双指针归并**。用一个 dummy 头节点简化边界处理。

逐个比较两个链表的当前节点，较小的接到结果链表尾部。最后把剩余部分直接接上。

## 解法

```cpp
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode dummy(0);
        ListNode* tail = &dummy;
        while (list1 && list2) {
            if (list1->val <= list2->val) { tail->next = list1; list1 = list1->next; }
            else { tail->next = list2; list2 = list2->next; }
            tail = tail->next;
        }
        tail->next = list1 ? list1 : list2;
        return dummy.next;
    }
};
```

## 复杂度

- **时间**：O(m + n)
- **空间**：O(1)

## 关键点

1. **dummy 头节点**避免了处理"结果链表为空时的第一个节点"的特殊情况
2. 剩余部分直接接上（不需要逐个复制），因为链表本身有序
3. 这是归并排序的核心子操作，也是第 23 题（合并 K 个有序链表）的基础
