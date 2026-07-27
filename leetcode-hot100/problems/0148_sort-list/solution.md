# 148. 排序链表 - 题解

## 思路

链表的 **归并排序**——链表天然适合归并（不需要随机访问，只需 O(1) 空间合并）。

1. **找中点**：快慢指针，断开为两半
2. **递归排序**左半和右半
3. **合并**两个有序链表（第 21 题）

## 解法

```cpp
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        if (!head || !head->next) return head;
        ListNode *slow = head, *fast = head->next;
        while (fast && fast->next) { slow = slow->next; fast = fast->next->next; }
        ListNode* mid = slow->next;
        slow->next = nullptr;
        return merge(sortList(head), sortList(mid));
    }
private:
    ListNode* merge(ListNode* l1, ListNode* l2) {
        ListNode dummy(0); ListNode* tail = &dummy;
        while (l1 && l2) {
            if (l1->val <= l2->val) { tail->next = l1; l1 = l1->next; }
            else { tail->next = l2; l2 = l2->next; }
            tail = tail->next;
        }
        tail->next = l1 ? l1 : l2;
        return dummy.next;
    }
};
```

## 复杂度

- **时间**：O(n log n)
- **空间**：O(log n)（递归栈深度）

## 关键点

1. 找中点时 `fast = head->next`（不是 head），确保偶数长度时 slow 停在前半的最后一个
2. 数组排序首选快排，链表排序首选归并——链表归并的合并操作是 O(1) 空间
3. 可以用自底向上的迭代归并做到 O(1) 空间，但实现更复杂
