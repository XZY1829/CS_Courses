# 19. 删除链表的倒数第 N 个结点 - 题解

## 思路

**快慢指针，一次遍历**：

1. 让 `fast` 先走 `n+1` 步（从 dummy 开始），这样 fast 和 slow 之间相隔 n+1 个节点
2. 然后 fast 和 slow 同时走，直到 fast 到 null
3. 此时 slow 指向待删除节点的**前一个**节点，执行删除

使用 dummy 头节点处理删除头节点的边界情况。

## 解法

```cpp
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode dummy(0);
        dummy.next = head;
        ListNode *fast = &dummy, *slow = &dummy;
        for (int i = 0; i <= n; i++) fast = fast->next;
        while (fast) { fast = fast->next; slow = slow->next; }
        slow->next = slow->next->next;
        return dummy.next;
    }
};
```

## 复杂度

- **时间**：O(L)，L 为链表长度
- **空间**：O(1)

## 关键点

1. dummy 节点解决了"删除头节点"的特殊情况
2. fast 先走 `n+1` 步（不是 n 步），这样 slow 停在待删节点的前驱
