# 206. 反转链表 - 题解

## 思路

**迭代法**：三个指针 `prev, cur, next`，逐个反转指向。

每一步：
1. 保存 `cur->next` 到 `next`
2. 把 `cur->next` 指向 `prev`（反转）
3. `prev` 和 `cur` 各前进一步

## 解法

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* prev = nullptr;
        ListNode* cur = head;
        while (cur) {
            ListNode* next = cur->next;
            cur->next = prev;
            prev = cur;
            cur = next;
        }
        return prev;
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(1)

## 关键点

1. 初始 `prev = nullptr`（反转后的尾节点指向 null）
2. 循环结束时 `cur == nullptr`，`prev` 指向新头节点
3. 递归写法也很简洁但空间 O(n)：`reverseList(head->next)` 然后 `head->next->next = head; head->next = nullptr`
4. 这是链表操作的基础中的基础，大量链表题（如 K 个一组翻转、回文链表）都以此为子操作
