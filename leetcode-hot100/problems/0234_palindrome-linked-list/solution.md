# 234. 回文链表 - 题解

## 思路

O(1) 空间的做法：**快慢指针找中点 → 反转后半 → 逐一比较**。

1. 快慢指针：`fast` 每次走 2 步，`slow` 走 1 步。`fast` 到尾时 `slow` 在中点
2. 从 `slow` 开始反转后半链表
3. 从头和反转后的后半逐个比较值

## 解法

```cpp
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        ListNode *slow = head, *fast = head;
        while (fast && fast->next) { slow = slow->next; fast = fast->next->next; }
        ListNode* prev = nullptr;
        while (slow) { ListNode* next = slow->next; slow->next = prev; prev = slow; slow = next; }
        ListNode *left = head, *right = prev;
        while (right) {
            if (left->val != right->val) return false;
            left = left->next; right = right->next;
        }
        return true;
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(1)

## 关键点

1. 奇偶长度都适用：奇数长度时中间节点归入后半部分，反转后比较时 right 先到 null
2. 严格来说应该在比较后恢复链表（面试可能会追问），再反转一次即可
3. 综合了"快慢指针找中点"和"反转链表"两个基础操作
