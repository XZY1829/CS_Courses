# 2. 两数相加 - 题解

## 思路

模拟竖式加法。链表本身就是逆序存储（低位在前），正好符合从低位到高位相加的顺序。

同时遍历两个链表，逐位相加，处理进位。当两个链表都遍历完且无进位时结束。

## 解法

```cpp
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode dummy(0);
        ListNode* cur = &dummy;
        int carry = 0;
        while (l1 || l2 || carry) {
            int sum = carry;
            if (l1) { sum += l1->val; l1 = l1->next; }
            if (l2) { sum += l2->val; l2 = l2->next; }
            carry = sum / 10;
            cur->next = new ListNode(sum % 10);
            cur = cur->next;
        }
        return dummy.next;
    }
};
```

## 复杂度

- **时间**：O(max(m, n))
- **空间**：O(max(m, n))（新链表）

## 关键点

1. 循环条件 `l1 || l2 || carry` 统一处理了三种继续条件，代码简洁
2. 不需要预先对齐两个链表——短的链表遍历完后当 0 处理
3. 最后可能有最高位进位（如 `99 + 1 = 100`），`carry` 条件保证不遗漏
