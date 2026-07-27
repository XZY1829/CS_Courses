# 25. K 个一组翻转链表 - 题解

## 思路

分组翻转：每次取 k 个节点翻转，不足 k 个保持不变。

1. 先检查剩余节点数 ≥ k
2. 翻转当前 k 个节点（标准反转链表操作）
3. 将翻转后的组接回主链
4. 移动到下一组

关键是维护 `prevGroupEnd`（上一组的尾节点 / 当前组的前驱）来连接各组。

## 解法

```cpp
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode dummy(0);
        dummy.next = head;
        ListNode* prevGroupEnd = &dummy;
        while (true) {
            ListNode* check = prevGroupEnd;
            for (int i = 0; i < k; i++) { check = check->next; if (!check) return dummy.next; }
            ListNode* groupStart = prevGroupEnd->next;
            ListNode* prev = check->next;
            ListNode* cur = groupStart;
            for (int i = 0; i < k; i++) { ListNode* next = cur->next; cur->next = prev; prev = cur; cur = next; }
            prevGroupEnd->next = prev;
            prevGroupEnd = groupStart;
        }
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(1)

## 关键点

1. 翻转时 `prev` 初始化为**下一组的头节点**（不是 nullptr），这样翻转后自动接上后续
2. 翻转后 `groupStart` 变成了组尾，`prev` 变成了组头
3. 每组翻转就是标准的反转链表，只是限定了次数为 k
