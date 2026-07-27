# 141. 环形链表 - 题解

## 思路

**Floyd 快慢指针**（龟兔赛跑）：

- `slow` 每次走 1 步，`fast` 每次走 2 步
- 如果有环，fast 一定会追上 slow（在环内相遇）
- 如果无环，fast 会先到达 null

## 解法

```cpp
class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode *slow = head, *fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) return true;
        }
        return false;
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(1)

## 关键点

1. `while (fast && fast->next)` 保证 fast 走两步不会空指针
2. fast 的速度是 slow 的 2 倍，所以在环内每一轮 fast 追近 slow 一步，一定会相遇
3. 进阶版 142 题要找环的入口节点，需要在相遇后再做一步操作
