# 142. 环形链表 II - 题解

## 思路

在 141 的基础上，找到**环的入口节点**。

**数学推导**：设头到入环点距离 a，入环点到相遇点距离 b，环长 c。
- slow 走了 `a + b`
- fast 走了 `a + b + k*c`（多走了 k 圈）
- `2(a+b) = a + b + k*c` → `a = k*c - b = (k-1)*c + (c-b)`

即从头节点走 a 步 = 从相遇点走 `(c-b)` 步（都到达入环点）。所以让一个指针从头出发、一个从相遇点出发，同速前进，再次相遇就是入环点。

## 解法

```cpp
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode *slow = head, *fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) {
                ListNode *p = head;
                while (p != slow) { p = p->next; slow = slow->next; }
                return p;
            }
        }
        return nullptr;
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(1)

## 关键点

1. 理解 `a = (k-1)*c + (c-b)` 是核心——"从头走 a 步"等价于"从相遇点绕若干圈再走 c-b 步"
2. 第二阶段两个指针都走 1 步/轮，不再有快慢之分
