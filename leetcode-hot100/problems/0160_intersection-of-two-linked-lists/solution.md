# 160. 相交链表 - 题解

## 思路

**双指针等距走法**：指针 a 走完链表 A 后走链表 B，指针 b 走完链表 B 后走链表 A。两个指针走过的总长度相同（`lenA + lenB`），如果有交点必然同时到达。

设 A 独有部分长 a，B 独有部分长 b，公共部分长 c：
- 指针 a 走 `a + c + b` 步到达交点
- 指针 b 走 `b + c + a` 步到达交点
- 两者相等！

如果没有交点，两者最终都走到 null。

## 解法

```cpp
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode *a = headA, *b = headB;
        while (a != b) {
            a = a ? a->next : headB;
            b = b ? b->next : headA;
        }
        return a;
    }
};
```

## 复杂度

- **时间**：O(m + n)
- **空间**：O(1)

## 关键点

1. 比较的是**节点指针**（内存地址），不是节点值
2. 切换链表时，`a = a ? a->next : headB`——当 a 到达 null 时切到 B 的头部
3. 无交点时，两个指针最终都变成 null，`null == null` 退出循环
