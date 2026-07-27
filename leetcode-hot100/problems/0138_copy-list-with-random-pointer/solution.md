# 138. 随机链表的复制 - 题解

## 思路

难点在于 `random` 指针可以指向任意节点，复制时需要知道新旧节点的映射关系。

**哈希表法**（最直观）：
1. 第一遍遍历：为每个旧节点创建对应的新节点，存入 `oldToNew` 映射
2. 第二遍遍历：设置每个新节点的 `next` 和 `random` 指针

## 解法

```cpp
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if (!head) return nullptr;
        unordered_map<Node*, Node*> oldToNew;
        Node* cur = head;
        while (cur) { oldToNew[cur] = new Node(cur->val); cur = cur->next; }
        cur = head;
        while (cur) {
            oldToNew[cur]->next = oldToNew[cur->next];
            oldToNew[cur]->random = oldToNew[cur->random];
            cur = cur->next;
        }
        return oldToNew[head];
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(n)

## 其他解法

**原地交织法**（O(1) 空间）：在每个原节点后插入复制节点 → 设置 random → 拆分两个链表。代码更复杂但空间更优。

## 关键点

1. `oldToNew[nullptr]` 在 `unordered_map` 中默认为 `nullptr`，无需特殊处理
2. 必须先创建所有节点再设置指针——random 可能指向后面的节点
