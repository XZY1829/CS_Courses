# 208. 实现 Trie (前缀树) - 题解

## 思路

Trie 是一棵 26 叉树，每个节点有 26 个子指针（对应 a-z）和一个 `isEnd` 标记。

- **insert**：沿字符路径创建节点，最后标记 `isEnd`
- **search**：沿路径走到末尾，检查 `isEnd`
- **startsWith**：沿路径走到末尾，只要不中途为 null 就返回 true

## 解法

```cpp
class Trie {
    struct TrieNode {
        TrieNode* children[26] = {};
        bool isEnd = false;
    };
    TrieNode* root;
public:
    Trie() { root = new TrieNode(); }
    void insert(string word) { /* 沿路径创建 */ }
    bool search(string word) { auto* n = find(word); return n && n->isEnd; }
    bool startsWith(string prefix) { return find(prefix) != nullptr; }
private:
    TrieNode* find(const string& s) { /* 沿路径查找 */ }
};
```

## 复杂度

- **insert / search / startsWith**：O(L)，L 为字符串长度
- **空间**：O(N × 26)，N 为所有字符总数

## 关键点

1. `search` 和 `startsWith` 的区别仅在于是否检查 `isEnd`
2. 用数组 `children[26]` 比 `unordered_map` 更快
3. Trie 是字符串匹配、自动补全、拼写检查的基础数据结构
