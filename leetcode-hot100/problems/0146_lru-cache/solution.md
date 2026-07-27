# 146. LRU 缓存 - 题解

## 思路

LRU 缓存需要两个操作都是 O(1)：
- `get(key)`：快速查找 → 哈希表
- `put(key, value)`：维护使用顺序、淘汰最久未使用 → 双向链表

**哈希表 + 双向链表**：
- 链表头部 = 最近使用，尾部 = 最久未使用
- 哈希表存 `key → 链表迭代器`，实现 O(1) 定位
- 访问/更新时，将节点移到头部（`splice`）
- 容量满时，删除尾部节点

## 解法

```cpp
class LRUCache {
    int cap;
    list<pair<int,int>> dll;
    unordered_map<int, list<pair<int,int>>::iterator> map;
public:
    LRUCache(int capacity) : cap(capacity) {}
    int get(int key) {
        if (!map.count(key)) return -1;
        dll.splice(dll.begin(), dll, map[key]);
        return map[key]->second;
    }
    void put(int key, int value) {
        if (map.count(key)) {
            map[key]->second = value;
            dll.splice(dll.begin(), dll, map[key]);
        } else {
            if ((int)dll.size() == cap) {
                map.erase(dll.back().first);
                dll.pop_back();
            }
            dll.emplace_front(key, value);
            map[key] = dll.begin();
        }
    }
};
```

## 复杂度

- `get`：O(1)
- `put`：O(1)
- **空间**：O(capacity)

## 关键点

1. **`list::splice`** 是 O(1) 的节点移动操作，是用 STL 实现 LRU 的关键
2. 淘汰时要记得同时从哈希表中删除 key
3. 面试中可能要求手写双向链表，而非用 `std::list`
4. 相关变体：LFU 缓存（按频率淘汰，更复杂）
