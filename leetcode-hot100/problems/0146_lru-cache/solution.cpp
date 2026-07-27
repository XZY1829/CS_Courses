// 146. LRU 缓存
// https://leetcode.cn/problems/lru-cache/

#include <iostream>
#include <unordered_map>
#include <list>
#include <cassert>
using namespace std;

class LRUCache {
public:
    LRUCache(int capacity) {
        // TODO: 在此实现
    }

    int get(int key) {
        // TODO: 在此实现
    }

    void put(int key, int value) {
        // TODO: 在此实现
    }

};

int main() {
    // 示例: ["LRUCache","put","put","get","put","get","put","get","get","get"]
    //       [[2],[1,1],[2,2],[1],[3,3],[2],[4,4],[1],[3],[4]]
    LRUCache cache(2);
    cache.put(1, 1);
    cache.put(2, 2);
    assert(cache.get(1) == 1);
    cache.put(3, 3);       // 淘汰 key=2
    assert(cache.get(2) == -1);
    cache.put(4, 4);       // 淘汰 key=1（不是3，因为get(1)让1变成最近使用）
    // 实际上 put(3,3) 后缓存是 [3,1]，get(2)=-1
    // put(4,4) 淘汰最久未使用的 key=1? 不对，让我重新看
    // put(1,1) → [1]
    // put(2,2) → [2,1]
    // get(1) → 1, [1,2]
    // put(3,3) → 淘汰2, [3,1]
    // get(2) → -1
    // put(4,4) → 淘汰1, [4,3]
    assert(cache.get(1) == -1);
    assert(cache.get(3) == 3);
    assert(cache.get(4) == 4);

    cout << "All tests passed!" << endl;
    return 0;
}
