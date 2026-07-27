// 208. 实现 Trie (前缀树)
// https://leetcode.cn/problems/implement-trie-prefix-tree/

#include <iostream>
#include <string>
#include <cassert>
using namespace std;

class Trie {
public:
    Trie() {
        // TODO: 在此实现
    }

    void insert(string word) {
        // TODO: 在此实现
    }

    bool search(string word) {
        // TODO: 在此实现
    }

    bool startsWith(string prefix) {
        // TODO: 在此实现
    }

};

int main() {
    Trie trie;
    trie.insert("apple");
    assert(trie.search("apple") == true);
    assert(trie.search("app") == false);
    assert(trie.startsWith("app") == true);
    trie.insert("app");
    assert(trie.search("app") == true);
    cout << "All tests passed!" << endl;
    return 0;
}
