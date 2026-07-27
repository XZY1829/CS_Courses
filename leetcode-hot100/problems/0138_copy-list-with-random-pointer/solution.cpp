// 138. 随机链表的复制
// https://leetcode.cn/problems/copy-list-with-random-pointer/

#include <iostream>
#include <unordered_map>
#include <cassert>
using namespace std;

class Node {
public:
    int val;
    Node* next;
    Node* random;
    Node(int _val) : val(_val), next(nullptr), random(nullptr) {}
};

class Solution {
public:
    Node* copyRandomList(Node* head) {
        // TODO: 在此实现
    }

};

int main() {
    // 构造 [[7,null],[13,0],[11,4],[10,2],[1,0]]
    Node *n0 = new Node(7), *n1 = new Node(13), *n2 = new Node(11),
         *n3 = new Node(10), *n4 = new Node(1);
    n0->next = n1; n1->next = n2; n2->next = n3; n3->next = n4;
    n0->random = nullptr; n1->random = n0; n2->random = n4;
    n3->random = n2; n4->random = n0;

    Solution sol;
    Node* copy = sol.copyRandomList(n0);

    // 验证值和结构正确
    assert(copy->val == 7 && copy->random == nullptr);
    assert(copy->next->val == 13 && copy->next->random == copy);
    assert(copy->next->next->val == 11);
    // 确认是深拷贝
    assert(copy != n0);

    cout << "All tests passed!" << endl;
    return 0;
}
