// 142. 环形链表 II
// https://leetcode.cn/problems/linked-list-cycle-ii/

#include <iostream>
#include <cassert>
using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [3,2,0,-4], pos=1 → 节点 2
    ListNode *n0 = new ListNode(3), *n1 = new ListNode(2),
             *n2 = new ListNode(0), *n3 = new ListNode(-4);
    n0->next = n1; n1->next = n2; n2->next = n3; n3->next = n1;
    assert(sol.detectCycle(n0) == n1);

    // 示例 2: [1,2], pos=0 → 节点 1
    ListNode *a = new ListNode(1), *b = new ListNode(2);
    a->next = b; b->next = a;
    assert(sol.detectCycle(a) == a);

    // 示例 3: [1], pos=-1 → null
    assert(sol.detectCycle(new ListNode(1)) == nullptr);

    cout << "All tests passed!" << endl;
    return 0;
}
