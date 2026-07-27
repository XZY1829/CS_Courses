// 19. 删除链表的倒数第 N 个结点
// https://leetcode.cn/problems/remove-nth-node-from-end-of-list/

#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};
ListNode* makeList(const vector<int>& v) {
    ListNode dummy(0); ListNode* cur = &dummy;
    for (int x : v) { cur->next = new ListNode(x); cur = cur->next; }
    return dummy.next;
}
vector<int> toVec(ListNode* head) {
    vector<int> r; for (; head; head = head->next) r.push_back(head->val); return r;
}

class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [1,2,3,4,5], n=2 → [1,2,3,5]
    auto r1 = sol.removeNthFromEnd(makeList({1,2,3,4,5}), 2);
    assert((toVec(r1) == vector<int>{1,2,3,5}));

    // 示例 2: [1], n=1 → []
    auto r2 = sol.removeNthFromEnd(makeList({1}), 1);
    assert(r2 == nullptr);

    // 示例 3: [1,2], n=1 → [1]
    auto r3 = sol.removeNthFromEnd(makeList({1,2}), 1);
    assert((toVec(r3) == vector<int>{1}));

    cout << "All tests passed!" << endl;
    return 0;
}
