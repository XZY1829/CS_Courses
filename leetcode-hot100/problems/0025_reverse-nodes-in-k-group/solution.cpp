// 25. K 个一组翻转链表
// https://leetcode.cn/problems/reverse-nodes-in-k-group/

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
    ListNode* reverseKGroup(ListNode* head, int k) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [1,2,3,4,5], k=2 → [2,1,4,3,5]
    auto r1 = sol.reverseKGroup(makeList({1,2,3,4,5}), 2);
    assert((toVec(r1) == vector<int>{2,1,4,3,5}));

    // 示例 2: [1,2,3,4,5], k=3 → [3,2,1,4,5]
    auto r2 = sol.reverseKGroup(makeList({1,2,3,4,5}), 3);
    assert((toVec(r2) == vector<int>{3,2,1,4,5}));

    // k=1 不变
    auto r3 = sol.reverseKGroup(makeList({1,2,3}), 1);
    assert((toVec(r3) == vector<int>{1,2,3}));

    cout << "All tests passed!" << endl;
    return 0;
}
