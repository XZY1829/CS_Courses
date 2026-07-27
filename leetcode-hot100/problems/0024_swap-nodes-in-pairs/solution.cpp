// 24. 两两交换链表中的节点
// https://leetcode.cn/problems/swap-nodes-in-pairs/

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
    ListNode* swapPairs(ListNode* head) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [1,2,3,4] → [2,1,4,3]
    auto r1 = sol.swapPairs(makeList({1,2,3,4}));
    assert((toVec(r1) == vector<int>{2,1,4,3}));

    // 示例 2: [] → []
    assert(sol.swapPairs(nullptr) == nullptr);

    // 示例 3: [1] → [1]
    auto r3 = sol.swapPairs(makeList({1}));
    assert((toVec(r3) == vector<int>{1}));

    cout << "All tests passed!" << endl;
    return 0;
}
