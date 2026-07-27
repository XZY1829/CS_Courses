// 206. 反转链表
// https://leetcode.cn/problems/reverse-linked-list/

#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

ListNode* makeList(const vector<int>& v) {
    ListNode dummy(0);
    ListNode* cur = &dummy;
    for (int x : v) { cur->next = new ListNode(x); cur = cur->next; }
    return dummy.next;
}
vector<int> toVec(ListNode* head) {
    vector<int> r; for (; head; head = head->next) r.push_back(head->val); return r;
}

class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [1,2,3,4,5] → [5,4,3,2,1]
    auto r1 = sol.reverseList(makeList({1,2,3,4,5}));
    assert((toVec(r1) == vector<int>{5,4,3,2,1}));

    // 示例 2: [1,2] → [2,1]
    auto r2 = sol.reverseList(makeList({1,2}));
    assert((toVec(r2) == vector<int>{2,1}));

    // 示例 3: [] → []
    auto r3 = sol.reverseList(nullptr);
    assert(r3 == nullptr);

    cout << "All tests passed!" << endl;
    return 0;
}
