// 21. 合并两个有序链表
// https://leetcode.cn/problems/merge-two-sorted-lists/

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
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [1,2,4] + [1,3,4] → [1,1,2,3,4,4]
    auto r1 = sol.mergeTwoLists(makeList({1,2,4}), makeList({1,3,4}));
    assert((toVec(r1) == vector<int>{1,1,2,3,4,4}));

    // 示例 2: [] + [] → []
    auto r2 = sol.mergeTwoLists(nullptr, nullptr);
    assert(r2 == nullptr);

    // 示例 3: [] + [0] → [0]
    auto r3 = sol.mergeTwoLists(nullptr, makeList({0}));
    assert((toVec(r3) == vector<int>{0}));

    cout << "All tests passed!" << endl;
    return 0;
}
