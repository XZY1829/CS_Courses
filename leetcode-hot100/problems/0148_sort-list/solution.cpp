// 148. 排序链表
// https://leetcode.cn/problems/sort-list/

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
    ListNode* sortList(ListNode* head) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [4,2,1,3] → [1,2,3,4]
    auto r1 = sol.sortList(makeList({4,2,1,3}));
    assert((toVec(r1) == vector<int>{1,2,3,4}));

    // 示例 2: [-1,5,3,4,0] → [-1,0,3,4,5]
    auto r2 = sol.sortList(makeList({-1,5,3,4,0}));
    assert((toVec(r2) == vector<int>{-1,0,3,4,5}));

    // 空链表
    assert(sol.sortList(nullptr) == nullptr);

    cout << "All tests passed!" << endl;
    return 0;
}
