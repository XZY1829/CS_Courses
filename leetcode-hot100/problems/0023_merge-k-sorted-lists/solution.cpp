// 23. 合并 K 个升序链表
// https://leetcode.cn/problems/merge-k-sorted-lists/

#include <iostream>
#include <vector>
#include <queue>
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
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [[1,4,5],[1,3,4],[2,6]] → [1,1,2,3,4,4,5,6]
    vector<ListNode*> lists1 = {makeList({1,4,5}), makeList({1,3,4}), makeList({2,6})};
    auto r1 = sol.mergeKLists(lists1);
    assert((toVec(r1) == vector<int>{1,1,2,3,4,4,5,6}));

    // 示例 2: [] → []
    vector<ListNode*> lists2 = {};
    auto r2 = sol.mergeKLists(lists2);
    assert(r2 == nullptr);

    // 示例 3: [[]] → []
    vector<ListNode*> lists3 = {nullptr};
    auto r3 = sol.mergeKLists(lists3);
    assert(r3 == nullptr);

    cout << "All tests passed!" << endl;
    return 0;
}
