// 160. 相交链表
// https://leetcode.cn/problems/intersection-of-two-linked-lists/

#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        // TODO: 在此实现
    }

};

int main() {
    // 构造相交链表: A=[4,1,8,4,5], B=[5,6,1,8,4,5], 交点=8
    ListNode *common = new ListNode(8);
    common->next = new ListNode(4);
    common->next->next = new ListNode(5);

    ListNode *headA = new ListNode(4);
    headA->next = new ListNode(1);
    headA->next->next = common;

    ListNode *headB = new ListNode(5);
    headB->next = new ListNode(6);
    headB->next->next = new ListNode(1);
    headB->next->next->next = common;

    Solution sol;
    assert(sol.getIntersectionNode(headA, headB) == common);

    // 无交点
    ListNode *h1 = new ListNode(1);
    ListNode *h2 = new ListNode(2);
    assert(sol.getIntersectionNode(h1, h2) == nullptr);

    cout << "All tests passed!" << endl;
    return 0;
}
