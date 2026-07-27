// 141. 环形链表
// https://leetcode.cn/problems/linked-list-cycle/

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
    bool hasCycle(ListNode *head) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [3,2,0,-4], pos=1 → true
    ListNode *n0 = new ListNode(3), *n1 = new ListNode(2),
             *n2 = new ListNode(0), *n3 = new ListNode(-4);
    n0->next = n1; n1->next = n2; n2->next = n3; n3->next = n1;
    assert(sol.hasCycle(n0) == true);

    // 示例 3: [1], pos=-1 → false
    ListNode *single = new ListNode(1);
    assert(sol.hasCycle(single) == false);

    // 空链表
    assert(sol.hasCycle(nullptr) == false);

    cout << "All tests passed!" << endl;
    return 0;
}
