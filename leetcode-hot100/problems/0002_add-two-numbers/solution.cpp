// 2. 两数相加
// https://leetcode.cn/problems/add-two-numbers/

// 双指针同步遍历 + 进位传递。

#include <cassert>
#include <iostream>
#include <vector>

using namespace std;

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};
ListNode* makeList(const vector<int>& v) {
    ListNode dummy(0);
    ListNode* cur = &dummy;
    for (int x : v) {
        cur->next = new ListNode(x);
        cur = cur->next;
    }
    return dummy.next;
}
vector<int> toVec(ListNode* head) {
    vector<int> r;
    for (; head; head = head->next)
        r.push_back(head->val);
    return r;
}

class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        // TODO: 在此实现
        ListNode* dummy = new ListNode(0); // 哑节点，方便处理头节点。
        ListNode* cur = dummy;
        int carry = 0;
        while (l1 || l2 || carry) {
            int sum = carry;
            if (l1) {
                sum += l1->val;
                l1 = l1->next;
            }
            if (l2) {
                sum += l2->val;
                l2 = l2->next;
            }
            carry = sum / 10;
            cur->next = new ListNode(sum % 10);
            cur = cur->next;
        }
        return dummy->next;
    }
};

int main() {
    Solution sol;

    // 示例 1: [2,4,3] + [5,6,4] → [7,0,8] (342+465=807)
    auto r1 = sol.addTwoNumbers(makeList({2, 4, 3}), makeList({5, 6, 4}));
    assert((toVec(r1) == vector<int>{7, 0, 8}));

    // 示例 2: [0] + [0] → [0]
    auto r2 = sol.addTwoNumbers(makeList({0}), makeList({0}));
    assert((toVec(r2) == vector<int>{0}));

    // 示例 3: [9,9,9,9,9,9,9] + [9,9,9,9] → [8,9,9,9,0,0,0,1]
    auto r3 = sol.addTwoNumbers(makeList({9, 9, 9, 9, 9, 9, 9}), makeList({9, 9, 9, 9}));
    assert((toVec(r3) == vector<int>{8, 9, 9, 9, 0, 0, 0, 1}));

    cout << "All tests passed!" << endl;
    return 0;
}
