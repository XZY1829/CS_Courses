// 234. 回文链表
// https://leetcode.cn/problems/palindrome-linked-list/

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

class Solution {
public:
    bool isPalindrome(ListNode* head) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;

    // 示例 1: [1,2,2,1] → true
    assert(sol.isPalindrome(makeList({1,2,2,1})) == true);

    // 示例 2: [1,2] → false
    assert(sol.isPalindrome(makeList({1,2})) == false);

    // 奇数长度回文: [1,2,1] → true
    assert(sol.isPalindrome(makeList({1,2,1})) == true);

    // 单节点: [1] → true
    assert(sol.isPalindrome(makeList({1})) == true);

    cout << "All tests passed!" << endl;
    return 0;
}
