// 169. 多数元素
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        // TODO: 在此实现
    }

};
int main() { Solution sol; vector<int> n1={3,2,3}; assert(sol.majorityElement(n1)==3); vector<int> n2={2,2,1,1,1,2,2}; assert(sol.majorityElement(n2)==2); cout << "All tests passed!" << endl; }
