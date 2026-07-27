// 118. 杨辉三角
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;
class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        // TODO: 在此实现
    }

};
int main() { Solution sol; auto r = sol.generate(5); assert(r.size()==5); assert((r[4]==vector<int>{1,4,6,4,1})); cout << "All tests passed!" << endl; }
