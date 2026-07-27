// 207. 课程表
// https://leetcode.cn/problems/course-schedule/

#include <iostream>
#include <vector>
#include <queue>
#include <cassert>
using namespace std;

class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        // TODO: 在此实现
    }

};

int main() {
    Solution sol;
    // numCourses=2, [[1,0]] → true
    vector<vector<int>> p1 = {{1,0}};
    assert(sol.canFinish(2, p1) == true);
    // numCourses=2, [[1,0],[0,1]] → false (有环)
    vector<vector<int>> p2 = {{1,0},{0,1}};
    assert(sol.canFinish(2, p2) == false);
    cout << "All tests passed!" << endl;
    return 0;
}
