// 155. 最小栈
#include <iostream>
#include <stack>
#include <cassert>
using namespace std;
class MinStack {
public:
    MinStack() {
        // TODO: 在此实现
    }

    void push(int val) {
        // TODO: 在此实现
    }

    void pop() {
        // TODO: 在此实现
    }

    int top() {
        // TODO: 在此实现
    }

    int getMin() {
        // TODO: 在此实现
    }

};
int main() {
    MinStack ms;
    ms.push(-2); ms.push(0); ms.push(-3);
    assert(ms.getMin() == -3);
    ms.pop();
    assert(ms.top() == 0);
    assert(ms.getMin() == -2);
    cout << "All tests passed!" << endl;
}
