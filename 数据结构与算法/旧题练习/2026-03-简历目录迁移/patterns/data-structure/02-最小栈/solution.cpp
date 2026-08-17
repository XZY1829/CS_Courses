#include <iostream>
#include <deque>
using namespace  std;

class MinStack {
public:
    MinStack() {

    }

    void push(int val) {
        dq.push_back(val);
    }

    void pop() {
        dq.pop_back();
    }

    int top() {
        return dq.back();
    }

    int getMin() {
        int min = INT_MAX;
        for (deque<int>::iterator it = dq.begin(); it != dq.end(); ++it)
        {
            if (*it < min)
            {
                min = *it;
            }
        }
        return min;
    }
    deque<int> dq;
};

int main()
{
    MinStack ms;
    ms.push(-2);
    ms.push(0);
    ms.push(-3);
    cout << ms.getMin() << endl;
    ms.pop();
    cout << ms.top() << endl;
    cout << ms.getMin() << endl;
    return 0;
}