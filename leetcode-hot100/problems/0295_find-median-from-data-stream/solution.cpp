// 295. 数据流的中位数
#include <iostream>
#include <queue>
#include <cassert>
#include <cmath>
using namespace std;
class MedianFinder {
public:
    MedianFinder() {
        // TODO: 在此实现
    }

    void addNum(int num) {
        // TODO: 在此实现
    }

    double findMedian() {
        // TODO: 在此实现
    }

};
int main() {
    MedianFinder mf;
    mf.addNum(1); mf.addNum(2);
    assert(fabs(mf.findMedian() - 1.5) < 1e-9);
    mf.addNum(3);
    assert(fabs(mf.findMedian() - 2.0) < 1e-9);
    cout << "All tests passed!" << endl;
}
