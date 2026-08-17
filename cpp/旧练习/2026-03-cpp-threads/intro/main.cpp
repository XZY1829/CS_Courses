#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <vector>
#include <condition_variable>

using namespace std;
vector<int> vec;
mutex g1;
condition_variable cv;
bool ready;

void f()
{
    unique_lock<mutex> lck(g1);
    while (!ready) cv.wait(lck);
    for (int i = 0; i < vec.size(); i++)
    {
        cout << "thread f: " << vec[i] << endl;
    }
}


int main()
{
    ready = false;
    thread t1(f);
    thread t2(f);
    {
        unique_lock<mutex> lck(g1);
        for (int i = 0; i < 10; i++)
        {
            vec.push_back(i);
        }
        ready = true;
        cout << "vector ready" << endl;
    }

    cv.notify_all();
    t1.join();
    t2.join();
    return 0;
}
