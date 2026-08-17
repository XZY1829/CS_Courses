#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

using namespace std;

mutex mtx;
queue<int> Q;
condition_variable cv_empty, cv_full;

//con
void consume()
{
    srand(time(0));
    unique_lock<mutex> lck(mtx, defer_lock);
    while (true)
    {
        lck.lock();
        while (Q.empty())
        {
            cv_empty.wait(lck);
        }
        cout << this_thread::get_id();
        cout << "consume " << Q.front() << endl;
        Q.pop();
        if (Q.size() <=10)
        {
            cv_full.notify_all();
        }
        lck.unlock();

        int t = rand() % 1000;
        this_thread::sleep_for(chrono::milliseconds(t));
    }
}

//pro
void produce()
{
    srand(time(0));
    unique_lock<mutex> lck(mtx, defer_lock);
    for (int i = 0; i < 60; i++)
    {
        lck.lock();
        while (Q.size() > 20)
        {
            cv_full.wait(lck);
        }
        cout << this_thread::get_id();
        cout << "produce " << i << endl;
        Q.push(i);
        cv_empty.notify_all();
        lck.unlock();

        int t = rand() % 1000;
        this_thread::sleep_for(chrono::milliseconds(t));
    }
}

int main()
{
    thread consumer[3], producer[3];
    for (int i = 0; i < 3; i++)
    {
        consumer[i] = thread(produce);
    }
    for (int i = 0; i < 3; i++)
    {
        producer[i] = thread(consume);
    }
    for (int i = 0; i < 3; i++)
    {
        consumer[i].join();
    }
    for (int i = 0; i < 3; i++)
    {
        producer[i].join();
    }
        return 0;
}
