#include <iostream>
#include <vector>
#include <queue>
using namespace std;

// TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
class Person
{
public:
    int p, q;
    int in_queue = 0, have_ys = 0, queued = 0;

    Person(int p, int q)
    {
        this->p = p;
        this->q = q;
    }

    Person() : p(0), q(0)
    {
    }
};


int main()
{
    //input
    int n = 0;
    cin >> n;
    int p, q;
    int maxlen = 0;
    vector<Person> People(n + 1);
    for (int i = 1; i <= n; i++)
    {
        cin >> p;
        cin >> q;
        People[i].p = p;
        People[i].q = q;
        maxlen = max(maxlen, People[i].q);
    }
    const int m = n + 1;
    vector<int> ans(m, 0);
    ans[1] = -1;

    //init
    People[1].in_queue = 1;
    People[1].have_ys = 1;
    People[1].queued = 1;

    queue<Person*> Queue;
    Queue.push(&People[1]);

    //walk
    int hold = 1;
    int last_index = 1;
    int front_index = 1;
    int ys = People[1].p;
    for (int i = People[1].p; i <= maxlen; i++)
    {
        cout << "now at " << i << endl;
        if (!Queue.empty() && Queue.front()->have_ys == 1)
        {
            ans[front_index]++;
        }
        if (Queue.empty())
        {

        }
        if (!Queue.empty() && Queue.front()->have_ys == 0 && i == ys)
        {
            Queue.front()->have_ys = 1;
        }

        //join
        if (i == People[last_index + 1].p && People[last_index + 1].queued == 0)
        {
            cout << "join" << last_index + 1 << " at " << People[last_index + 1].p << endl;
            Queue.push(&People[last_index + 1]);
            People[last_index + 1].in_queue = 1;
            People[last_index + 1].queued = 1;
            last_index ++;
        }
        //leave
        for (int j = 1; j <= n; j++)
        {
            if (i == People[j].q && People[j].in_queue == 1)
            {
                People[j].in_queue = 0;
                if (People[j].have_ys == 1)
                {
                    Queue.pop();
                    cout << "leave" << front_index << " at " << People[j].q << endl;
                    if (!Queue.empty())
                    {
                        Queue.front()->have_ys = 1;

                        for (int k = 1; k <= n; k++)
                        {
                            if (People[k].p == Queue.front()->p)
                            {
                                front_index = k;
                            }
                        }
                    }
                    else
                    {
                        int newpeople = 1;
                        int min_dist = maxlen + 1;
                        for (int j = 1; j <= ys; j++)
                        {
                            if (People[j].queued == 0)
                            {
                                if (min_dist > People[j].p - ys)
                                {
                                    min_dist = People[j].p - ys;
                                    newpeople = j;
                                }
                            }
                        }
                        if (People[newpeople].queued == 0)
                        {
                            People[newpeople].in_queue = 1;
                            People[newpeople].queued = 1;
                            People[newpeople].have_ys = 1;
                            Queue.push(&People[newpeople]);
                            front_index = newpeople;
                            last_index = newpeople;
                            cout << "empty! join" << newpeople <<"at" << i<< endl;
                        }
                        break;
                    }
                }
            }
        }
        if (!Queue.empty() && Queue.front()->have_ys == 1)
        {
            ys++;
        }
    }
    for (int j = 1; j <= n; j++)
    {
        cout << ans[j] << " ";
    }
    return 0;
}
