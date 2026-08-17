#include <iostream>
#include <vector>
#include <queue>
#include <climits>

using namespace std;

struct Person
{
    int p, q;
};

int main()
{
    int n;
    cin >> n;

    vector<Person> people(n + 1);
    for (int i = 1; i <= n; i++)
        cin >> people[i].p >> people[i].q;

    vector<long long> ans(n + 1, 0);
    vector<bool> job_done(n + 1, false);

    queue<int> Q;

    int pos = people[1].p;
    int next_join = 2;

    Q.push(1);

    while (!Q.empty() || next_join <= n)
    {
        // 队伍为空：最近的人来拿伞
        if (Q.empty())
        {
            pos = people[next_join].p;
            Q.push(next_join);
            next_join++;
            continue;
        }

        int front = Q.front();

        int next_p = (next_join <= n ? people[next_join].p : INT_MAX);
        int next_q = people[front].q;

        int next_event = min(next_p, next_q);

        // 计算贡献
        ans[front] += next_event - pos;
        pos = next_event;

        // join
        if (next_event == next_p)
        {
            Q.push(next_join);
            next_join++;
        }

        // job done
        if (next_event == next_q)
        {
            job_done[front] = true;
        }

        // leave
        while (!Q.empty() && job_done[Q.front()])
        {
            Q.pop();
        }
    }

    for (int i = 1; i <= n; i++)
        cout << ans[i] << " ";

    return 0;
}