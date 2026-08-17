#include <bits/stdc++.h>
using namespace std;

int group, target;
int sum;
bool found;
vector<int> sticks;
vector<int> vis;

void dfs(int cur_len, int start, int done) {
    if (found) return;

    // 拼完所有组
    if (done == group) {
        found = true;
        return;
    }

    // 当前木棒拼完
    if (cur_len == target) {
        dfs(0, 0, done + 1);
        return;
    }

    int prev = -1;

    for (int i = start; i < sticks.size(); i++) {
        if (vis[i]) continue;
        if (sticks[i] == prev) continue;

        if (cur_len + sticks[i] > target) continue;

        vis[i] = 1;

        dfs(cur_len + sticks[i], i + 1, done);

        vis[i] = 0;
        prev = sticks[i];

        // 剪枝
        if (cur_len == 0) return;
        if (cur_len + sticks[i] == target) return;
    }
}

int main() {
    while (true) {
        int n;
        cin >> n;
        if (n == 0) return 0;

        sticks.resize(n);
        vis.assign(n, 0);
        sum = 0;

        for (int i = 0; i < n; i++) {
            cin >> sticks[i];
            sum += sticks[i];
        }

        sort(sticks.begin(), sticks.end(), greater<int>());

        for (int s = sticks[0]; s <= sum; s++) {
            if (sum % s != 0) continue;

            target = s;
            group = sum / s;
            found = false;

            fill(vis.begin(), vis.end(), 0);

            dfs(0, 0, 0);

            if (found) {
                cout << target << endl;
                break;
            }
        }
    }
}