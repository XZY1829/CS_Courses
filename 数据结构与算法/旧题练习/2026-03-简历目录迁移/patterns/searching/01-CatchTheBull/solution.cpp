#include <bits/stdc++.h>

using namespace std;

int main() {
    int n, k;
    cin >> n >> k;
    queue<int> q;
    vector<int> vis(max(n, k) + 1);
    vis[n] = 1;
    q.push(n);
    int ans = 0;
    while (!q.empty()) {
        int sz = q.size();
        while (sz--) {
            const int x = q.front();
            q.pop();
            vis[x] = 1;
            if (x == k) {
                cout << ans << endl;
                return 0;
            }
            if (x - 1 >= 0 && vis[x - 1] != 1) {
                q.push(x - 1);
            }
            if (x + 1 <= max(n, k) && vis[x + 1] != 1) {
                q.push(x + 1);
            }
            if (2 * x >= 0 && 2 * x <= max(n, k) && vis[2 * x] != 1) {
                q.push(2 * x);
            }
        }
        ans++;
    }
    return 0;
}
