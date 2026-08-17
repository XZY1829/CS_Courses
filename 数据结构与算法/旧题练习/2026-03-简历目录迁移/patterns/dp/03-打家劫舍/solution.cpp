#include <iostream>
#include <string>
#include <vector>
using namespace std;

int rob(vector<int>& dp, const vector<int>& store, const int n, const int i) {
    if (dp[i] != 0) { return dp[i]; }
    if (n == 1) {
        dp[0] == store[0];
        return dp[0];
    }
    if (i == n - 1) {
        dp[n - 1] = store[n - 1];
        return dp[n - 1];
    }
    if (n > 1) {
        if (i == n - 2) {
            dp[n - 2] = max(store[n - 2], rob(dp, store, n, i + 1));
            return dp[n - 2];
        }
        dp[i] = max(store[i] + rob(dp, store, n, i + 2), rob(dp, store, n, i + 1));
        return dp[i];
    }
}

int main() {
    int t, n;
    cin >> t;
    while (t--) {
        cin >> n;
        vector<int> store(n);
        vector dp(n, 0);
        for (int i = 0; i < n; i++) {
            cin >> store[i];
        }
        rob(dp, store, n, 0);
        cout << dp[0] << endl;
    }
    return 0;
}
