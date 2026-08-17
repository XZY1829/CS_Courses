#include <iostream>
#include <vector>
using namespace std;

int main() {
    int n, m;
    cin >> n >> m;
    vector<int> coins(n);
    for (int i = 0; i < n; i++) {
        cin >> coins[i];
    }
    vector<vector<int>> dp(n + 1, vector<int>(m + 1,INT_MAX / 2));
    for (int i = 0; i <= n; i++) {
        dp[i][0] = 0;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= m; j++) {
            if (coins[i] <= j) {
                dp[i + 1][j] = min(dp[i][j], dp[i + 1][j - coins[i]] + 1);
            }
            else {
                dp[i + 1][j] = dp[i][j];
            }
        }
    }
    cout << (dp[n][m] >= INT_MAX / 2 ? 0 : dp[n][m]) << endl;
    return 0;
}
