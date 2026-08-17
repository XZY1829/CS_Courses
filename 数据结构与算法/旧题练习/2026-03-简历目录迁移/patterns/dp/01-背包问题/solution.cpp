#include <iostream>
#include <vector>

using namespace std;

int main() {
    int n, m;
    cin >> n >> m;
    vector<int> weight(n + 1);
    vector<int> like(n + 1);
    for (int i = 1; i <= n; i++) {
        cin >> weight[i] >> like[i];
    }
    vector<vector<int>> dp(n + 1, vector<int>(m + 1, -1));
    for (int i = 0; i < n; i++) {
        dp[i][0] = 0;
    }
    for (int j = 0; j <= m; j++) {
        if (weight[1] <= j) {
            dp[1][j] = like[1];
        }
        else {
            dp[1][j] = 0;
        }
    }
    for (int i = 2; i <= n; i++) {
        for (int j = 0; j <= m; j++) {
            dp[i][j] = dp[i - 1][j];
            // 选第 i 个物品（前提是容量足够）
            if (j >= weight[i]) {
                dp[i][j] = max(dp[i][j], dp[i - 1][j - weight[i]] + like[i]);
            }
        }
    }
    cout << dp[n][m] << endl;
    return 0;
}
