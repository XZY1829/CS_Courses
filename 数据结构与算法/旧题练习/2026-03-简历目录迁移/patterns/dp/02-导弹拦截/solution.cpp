#include <iostream>
#include <vector>
using namespace std;

vector<int> height;
vector<int> dp;

int maxNum(int i) {
    if (i == 0) {
        dp[0] = 1;
        return 1;
    }
    if (dp[i] != 0) {
        return dp[i];
    }
    for (int j = 0; j < i; j++) {
        if (height[i] <= height[j]) {
            dp[i] = max(dp[i], maxNum(j) + 1);
        }
    }
    if (dp[i] == 0) {
        dp[i] = 1;
    }
    return dp[i];
}

int main() {
    int n;
    cin >> n;
    height.resize(n);
    dp.resize(n);
    for (int i = 0; i < n; i++) {
        cin >> height[i];
        dp[i] = 0;
    }
    cout << maxNum(n - 1) << endl;
    return 0;
}
