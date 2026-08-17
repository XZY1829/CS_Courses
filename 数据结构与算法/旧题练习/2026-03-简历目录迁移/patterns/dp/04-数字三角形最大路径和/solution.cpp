#include <iostream>
#include <vector>


using namespace std;

vector<vector<int>> dp;
vector<vector<int>> triangle;

int maxSum(const int i, const int j) {
    if (j > i) return 0;
    if (dp[i][j] != -1) return dp[i][j];
    if (i == triangle.size() - 1) {
        return dp[i][j] = triangle[i][j];
    }
    return dp[i][j] = max(maxSum(i + 1, j), maxSum(i + 1, j + 1)) + triangle[i][j];
}

int main() {
    int n;
    cin >> n;
    dp.resize(n);
    for (int i = 0; i < n; i++) {
        triangle.resize(i + 1);
        for (int j = 0; j <= i; j++) {
            int val;
            cin >> val;
            triangle[i].push_back(val);
            dp[i].push_back(-1);
        }
    }
    cout << maxSum(0, 0) << endl;
    return 0;
}
