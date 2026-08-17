#include <iostream>
#include <vector>
using namespace std;

int main() {
    int n;
    cin >> n;
    vector<long long> nums(n);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }

    if (n == 0) {
        cout << 0;
        return 0;
    }

    // Kadane:
    // current = maximum subarray sum ending at i
    // best = maximum subarray sum seen so far
    long long current = nums[0];
    long long best = nums[0];
    for (int i = 1; i < n; i++) {
        current = max(nums[i], current + nums[i]);
        best = max(best, current);
    }

    cout << best;
    return 0;
}