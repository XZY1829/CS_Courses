#include <iostream>
#include <string>
using namespace std;

int main() {
    string s;
    cin >> s;

    if (s == "0") {
        cout << 0 << endl;
        return 0;
    }

    // Sliding window:
    // window is [left, right], with no duplicate chars inside.
    int lastPos[256];
    for (int i = 0; i < 256; i++) {
        lastPos[i] = -1;
    }

    int left = 0;
    int longestSub = 0;
    for (int right = 0; right < static_cast<int>(s.size()); right++) {
        unsigned char c = static_cast<unsigned char>(s[right]);
        if (lastPos[c] >= left) {
            left = lastPos[c] + 1;
        }
        lastPos[c] = right;
        longestSub = max(longestSub, right - left + 1);
    }

    cout << longestSub << endl;
    return 0;
}
