#include <bits/stdc++.h>
using namespace std;

// 字符 → 数字
int val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'z') return c - 'a' + 10;
    return c - 'A' + 36;
}

// 数字 → 字符
char ch(int x) {
    if (x <= 9) return x + '0';
    if (x <= 35) return x - 10 + 'a';
    return x - 36 + 'A';
}

// 比较 a >= b ?
bool ge(const string &a, const string &b) {
    if (a.size() != b.size()) return a.size() > b.size();
    return a >= b;
}

// 减法：a - b（保证 a >= b）
string sub(string a, string b) {
    int n = a.size(), m = b.size();
    reverse(a.begin(), a.end());
    reverse(b.begin(), b.end());

    string res;
    int borrow = 0;

    for (int i = 0; i < n; i++) {
        int x = val(a[i]) - borrow;
        int y = (i < m ? val(b[i]) : 0);

        if (x < y) {
            x += 62;
            borrow = 1;
        } else {
            borrow = 0;
        }

        res += ch(x - y);
    }

    while (res.size() > 1 && res.back() == '0') res.pop_back();
    reverse(res.begin(), res.end());
    return res;
}

int main() {
    string s;
    cin >> s;
    int n = s.size();

    string p = s;

    // Step 1: 镜像
    for (int i = 0; i < n / 2; i++) {
        p[n - 1 - i] = p[i];
    }

    // Step 2: 如果不够大 → 中间进位
    if (!ge(p, s)) {
        int mid = (n - 1) / 2;

        while (mid >= 0) {
            int v = val(p[mid]) + 1;
            if (v < 62) {
                p[mid] = ch(v);
                break;
            }
            p[mid] = ch(0);
            mid--;
        }

        // 全是 Z → 变成 100...001
        if (mid < 0) {
            p = "1" + string(n - 1, '0') + "1";
        } else {
            // 再镜像一次
            for (int i = 0; i < n / 2; i++) {
                p[n - 1 - i] = p[i];
            }
        }
    }

    // Step 3: y = p - s
    cout << sub(p, s) << endl;
}




// #include <iostream>
// #include <string>
// #include <vector>
// #include <algorithm>
// #include <cmath>
// using namespace std;
//
// long long to_ten(string s) {
//     long long res = 0;
//     for (int i = 0; i < s.length(); i++) {
//         if (s[i] >= '0' && s[i] <= '9') {
//             res = res * 62 + s[i] - '0';
//         }
//         if (s[i] >= 'a' && s[i] <= 'z') {
//             res = res * 62 + s[i] - 'a' + 10;
//         }
//         if (s[i] >= 'A' && s[i] <= 'Z') {
//             res = res * 62 + s[i] - 'A' + 36;
//         }
//     }
//     return res;
// }
//
// string toString(long long num) {
//     string res;
//     while (num > 0) {
//         const int digit = num % 62;
//         if (digit <= 9) {
//             res += char(digit + '0');
//         }
//         else if (digit <=35) {
//             res += char(digit - 10) + 'a';
//         }
//         else if (digit <=61) {
//             res += char(digit - 36) + 'A';
//         }
//         num /= 62;
//     }
//     reverse(res.begin(), res.end());
//     return res;
// }
//
// string add(const string& a, const string& b) {
//     long long res = to_ten(a) + to_ten(b);
//     return toString(res);
// };
//
// int main() {
//     string s;
//     cin >> s;
//     int l = 0,r = s.length() - 1;
//     long long res = 0;
//     while (l <= r) {
//         while (s[l] != s[r]) {
//             res += pow(62,l);
//             string t = toString(pow(62,l));
//             s = add(s,t);
//             cout << s << endl;
//         }
//         l++;
//         r--;
//     }
//     cout << toString(res) << endl;
//     return 0;
// }
