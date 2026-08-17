#include <bits/stdc++.h>

using namespace std;

int r, c;
int n;
int cnt = 0;

struct point {
    int x, y;
};

bool outside(const int x, const int y) {
    if (x >= 1 && x <= r && y >= 1 && y <= c) return false;
    return true;
}

inline bool flag(const int x, const int y, const vector<point>& points) {
    for (const auto point : points) {
        if (point.x == x && point.y == y) {
            return true;
        }
    }
    return false;
}

void count(const int start, const int next, const vector<point>& points) {
    if (start == next) { return; }
    const int dx = points[next].x - points[start].x;
    const int dy = points[next].y - points[start].y;
    if (!outside(points[start].x - dx, points[start].y - dy)) return;
    if (outside(points[start].x + cnt * dx, points[start].y + cnt * dy)) return;
    int k = 2;
    int new_x = points[next].x + dx;
    int new_y = points[next].y + dy;
    while (!outside(new_x, new_y) && flag(new_x, new_y, points)) {
        k++;
        new_x += dx;
        new_y += dy;
    }
    if (outside(new_x, new_y) && k > cnt) cnt = k;
}

int main() {
    cin >> r >> c;
    cin >> n;
    vector<point> points;
    while (n--) {
        int x, y;
        cin >> x >> y;
        points.push_back(point{x, y});
    }
    for (int i = 0; i < points.size(); i++) {
        for (int j = i + 1; j < points.size(); j++) {
            count(i, j, points);
        }
    }
    cout << cnt << endl;
    return 0;
}
