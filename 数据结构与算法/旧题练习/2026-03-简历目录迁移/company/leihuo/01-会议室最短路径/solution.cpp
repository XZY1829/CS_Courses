#include <bits/stdc++.h>
using namespace std;

/*
    节点编号规则：
    A x M y  →  id = (x-1)*5 + (y-1)

    例如：
    A1M1 → 0
    A1M5 → 4
    A2M1 → 5
    A3M5 → 14
*/

// 将 (楼层, 房间号) 转为 id
int get_id(int floor, int room) {
    return (floor - 1) * 5 + (room - 1);
}

// 将 id 转回字符串 "AxMy"
string get_name(int id) {
    int floor = id / 5 + 1;
    int room = id % 5 + 1;
    return "A" + to_string(floor) + "M" + to_string(room);
}

int main() {
    const int N = 15; // 总共 15 个会议室
    // 邻接表：g[u] = { {v, w}, ... }
    vector<pair<int, int>> g[N];
    // 一、建图
    // 1️⃣ 同一层：左右相邻会议室，权重 = 1
    for (int floor = 1; floor <= 3; floor++) {
        for (int room = 1; room <= 5; room++) {
            int u = get_id(floor, room);
            // 向右
            if (room < 5) {
                int v = get_id(floor, room + 1);
                g[u].push_back({v, 1});
                g[v].push_back({u, 1}); // 双向
            }
        }
    }
    // 2️⃣ 电梯（在 M1 和 M5），上下权重 = 2
    for (int room : {1, 5}) {
        for (int floor = 1; floor <= 2; floor++) {
            int u = get_id(floor, room);
            int v = get_id(floor + 1, room);

            g[u].push_back({v, 2});
            g[v].push_back({u, 2}); // 上下相同
        }
    }
    // 3️⃣ 楼梯（在 M3）
    // 上楼：6，下楼：3（注意是有向边）
    for (int floor = 1; floor <= 2; floor++) {
        int u = get_id(floor, 3);
        int v = get_id(floor + 1, 3);

        g[u].push_back({v, 6}); // 上楼
        g[v].push_back({u, 3}); // 下楼
    }
    // 二、输入起点
    // 输入格式示例：A2M3
    string start_str;
    cin >> start_str;

    int start_floor = start_str[1] - '0';
    int start_room = start_str[3] - '0';

    int start = get_id(start_floor, start_room);

    // 三、Dijkstra
    vector<int> dist(N, INT_MAX); // 最短距离
    vector<int> pre(N, -1); // 路径记录
    vector<bool> vis(N, false); // 是否访问过

    // 小根堆：{距离, 节点}
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;

    dist[start] = 0;
    pq.push({0, start});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        // 已处理过，跳过
        if (vis[u]) continue;
        vis[u] = true;
        // 遍历所有邻居
        for (auto [v, w] : g[u]) {
            if (dist[v] > dist[u] + w) {
                dist[v] = dist[u] + w;
                pre[v] = u; // 记录路径
                pq.push({dist[v], v});
            }
        }
    }
    // 四、输出结果
    for (int i = 0; i < N; i++) {
        cout << "到达 " << get_name(i) << " 的最短步长: " << dist[i] << endl;
        // 回溯路径
        vector<int> path;
        int cur = i;
        while (cur != -1) {
            path.push_back(cur);
            cur = pre[cur];
        }
        reverse(path.begin(), path.end());
        // 输出路径
        cout << "路径: ";
        for (int j = 0; j < path.size(); j++) {
            cout << get_name(path[j]);
            if (j != path.size() - 1) cout << " -> ";
        }
        cout << endl << endl;
    }

    return 0;
}
