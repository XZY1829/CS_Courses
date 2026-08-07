---
source_pdf: 05-算法复习-搜索.pdf
part: 1
keywords: BFS, breadth first search, shortest path, queue
---

# BFS广度优先搜索（★★★）

#algorithm #search #bfs #state-compression #flood-fill #concept

## 概览表（一目了然）

| | BFS | DFS |
|---|---|---|
| **数据结构** | 队列（FIFO） | 栈 / 递归 |
| **最优性** | **保证最短路**（边权均为 1） | 不保证 |
| **空间** | O(宽度) — 需存储整层节点 | O(深度) — 只存当前路径 |
| **扩展顺序** | 按层次，距离 1 → 2 → 3 → ... | 沿一条路径深入到底 |
| **适用** | 最短路 / 层次遍历 | 连通性 / 全部方案枚举 |

---

## BFS 核心思想

- **按层次扩展**：先扩展距离为 1 的节点，再距离为 2，依此类推
- **第一次到达目标 = 最短路**（无权图或边权均为 1）
- 用队列保证"先进先出"的层次顺序

```
层次 0:  S
层次 1:  ● ● ●         ← 一步可达的所有节点
层次 2:  ● ● ● ● ●     ← 两步可达的所有节点
层次 3:  ● ● T          ← 第一次到达 T，距离一定是 3
```

> [!important] BFS 最优性前提
> 仅当所有边权相等（通常为 1）时，BFS 保证最短步数。带权图需 Dijkstra 等算法。

---

## DFS vs BFS — 同一问题的两种策略

课件以**抓住那头牛**为例直观对比：

> 农夫在 N=5，牛在 K=17，每步可 X−1、X+1 或 2X。

| | DFS 策略 | BFS 策略 |
|---|---|---|
| **思路** | 随机选一个方向一路走到底，走不通再回溯 | 给节点分层，第 1 层 = 一步可达，第 2 层 = 两步可达… |
| **扩展** | `5→10→20→...` 可能走很远才发现不对 | `5→(4,6,10)→(3,5,7,9,11,12,20)→...→17` |
| **空间** | O(深度)，小 | O(宽度)，大 |
| **最优** | 不保证，方向随机 | **保证**，第一次遇到目标即最短 |
| **关键** | 需要剪枝节省时间 | 需要较大空间存储队列 |

```
BFS 分层搜索过程：
5 → (4, 6, 10) → (3, 7, 9, 11, 12, 20) → (8, 18, 21, 40) → 17 ✓ (4步)
     ↑ 第1层         ↑ 第2层                  ↑ 第3层             ↑ 第4层
```

> [!tip] 选择原则
> 求**最短/最少** → BFS；求**所有方案/连通性/可行性** → DFS。

---

## BFS 模板代码

```cpp
queue<Node> que;
vis[start] = true;         // 入队时立即标记
que.push({start, 0});

while (!que.empty()) {
    Node cur = que.front(); que.pop();
    if (cur == goal) return cur.dist;
    for (each neighbor of cur) {
        if (!vis[neighbor]) {
            vis[neighbor] = true;   // 入队时标记，不是出队时！
            neighbor.dist = cur.dist + 1;
            que.push(neighbor);
        }
    }
}
```

**要点**：

| 要点 | 说明 |
|------|------|
| 入队即标记 | 避免同一节点被多次入队，浪费空间和时间 |
| 记录距离 | `dist = parent.dist + 1`，或在入队时传递层数 |
| 路径还原 | 需要路径时记录 `parent[]`，从终点回溯到起点再反转 |

---

## 抓住那头牛（一维 BFS）

**问题**：农夫在 N，牛在 K，每步可 X−1、X+1 或 2X，求最少步数。

**建模**：一维坐标 BFS，三种转移：

```cpp
int next[] = { x - 1, x + 1, 2 * x };
```

**注意**：坐标可能为负或超过上界，需设置合理 visited 范围（如 `[0, 200001]`）。

```cpp
int bfs(int n, int k) {
    queue<pair<int,int>> que;
    bool vis[200001] = {};
    vis[n] = true;
    que.push({n, 0});
    while (!que.empty()) {
        auto [x, d] = que.front(); que.pop();
        if (x == k) return d;
        for (int nx : {x-1, x+1, 2*x}) {
            if (nx >= 0 && nx <= 200000 && !vis[nx]) {
                vis[nx] = true;
                que.push({nx, d+1});
            }
        }
    }
    return -1;
}
```

---

## 单词序列（BFS 层次扩展）

**问题**：从 beginWord 到 endWord，每次只改一个字母，每步必须在字典中，求最短变换序列长度。

**建模**：两词"相邻" = 恰好差一个字母。BFS 逐层扩展。

### 关键函数

```cpp
int CompareStr(string s1, string s2) {
    int cnt = 0;
    for (int i = 0; i < s1.length(); i++)
        if (s1[i] != s2[i]) cnt++;
    return cnt;   // 返回不同字符个数
}
```

### BFS 主体

```cpp
que.push(Node(start, 1));    // 起始词，序列长度 1
while (!que.empty()) {
    Node u = que.front(); que.pop();
    if (CompareStr(u.str, target) == 1)  // 与终点只差 1 个字母
        return u.len + 1;
    for (int i = 0; i < num_str; i++) {
        if (!vis[i] && CompareStr(u.str, w[i]) == 1) {
            vis[i] = true;
            que.push(Node(w[i], u.len + 1));
        }
    }
}
```

**第一次到达 endWord 的层数即为答案**。

---

## 迷宫问题（BFS + parent 回溯）

**问题**：5×5 网格迷宫，1 为墙 0 为通路，求左上角到右下角的最短路径并**输出完整路径**。

### 队列元素设计

课件中强调：不仅存坐标，还要记录父节点在队列中的下标，用于回溯路径。

```cpp
struct Node {
    int r, c;
    int f;     // 父节点在队列数组中的下标
};
```

### 路径还原

```
parent 链回溯：
end(4,4) ← ... ← node ← ... ← start(0,0)
反转后输出：(0,0) → ... → (4,4)
```

**实现要点**：
- 四方向扩展 + 边界检查 + 墙判断
- 用数组（而非 STL queue）存队列元素，便于通过下标 `f` 回溯
- BFS 保证第一次到达终点为最短步数
- 到达终点后沿 `f` 指针回溯到起点，反转得路径

---

## 拯救公主（BFS + 状态压缩）

**问题**：网格中有禁区、传送门（`$`）、宝石（`0`~`4`），需集齐 K 种宝石才能解除结界抵达终点。

### 状态设计

```cpp
struct Path {
    int x, y, tot;            // 坐标、步数
    bool jewel[5];            // 每种宝石是否已收集
    bool enchantment;         // 结界是否仍存在
};
```

### 三维判重

```cpp
bool used[200][200][32];
// used[x][y][gem_state] — gem_state 用二进制表示已收集的宝石种类
```

**为什么不能用二维 `visited[x][y]`？**

同一格子 `(x, y)` 在不同宝石组合下是不同状态。若只用二维 visited，会错误剪掉仍需探索的路径。例如"持有宝石 {0}" 与"持有宝石 {0,1}" 经过同一格子时是完全不同的搜索状态。

### BFS 搜索策略（课件代码）

```cpp
while (!que.empty()) {
    for (int i = 0; i < 4; i++) {
        start = que.front();
        start.tot++; start.x += dir[i][0]; start.y += dir[i][1];

        // 1) 边界和禁区检查
        if (越界 || maze[start.x][start.y] == '#') continue;

        // 2) 宝石检查
        if ('0' <= maze[start.x][start.y] && maze[start.x][start.y] <= '4')
            start.jewel[maze[start.x][start.y] - '0'] = true;

        // 3) 计算宝石状态的二进制表示
        int get_jewel = 0, ss = 0;
        for (int i = 0; i < 5; i++)
            if (start.jewel[i])
                get_jewel += pow(2, i), ss++;

        // 4) 三维判重
        if (used[start.x][start.y][get_jewel]) continue;
        used[start.x][start.y][get_jewel] = true;

        // 5) 检查结界
        if (ss >= needkind) start.enchantment = false;

        // 6) 传送门处理：入队所有其他传送门位置
        if (maze[start.x][start.y] == '$') {
            for (int i = 0; i < door_num; i++)
                if (door_list[i] != 当前传送门)
                    que.push(从当前状态传送过去);
        }

        // 7) 终点检查：结界已解除才算到达
        if (maze[start.x][start.y] == 'E' && !start.enchantment) {
            printf("%d\n", start.tot);
            finish = true;
        }
        que.push(start);
    }
    que.pop();
}
```

> [!tip] 状态压缩 + BFS
> 当位置不足以区分状态时，增加 bitmask 维度。状态数 = 网格大小 × 2^宝石种类数（最多 200×200×32 = 1,280,000）。

---

## 寻找 NEMO（BFS + 最少穿门数）

**问题**：网格迷宫中有墙（不可穿）和门（可穿但有代价），求从起点到 Nemo 位置最少穿过多少道门。

### 坐标转化（难点）

输入为线段形式的墙/门 `(x, y, d, t)`，需转化为格子间的关系：

```
wall[x][y][4]  — 格子 (x,y) 四个方向上的状态
  0 = 无障碍（空白）
  1 = 墙（不可穿）
  2 = 门（可穿，代价+1）
```

墙 `(x, y, d=1, t=3)` 表示左下角 `(1,1)`，与 Y 轴平行，长度 3，位于 `(1,1)` 到 `(1,4)`。对于格子来说，对应 `(0, 1-3)` 和 `(1, 1-3)` 之间的边界：

```cpp
int dir[4][2] = {{0,-1},{-1,0},{0,1},{1,0}};
// 对于一道墙/门 (x, y, d, t):
// d=1 (平行Y轴): 处理 wall[x][y..y+t][1] 和 wall[x-1][y..y+t][3]
// d=0 (平行X轴): 处理 wall[x..x+t][y][0] 和 wall[x..x+t][y-1][2]
```

### `walked` 数组 — 非 bool 的关键设计

```cpp
int walked[200][200];   // 初始化为 -1（未访问）
// walked[i][j] = 从 Nemo 位置到达 (i,j) 最少穿过几道门
```

**为什么用 int 而非 bool？**

因为同一格子可能通过不同路径到达，穿门数不同。只有新穿门数 < 已记录值时才更新入队，类似 **Dijkstra 的松弛思想**。

### BFS 搜索

```cpp
while (!Path.empty()) {
    for (int i = 0; i < 4; i++) {
        start = Path.front();
        Maze next = start;
        next.x += dir[i][0]; next.y += dir[i][1];

        if (walls[start.x][start.y].wall[i] == 2)
            next.tot++;   // 穿过门，代价+1

        if (越界
            || (walked[next.x][next.y] <= next.tot && walked[next.x][next.y] != -1)
            || walls[start.x][start.y].wall[i] == 1)  // 是墙，不可穿
            continue;

        walked[next.x][next.y] = next.tot;
        Path.push(next);
    }
    Path.pop();
}
```

**注意**：到达 `(0,0)` 后**不能立即退出搜索**，要继续搜到队列清空，保证 `walked` 每个元素都是全局最小值。

### 搜索边界优化

遍历 200×200 格子可能 TLE。优化：在输入时找到迷宫中墙/门出现的最远位置，**加 1 留一条边缘通路**作为搜索边界。若 Nemo 位置更远，则取 Nemo 坐标 +1。

### DFS 替代方案

课件指出本题也可用 DFS。用全局变量 `ans` 存最少穿门数，每次到达 `(0,0)` 更新最小值：

```cpp
void dfs(int x, int y, int tot) {
    if (x == (int)Nemo_x && y == (int)Nemo_y) {
        ans = min(ans, tot);
        return;
    }
    if (walked[x][y] <= tot && walked[x][y] != -1) return;
    walked[x][y] = tot;
    for (int i = 0; i < 4; i++) {
        int nx = x + dir[i][0], ny = y + dir[i][1];
        if (越界) continue;
        if (walls[x][y].wall[i] == 2) dfs(nx, ny, tot + 1); // 有门
        if (walls[x][y].wall[i] == 0) dfs(nx, ny, tot);     // 无障碍
    }
}
```

---

## 考试/测试常见模式

| 场景/关键词 | 解题思路 |
|-------------|----------|
| "最少步数/最短路径" | BFS（边权为 1） |
| "三种移动/变换操作" | 一维或高维 BFS |
| "改一个字母/单词接龙" | BFS 层次扩展 + 字符串比较 |
| "迷宫最短路+还原路径" | BFS + parent 下标回溯 |
| "收集物品/传送门" | BFS + 状态压缩 `visited[x][y][state]` |
| "穿门数最少/多重代价" | `walked` 存最优值（int）而非 bool，类似 Dijkstra 松弛 |
| "层次遍历/按距离分层" | BFS 天然按层处理 |
| "坐标为线段/格子间关系" | 先做坐标转化：线段 → 格子四方向状态数组 |

---

## 相关笔记

- [[DFS深度优先搜索]]
- [[搜索剪枝技巧]]
- [[搜索练习]]
