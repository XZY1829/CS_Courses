---
source_pdf: 05-算法复习-搜索.pdf
part: 1
keywords: DFS, depth first search, backtracking, connected component
---

# DFS深度优先搜索（★★★）

#algorithm #search #dfs #connected-component #pruning #concept

## 概览表（一目了然）

| 特点 | 说明 |
|------|------|
| **数据结构** | 隐式栈（递归）或显式栈 |
| **搜索策略** | 沿一条路走到底，走不通则回溯 |
| **最优性** | **不保证**最优解，需遍历所有可能 |
| **空间复杂度** | O(深度) — 只需存储当前路径 |
| **时间复杂度** | 最坏 O(V+E)（图）或 O(分支^深度)（搜索树） |
| **适用场景** | 连通块计数、全排列/组合枚举、路径存在性、回溯求方案数 |

---

## DFS 核心思想

- **深度优先**：选定一个方向后不断深入，直到无法继续
- **回溯**：撤销当前选择，尝试其他分支（"此路不通，退一步换条路"）
- **隐式栈**：递归调用天然维护搜索栈
- **不保证最优**：第一次找到的解不一定最优，因此常需遍历所有分支

```
        A            搜索顺序：A → B → D → (回溯) → E → (回溯)
       / \                      → C → F → (回溯) → G
      B   C
     / \   / \
    D   E F   G
```

### DFS 回溯三步曲

| 步骤     | 操作                     | 说明            |
| ------ | ---------------------- | ------------- |
| ① 做选择  | `mark_visited(next)`   | 标记当前节点，避免重复访问 |
| ② 递归   | `dfs(next_state)`      | 进入下一层决策       |
| ③ 撤销选择 | `unmark_visited(next)` | 恢复状态，尝试其他分支   |

> [!tip] 何时选 DFS
> 需要**枚举所有方案**、**统计连通块**、或**深度不大的回溯**时优先 DFS；求最短路应选 [[BFS广度优先搜索|BFS]]。

---

## DFS vs BFS（以"抓住那头牛"为例）

课件用同一道题展示了两种策略的差异：

> 农夫在 N，牛在 K，每步可 X−1、X+1 或 2X，求最少步数。

| 策略 | DFS | BFS |
|------|-----|-----|
| **扩展方式** | 随机选一个方向一路走到底 | 按层次先走 1 步的、再走 2 步的 |
| **空间** | O(深度)，小 | O(宽度)，大 |
| **最优性** | 不保证最短，需剪枝 | **保证最短** |
| **适合本题** | ✗（方向太多，易搜索爆炸） | ✓（求最短步数） |

```
DFS: 5 → 6 → 7 → 8 → 16 → ...  方向随机，可能绕远
BFS: 5 → (4, 6, 10) → (3, 5, 7, 9, 11, 12, 20) → ... → 17  按层推进
```

> [!important] 选择原则
> - **求最短 / 最少** → BFS
> - **求所有方案 / 连通性 / 可行性** → DFS
> - **限制条件多、需大量剪枝** → DFS + 剪枝

---

## DFS 模板代码

```cpp
void dfs(State state) {
    if (is_goal(state)) { update_answer(); return; }
    for (each next_state from state) {
        if (is_valid(next_state)) {
            mark_visited(next_state);
            dfs(next_state);
            unmark_visited(next_state); // 回溯
        }
    }
}
```

**两种 visited 策略**：

| 类型 | mark/unmark | 适用场景 |
|------|-------------|----------|
| 回溯型 | 成对出现 | 排列/组合枚举，每条路径独立 |
| 染色型 | 只 mark 不 unmark | 连通块、flood fill，一次标记永久 |

---

## 连通块问题 — 城堡

**问题**：`m×n` 网格，每个格子用 0~15 的数字编码四面墙，求房间数和最大房间面积。

### 位运算解析墙体

墙的值分别为 1（西）、2（北）、4（东）、8（南），即 2⁰、2¹、2²、2³。将格子值转为二进制，判断对应位是否为 1：

| 方向 | 对应位 | 判断方式 |
|------|--------|----------|
| 西墙 | 2⁰ = 1 | `val % 2 == 1` |
| 北墙 | 2¹ = 2 | `(val / 2) % 2 == 1` |
| 东墙 | 2² = 4 | `(val / 4) % 2 == 1` |
| 南墙 | 2³ = 8 | `(val / 8) % 2 == 1` |

```cpp
// 通用方法：循环除以 2，依次提取每一位
for (int a = val; a > 0; a /= 2)
    wall[i] = a % 2;
```

### DFS 解法

```cpp
int dir[4][2] = {{0,-1},{-1,0},{0,1},{1,0}}; // 西、北、东、南
int max_area = 0;

void dfs(int curr_x, int curr_y) {
    int val = m[curr_x][curr_y];
    for (int i = 0; i < 4; i++) {
        int next_x = curr_x + dir[i][0];
        int next_y = curr_y + dir[i][1];
        if (next_x >= 1 && next_x <= rows && next_y >= 1 && next_y <= cols
            && val % 2 == 0          // 该方向无墙
            && !used[next_x][next_y]) {
            used[next_x][next_y] = true;
            max_area++;
            dfs(next_x, next_y);
        }
        val /= 2;  // 右移一位，检查下一方向
    }
}
```

**关键细节**：`val /= 2` 放在循环体末尾，每轮检查一个方向后右移；方向顺序必须与 `dir` 数组匹配（西→北→东→南 对应 2⁰→2¹→2²→2³）。

### 主流程

```
遍历 m×n 格子：
  若 used[i][j] == false：
    房间数++
    area = 0
    DFS(i, j) 统计当前连通块大小
    更新最大面积
```

---

## 单词接龙（DFS + 回溯）

**问题**：给定 n 个单词和一个起始字母，每个单词最多使用两次，拼接时重合部分合并（不允许包含关系），求最长"龙"的长度。

**样例**：`at, touch, cheat, choose, tact`，起始 `a` → `atoucheatactactouchoose`，长度 23。

### 预处理重合部分

将后一个字符串的第 i 个字符与前一个字符串的最后一个字符对比，找到重合起点后逐一核对：

```cpp
void check(int x, int m) {   // x: 当前末尾单词编号, m: 当前总长度
    ans = max(ans, m);
    for (int y = 1; y <= n; y++) {
        if (k[y] <= 0) continue;       // 已用完两次
        for (int i = 0; i < s[x].length(); i++)
            if (s[x][i] == s[y][0]) {   // 找到可能的重合起点
                int iy = 1; bool flag = true;
                for (int ix = i+1; ix < s[x].length() && iy < s[y].length(); ix++, iy++)
                    if (s[x][ix] != s[y][iy]) { flag = false; break; }
                if (flag && iy < s[y].length()) { // iy < len: 排除包含关系
                    k[y]--;                  // 使用次数减 1
                    check(y, m + s[y].length() - iy);  // 只加不重合部分的长度
                    k[y]++;                  // 回溯
                }
            }
    }
}
```

**注意点**：
- 每个单词用 `k[i]` 记录剩余可用次数（初始为 2）
- 重合部分的检查从前一个词的任意位置开始到末尾，与后一个词的开头逐字符比对
- `iy < s[y].length()` 确保不是完全包含（如 `at` 不能接 `a`）

---

## 碎纸机（DFS 枚举断点）

**问题**：把数字字符串在任意位置切割成若干段，使各段之和 ≤ 目标值且尽量接近，输出最优切割方案。

**特殊规则**：
1. 数等于目标 → 不切割
2. 所有切法之和 > 目标 → 输出 `error`
3. 多种切法得到相同最优和 → 输出 `rejected`

### DFS 设计

```cpp
void dfs(int dq, int sum, int cnt) {
    if (dq >= m) {            // 切割完了
        vis[sum]++;           // 记录该和值出现的次数（检测 rejected）
        if (sum > ans) {
            ans = sum;
            num = cnt;
            for (int i = 0; i < cnt; i++)
                step[i] = now[i];  // 保存最优切割方案
        }
        return;
    }
    int t = 0;
    for (int i = dq; i < m; i++) {
        t = t * 10 + str[i] - '0';  // 逐位构造当前段的数值
        if (sum + t > n)             // 剪枝：已超过目标值
            return;
        now[cnt] = t;                // 记录当前段
        dfs(i + 1, sum + t, cnt + 1);
    }
}
```

**关键设计**：
- `dq` 为当前位置（字符串下标），从 `dq` 开始向后切割
- `t = t * 10 + str[i] - '0'` 逐位构造数值，避免额外的子串提取
- `vis[sum]++` 用于检测多解：最终若 `vis[ans] > 1` 则输出 `rejected`

---

## 生日蛋糕（DFS + 多重剪枝）

**问题**：M 层蛋糕总体积 Nπ，每层半径/高度逐层递减（均为正整数），求最小外表面积 S（不含最底层下底面）。

**公式**：体积 V = πR²H，侧面积 A' = 2πRH，底面积 A = πR²

**关键观察**：上下两个底面的表面积完全取决于底层蛋糕的半径（上层蛋糕只有侧面积参与）。

### DFS 框架

从底层向上枚举每层的半径 r 和高 h，第 i 层最小的 r 和 h 都是 i（必须一层比一层大）。

```cpp
int MaxVforNRH(int n, int r, int h) {
    int v = 0;
    for (int i = 0; i < n; i++)
        v += (r - i) * (r - i) * (h - i);
    return v;
}

void dfs(int curr, int sumS, int sumV, int r, int h) {
    if (curr == 0) {
        if (sumV == N && best > sumS)
            best = sumS;
        return;
    }
    // --- 五种剪枝 ---
    if (sumV + leftMinV[curr] > N) return;             // ①体积下界
    if (sumV + MaxVforNRH(curr, r, h) < N) return;     // ②体积上界
    if (sumS + leftMinS[curr] > best) return;           // ③表面积下界
    if (2 * (N - sumV) / r + sumS >= best) return;      // ④单圆柱估算

    for (int i = r - 1; i >= curr; i--) {
        if (curr == M)
            sumS = i * i;     // 底层：加底面积
        for (int j = h - 1; j >= curr; j--) {
            dfs(curr - 1, sumS + i * j * 2, sumV + i * i * j, i, j);
        }
    }
}
```

### 五种剪枝详解

| # | 名称 | 条件 | 原理 |
|---|------|------|------|
| 1 | 体积下界 | `sumV + leftMinV[curr] > N` | 即使剩余层全取最小体积也超了 |
| 2 | 体积上界 | `sumV + MaxV(...) < N` | 即使剩余层全取最大体积也不够 |
| 3 | 尺寸不可安排 | `i < curr` 或 `j < curr` | 半径/高度为正整数且逐层递减，无法继续 |
| 4 | 表面积下界 | `sumS + leftMinS[curr] > best` | 当前面积加理论最小剩余已劣于最优 |
| 5 | 单圆柱估算 | `2(N-sumV)/r + sumS ≥ best` | 剩余体积全做成一个圆柱（面积最小）仍超 |

**预处理**：`leftMinV[i]` 和 `leftMinS[i]` 分别为第 1~i 层取最小半径/高度时的最小体积和最小面积之和。

> [!important] 剪枝顺序
> 廉价剪枝（体积边界）放前面，昂贵计算（`MaxVforNRH`）放后面，能大幅减少搜索树规模。

---

## 考试/测试常见模式

| 场景/关键词 | 解题思路 |
|-------------|----------|
| "房间/连通区域/岛屿" | DFS 四方向/八方向 flood fill，只 mark 不 unmark |
| "墙用数字编码" | 位运算 `%2`、`/2` 依次提取各方向墙信息 |
| "拼接/接龙/排列" | DFS + 回溯，注意重复使用限制和包含关系排除 |
| "切割/分段/和为定值" | DFS 枚举断点 + sum 剪枝 |
| "求所有方案/计数" | DFS 枚举，mark/unmark 成对回溯 |
| "层数递减/组合优化" | 多重剪枝 DFS，预处理上下界 |
| "分组/等长恢复" | 枚举目标和 → DFS 组合 → 排序+对称性剪枝 |

---

## 相关笔记

- [[BFS广度优先搜索]]
- [[搜索剪枝技巧]]
- [[搜索练习]]
