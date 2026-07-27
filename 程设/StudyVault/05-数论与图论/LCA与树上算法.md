---
source_pdf: 06-算法复习-数学很重要.pdf
part: 1
keywords: LCA, lowest common ancestor, Tarjan, sparse table
---

# LCA与树上算法（★★★）

#algorithm #graph-theory #lca #union-find #binary-search #concept

## 概览表：LCA 算法对比

| 算法 | 预处理 | 查询 | 适用 |
|------|--------|------|------|
| **暴力** | O(1) | O(n) | 少量查询 |
| **Tarjan（离线）** | O(n+m) | O(1) | 批量查询 |
| **ST 表（在线）** | O(n log n) | O(1) | 多次查询 |

---

## 二叉搜索树 LCA（简化版）

> 仅适用于 **BST**，一般树上的 LCA 不能靠大小关系判断左右子树。

- 利用 BST 性质：**左 < 根 < 右**
- 若 `p < root < q` → **root 就是 LCA**
- 若 p、q 都在同侧 → **递归该子树**

```cpp
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (p->val < root->val && q->val < root->val)
        return lowestCommonAncestor(root->left, p, q);
    if (p->val > root->val && q->val > root->val)
        return lowestCommonAncestor(root->right, p, q);
    return root;
}
```

---

## Tarjan 算法（DFS + 并查集，离线）

**思想**：DFS 回溯时，用并查集维护"已访问子树"的集合；当查询 `(u, e)` 中 `e` 已被访问，则 `LCA(u,e) = find(e)`。

**伪代码**：

```
Tarjan(u):
  for each child v:
    Tarjan(v)
    merge(u, v)       // union，将 v 子树合并到 u
  mark u as visited
  for each query(u, e):
    if e is visited:
      LCA(u, e) = find(e)
```

**时间复杂度**：O(n + m)，其中 m 为查询数。

**特点**：

- **离线**：必须一次性读入所有查询
- 辅助结构：**并查集**（`merge` + `find`）
- DFS 回溯时，`find(e)` 指向 u 子树中已访问节点的"代表"

---

## ST 表算法（欧拉序 + RMQ，在线）

**步骤 1**：DFS 构造欧拉序，记录 `<序号, 节点, 深度>` 表格

- 每进入/回溯节点都写入欧拉序
- 节点 u 在欧拉序中**第一次与最后一次出现**之间的区间 = u 的子树

**步骤 2**：ST 表预处理区间**最小深度**（深度最小者即为 LCA）

```
f[i][j] = min(f[i][j-1], f[i + 2^(j-1)][j-1])
```

**查询** `[x, y]` 区间最小深度：

```
k = log2(y - x + 1)
ans = min(f[x][k], f[y - (1<<k) + 1][k])
```

**时间复杂度**：预处理 O(n log n)，单次查询 O(1)。

---

## 两只鼹鼠（环形相遇 + 模运算）

**问题**：k 个扇形区域（0…k−1），A 顺时针、B 逆时针，求第一次同时在同一区域钻出地面的时间。

**相遇位置方程**：

- 第 x 秒：A 在 `la + x`，B 在 `lb - x`（模 k）
- 相遇：`la + x ≡ lb - x (mod k)` → **x = (lb - la + y·k) / 2**

**不可能相遇的判定**：

- k **偶数**时：`dis = (lb - la + k) % k`；若 **dis 为奇数** → 不可能相遇
- k **奇数**时：需调整 dis 的计算方式（见源码模板）

**枚举相遇时间并验证探头**：

```
(dis - fa) % ia == 0  &&  (dis - fb) % ib == 0
```

若满足则返回 `dis`；否则 `dis += step` 继续枚举，直到循环或超时。

```cpp
if (k % 2 == 0) {
    dis = (lb - la + k) % k;
    if (dis % 2 != 0) return -1;   // no answer
    dis = dis / 2;
    step = k / 2;
} else {
    dis = (lb - la + k) % k;
    if (dis % 2 == 0) dis = dis / 2;
    else dis = (dis + k) / 2;
    step = k;
}
```

**样例**：k=7, la=1, lb=5, ia=3, ib=5, fa=1, fb=2 → 答案 **37**。

---

## 考试/测试常见模式

| 场景/关键词 | 解题思路 |
|-------------|----------|
| "最近公共祖先/LCA" | 少量查询 → 暴力向上爬；批量离线 → **Tarjan**；多次在线 → **ST 表** |
| "一般树（非 BST）" | 不能用大小判断，用 Tarjan 或欧拉序 + RMQ |
| "环形相遇/模运算" | 列同余方程 → 枚举 x → 验证周期性条件 |
| "DFS + 并查集" | 想到 Tarjan 离线 LCA |
| "欧拉序 + 区间最值" | 想到 ST 表在线 LCA |

---

## 相关笔记

- [[并查集与连通性]]
- [[数论图论练习]]
