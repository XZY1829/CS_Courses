# 45. 跳跃游戏 II - 题解

## 思路
贪心 BFS 思想：维护当前跳跃的边界 `curEnd` 和能到达的最远位置 `farthest`。到达边界时必须跳一次。

## 解法
```cpp
int jump(vector<int>& nums) {
    int jumps = 0, curEnd = 0, farthest = 0;
    for (int i = 0; i < nums.size() - 1; i++) {
        farthest = max(farthest, i + nums[i]);
        if (i == curEnd) { jumps++; curEnd = farthest; }
    }
    return jumps;
}
```
## 复杂度
- **时间**：O(n)　**空间**：O(1)
