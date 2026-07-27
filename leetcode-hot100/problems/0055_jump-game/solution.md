# 55. 跳跃游戏 - 题解

## 思路
贪心维护最远可达位置 `maxReach`。如果当前位置 > maxReach，说明到不了。

## 解法
```cpp
bool canJump(vector<int>& nums) {
    int maxReach = 0;
    for (int i = 0; i < nums.size(); i++) {
        if (i > maxReach) return false;
        maxReach = max(maxReach, i + nums[i]);
    }
    return true;
}
```
## 复杂度
- **时间**：O(n)　**空间**：O(1)
