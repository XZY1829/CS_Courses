# 84. 柱状图中最大的矩形 - 题解

## 思路

**单调递增栈**：对每个柱子，找它向左向右能延伸的最远距离（即左右第一个比它矮的位置）。

末尾加哨兵 0 确保所有柱子都能被弹出处理。弹出时：
- 高度 = 弹出柱子的高度
- 宽度 = 栈空则为 `i`，否则为 `i - stk.top() - 1`

## 解法

```cpp
int largestRectangleArea(vector<int>& heights) {
    stack<int> stk;
    int maxArea = 0;
    heights.push_back(0);
    for (int i = 0; i < heights.size(); i++) {
        while (!stk.empty() && heights[i] < heights[stk.top()]) {
            int h = heights[stk.top()]; stk.pop();
            int w = stk.empty() ? i : i - stk.top() - 1;
            maxArea = max(maxArea, h * w);
        }
        stk.push(i);
    }
    heights.pop_back();
    return maxArea;
}
```

## 复杂度

- **时间**：O(n)　　**空间**：O(n)

## 关键点

1. 哨兵技巧：末尾加 0 避免循环后处理残余栈
2. 宽度计算是关键：`i - stk.top() - 1`（当前位置到栈顶下一个的距离）
3. 这是单调栈最经典也最难理解的应用题
