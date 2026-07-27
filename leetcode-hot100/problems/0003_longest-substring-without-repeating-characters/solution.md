# 3. 无重复字符的最长子串 - 题解

## 思路

经典**滑动窗口**题。维护一个窗口 `[left, right]`，保证窗口内没有重复字符。

- `right` 不断右移扩大窗口
- 当 `s[right]` 已在窗口中时，`left` 右移缩小窗口直到消除重复
- 每步更新最大长度

## 解法

```cpp
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_set<char> window;
        int left = 0, ans = 0;
        for (int right = 0; right < (int)s.size(); right++) {
            while (window.count(s[right])) {
                window.erase(s[left]);
                left++;
            }
            window.insert(s[right]);
            ans = max(ans, right - left + 1);
        }
        return ans;
    }
};
```

## 复杂度

- **时间**：O(n)，left 和 right 各最多移动 n 次
- **空间**：O(min(n, Σ))，Σ 为字符集大小

## 关键点

1. 滑动窗口的本质：right 扩张、left 收缩，维护窗口内的不变量（无重复）
2. 也可用 `unordered_map<char, int>` 记录字符上次出现的位置，直接跳转 left，避免逐步收缩
3. 字符串问题中，"子串"要求连续、"子序列"不要求连续——本题是子串
