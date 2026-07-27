# 76. 最小覆盖子串 - 题解

## 思路

**滑动窗口模板题**。维护一个窗口，用两个哈希表 `need` 和 `window` 分别记录 t 和窗口内的字符频次。

1. **扩张**：`right` 右移，将 `s[right]` 加入窗口
2. **收缩**：当窗口已覆盖 t 的所有字符时，`left` 右移尝试缩小窗口，更新最小长度
3. 用 `matched` 计数器记录有多少种字符的频次已满足需求

## 解法

```cpp
class Solution {
public:
    string minWindow(string s, string t) {
        unordered_map<char, int> need, window;
        for (char c : t) need[c]++;
        int left = 0, matched = 0;
        int start = 0, minLen = INT_MAX;
        int required = need.size();

        for (int right = 0; right < (int)s.size(); right++) {
            char c = s[right];
            if (need.count(c)) {
                window[c]++;
                if (window[c] == need[c]) matched++;
            }
            while (matched == required) {
                if (right - left + 1 < minLen) {
                    minLen = right - left + 1;
                    start = left;
                }
                char d = s[left];
                if (need.count(d)) {
                    if (window[d] == need[d]) matched--;
                    window[d]--;
                }
                left++;
            }
        }
        return minLen == INT_MAX ? "" : s.substr(start, minLen);
    }
};
```

## 复杂度

- **时间**：O(|s| + |t|)
- **空间**：O(|Σ|)，字符集大小

## 关键点

1. `matched` 记录的是**满足数量要求的字符种类数**，不是字符个数
2. `matched == required` 表示所有需要的字符种类都满足了频次要求
3. 这道题是滑动窗口的"模板中的模板"，掌握后 3、438 等题都是简化版
