# 438. 找到字符串中所有字母异位词 - 题解

## 思路

固定长度的滑动窗口 + 字符频次比较。

窗口大小恒为 `|p|`，滑过 s 的每个位置，比较窗口内字符频次是否与 p 的频次一致。

用两个长度 26 的数组 `need` 和 `window` 分别记录 p 和当前窗口的字符计数，直接用 `==` 比较即可。

## 解法

```cpp
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        if (s.size() < p.size()) return {};
        vector<int> result;
        vector<int> need(26, 0), window(26, 0);
        for (char c : p) need[c - 'a']++;
        int pLen = p.size();
        for (int i = 0; i < (int)s.size(); i++) {
            window[s[i] - 'a']++;           // 右端进窗
            if (i >= pLen)
                window[s[i - pLen] - 'a']--;  // 左端出窗
            if (window == need)
                result.push_back(i - pLen + 1);
        }
        return result;
    }
};
```

## 复杂度

- **时间**：O(n × 26) ≈ O(n)，每步比较两个长度 26 的数组
- **空间**：O(1)，两个固定大小数组

## 关键点

1. 窗口大小固定为 `|p|`，不需要动态伸缩
2. `vector<int>` 的 `==` 运算符可以直接比较两个频次数组
3. 也可以维护一个 `matched` 计数器避免每步比较 26 个元素，但纯小写字母场景下性能差异可忽略
