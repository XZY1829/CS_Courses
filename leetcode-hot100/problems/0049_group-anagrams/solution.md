# 49. 字母异位词分组 - 题解

## 思路

字母异位词（anagram）的本质：两个字符串包含**完全相同的字符及其出现次数**，只是排列不同。

因此需要一个"规范化"方法，让所有异位词映射到同一个 key。两种常见做法：

1. **排序法**：将每个字符串排序后作为 key，`"eat" → "aet"`、`"tea" → "aet"`
2. **计数法**：用 26 个字母的出现次数构成 key，`"eat" → "1a1e1t"`

排序法实现更简洁，在字符串长度不大时性能足够。

## 解法

```cpp
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> groups;
        for (const string& s : strs) {
            string key = s;
            sort(key.begin(), key.end()); // 排序后作为分组 key
            groups[key].push_back(s);
        }
        vector<vector<string>> result;
        for (auto& [k, v] : groups) {
            result.push_back(move(v));
        }
        return result;
    }
};
```

## 复杂度

- **时间**：O(n · k log k)，n 为字符串个数，k 为最大字符串长度
- **空间**：O(n · k)，哈希表存所有字符串

## 关键点

1. 排序法简洁但引入 O(k log k) 的排序开销；如果 k 很大，可用计数法将单个字符串处理降为 O(k)
2. 结果的**组间顺序和组内顺序**不影响正确性，题目允许任意顺序返回
3. 空字符串 `""` 排序后仍然是 `""`，能正确分组
