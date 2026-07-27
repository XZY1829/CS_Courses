# 763. 划分字母区间 - 题解

## 思路
1. 记录每个字母最后出现的位置
2. 贪心扫描：维护当前分区的右边界 = 分区内所有字母的最远位置
3. 当 `i == end` 时，当前分区结束

## 解法
```cpp
vector<int> partitionLabels(string s) {
    int last[26] = {};
    for (int i = 0; i < s.size(); i++) last[s[i]-'a'] = i;
    vector<int> result; int start=0, end=0;
    for (int i = 0; i < s.size(); i++) {
        end = max(end, last[s[i]-'a']);
        if (i == end) { result.push_back(end-start+1); start=end+1; }
    }
    return result;
}
```
## 复杂度
- **时间**：O(n)　**空间**：O(1)
