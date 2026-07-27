# 17. 电话号码的字母组合 - 题解

## 思路

回溯：对每个数字，遍历其对应的所有字母，逐位拼接。

## 解法

```cpp
class Solution {
    const string mapping[10] = {"","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
    vector<string> result;
    string path;
    void backtrack(const string& digits, int idx) {
        if (idx == digits.size()) { result.push_back(path); return; }
        for (char c : mapping[digits[idx] - '0']) {
            path.push_back(c);
            backtrack(digits, idx + 1);
            path.pop_back();
        }
    }
public:
    vector<string> letterCombinations(string digits) {
        if (digits.empty()) return {};
        result.clear(); path.clear(); backtrack(digits, 0); return result;
    }
};
```

## 复杂度

- **时间**：O(4^n × n)，n 为位数
- **空间**：O(n)

## 关键点

1. 映射表预存每个数字对应的字母
2. 7 和 9 对应 4 个字母，其余对应 3 个
