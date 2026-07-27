# 394. 字符串解码 - 题解

## 思路

双栈：数字栈 + 字符串栈。
- 遇数字：累积到 `num`
- 遇 `[`：把当前 `cur` 和 `num` 分别压栈，重置
- 遇 `]`：弹出数字和之前的字符串，重复 cur 拼接
- 遇字母：直接加到 `cur`

## 解法

```cpp
string decodeString(string s) {
    stack<string> strStk; stack<int> numStk;
    string cur; int num = 0;
    for (char c : s) {
        if (isdigit(c)) num = num*10 + (c-'0');
        else if (c == '[') { strStk.push(cur); numStk.push(num); cur=""; num=0; }
        else if (c == ']') { string tmp=cur; cur=strStk.top(); strStk.pop(); int k=numStk.top(); numStk.pop(); while(k--) cur+=tmp; }
        else cur += c;
    }
    return cur;
}
```

## 复杂度

- **时间**：O(输出长度)　　**空间**：O(嵌套深度)

## 关键点

1. 支持嵌套：`3[a2[c]]` → `accaccacc`
2. 数字可能多位：`100[a]`
