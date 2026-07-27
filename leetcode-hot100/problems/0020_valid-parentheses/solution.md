# 20. 有效的括号 - 题解

## 思路

栈：遇左括号入栈，遇右括号弹栈检查匹配。最后栈空则有效。

## 解法

```cpp
bool isValid(string s) {
    stack<char> stk;
    for (char c : s) {
        if (c=='(' || c=='[' || c=='{') stk.push(c);
        else {
            if (stk.empty()) return false;
            char top = stk.top(); stk.pop();
            if ((c==')' && top!='(') || (c==']' && top!='[') || (c=='}' && top!='{')) return false;
        }
    }
    return stk.empty();
}
```

## 复杂度

- **时间**：O(n)　　**空间**：O(n)
