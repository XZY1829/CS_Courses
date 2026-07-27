---
source_pdf: 3-拷贝构造函数、常量对象、静态成员、友元.pptx
part: 2.5
keywords: const-member-function, const-object, static-member, shared-data
---

# 常量对象与const成员函数（★★★）

#cpp #class #const #static-member

## 概览表（一目了然）

| 条目 | 要点 |
|------|------|
| 常成员函数 | 函数声明后加 `const`，承诺不修改对象状态 |
| 常量对象 | `const A obj` 或 `const A &ref`，只能调用常成员函数 |
| 区分 | 获取状态的函数 → const；改变状态的函数 → 非const |
| 编译器保证 | 在const成员函数中修改数据成员会报错 |

## 常成员函数 (const Member Function)

```cpp
class Date {
    int year, month, day;
public:
    void set(int y, int m, int d);     // 修改状态
    int get_day() const;                // 获取状态（const）
    int get_month() const;
    int get_year() const;
};

int Date::get_day() const {
    // day = 0;  // Error! 不能修改数据成员
    return day;  // OK，只读取
}
```

**规则**：
- `const` 写在参数列表的 `)` 后面
- 定义和声明都要写 `const`
- const函数中不能修改任何非mutable数据成员

### 常量对象只能调用常成员函数

```cpp
void f(const Date &d) {
    d.get_day();      // OK — 调用const成员函数
    d.set(2024,1,1);  // Error! 不能对常量对象调用非const函数
}
```

> [!warning] const成员函数的漏洞
> 如果成员是指针，const函数不能修改指针本身，但**可以修改指针指向的内容**！
> ```cpp
> void f() const {
>     p = new char[20];    // Error（修改了p）
>     strcpy(p, "ABCD");   // 编译通过！（p的值没变）
> }
> ```

---

## 考试/测试常见模式

| 场景/关键词 | 答案 |
|-------------|------|
| "const成员函数能修改什么" | 什么都不能改（mutable成员除外） |
| "常量对象能调什么" | 只能调用const成员函数 |
| "const写在哪" | 参数列表的 `)` 后面 |
| "指针成员的陷阱" | const不保护指针指向的内容 |

## 相关笔记

- [[静态成员]]
- [[友元]]
- [[类的定义与对象]]
