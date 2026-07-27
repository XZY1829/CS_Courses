---
source_pdf: 2-对象与类、this指针、构造函数与析构函数.pptx
part: 2.2
keywords: this-pointer, hidden-parameter, member-access, chaining
---

# this指针（★★★）

#cpp #class #this-pointer

## 概览表（一目了然）

| 条目 | 要点 |
|------|------|
| 本质 | 每个非静态成员函数的隐藏参数，指向调用对象 |
| 类型 | `ClassName *const this`（不可修改指向） |
| 自动性 | 编译器自动添加、自动传递对象地址 |
| 显式使用场景 | 返回对象自身、参数与成员同名、传递自身地址 |

## this指针的本质

**成员函数只有一份拷贝**，所有对象共享。编译器通过隐藏参数 `this` 区分是哪个对象在调用。

```cpp
class A {
    int x;
public:
    void g(int i) { x = i; }
};
```

编译器实际生成的代码等价于：

```cpp
void g(A *const this, int i) {
    this->x = i;
}

A a, b;
a.g(1);  // 编译为: g(&a, 1)
b.g(2);  // 编译为: g(&b, 2)
```

## 必须显式使用this的场景

### 1. 返回对象自身（链式调用）

```cpp
class A {
    int x;
public:
    A& inc() {
        ++x;
        return *this;  // 返回对象自身的引用
    }
};

A a;
a.inc().inc();  // x增加2（链式调用）
```

> [!warning] 返回引用 vs 返回值
> - `A& inc()` + `return *this` → 链式操作作用于同一对象
> - `A inc()` + `return *this` → 返回副本，后续操作不影响原对象

### 2. 参数名与成员名冲突

```cpp
class A {
    int x;
public:
    void set(int x) {
        this->x = x;  // this->x是成员，x是参数
    }
};
```

### 3. 将自身地址传递给其他函数

```cpp
void external(A *p);

class A {
public:
    void doSomething() {
        external(this);  // 把自己的地址传出去
    }
};
```

## 用C模拟C++的类

```
C++:                          C等价实现:
class A {                     struct A { int x, y; };
    int x, y;                 void f_A(struct A *this);
public:                       void g_A(struct A *this, int i) {
    void f();                     this->x = i;
    void g(int i) {               f_A(this);
        x = i; f();           }
    }
};                            struct A a, b;
A a, b;                       f_A(&a);
a.f();                        g_A(&a, 1);
a.g(1);                       f_A(&b);
b.f();                        g_A(&b, 2);
b.g(2);
```

---

## 考试/测试常见模式

| 场景/关键词 | 答案 |
|-------------|------|
| "this的类型" | `ClassName *const this`（指针常量） |
| "静态成员函数有this吗" | 没有，所以不能访问非静态成员 |
| "链式调用实现" | 返回 `*this` 的引用 |
| "成员名与参数名冲突" | 用 `this->member` 区分 |
| "为什么成员函数只有一份" | 所有对象共享代码，通过this区分数据 |

## 相关笔记

- [[类的定义与对象]]
- [[构造函数与析构函数]]
- [[静态成员]]
