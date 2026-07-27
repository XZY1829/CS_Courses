---
source_pdf: 9-面向对象的IO.ppt
part: 5
keywords: iostream, file-io, stream-operator, inheritance-hierarchy
---

# iostream体系（★★★）

#cpp #oop #io #iostream

## 概览表（一目了然）

| 条目 | 要点 |
|------|------|
| 设计思想 | 用OOP（类继承+操作符重载）实现类型安全I/O |
| 核心类 | istream(输入)、ostream(输出)、iostream(双向) |
| 文件IO | ifstream、ofstream、fstream |
| 操作符 | `<<`(插入/输出) 、`>>`(提取/输入) |
| vs printf/scanf | 类型安全、可扩展、面向对象 |

## I/O类继承体系

```
         ios_base
            │
           ios
          /   \
    istream   ostream
      │    \  / │
      │  iostream│
      │         │
  ifstream  ofstream
         \  /
        fstream
```

**预定义对象**：
- `cin` — istream对象，关联标准输入
- `cout` — ostream对象，关联标准输出
- `cerr` — ostream对象，关联标准错误

## cout/cin 的优势

| 维度 | printf/scanf | cout/cin |
|------|-------------|----------|
| 类型安全 | ✗（格式串不匹配不报错） | ✓（自动推断类型） |
| 可扩展 | ✗（不能输出自定义类型） | ✓（重载<<即可） |
| 类型判断 | 程序员指定(%d,%f) | 编译器自动 |
| 面向对象 | ✗ | ✓ |

## 基本I/O操作

### 字节级操作

```cpp
istream in(...);
in.get(ch);         // 读一个字符
in.read(buf, 100);  // 读100字节

ostream out(...);
out.put(ch);          // 写一个字符
out.write(buf, 100);  // 写100字节
```

### 格式化操作（<<和>>）

```cpp
int x; double y;
cin >> x >> y;           // 自动类型识别
cout << x << " " << y;  // 链式输出
```

## 文件I/O

```cpp
#include <fstream>

// 写文件
ofstream fout("data.txt");
fout << "Hello" << endl;
fout.close();

// 读文件
ifstream fin("data.txt");
string line;
getline(fin, line);
fin.close();

// 文件打开模式
ofstream f("log.txt", ios::app);  // 追加模式
ifstream f("data.bin", ios::binary);  // 二进制模式
```

## 为自定义类重载<<和>>

```cpp
class Point {
    int x, y;
    friend ostream& operator<<(ostream& os, const Point& p);
    friend istream& operator>>(istream& is, Point& p);
};
ostream& operator<<(ostream& os, const Point& p) {
    return os << "(" << p.x << "," << p.y << ")";
}
istream& operator>>(istream& is, Point& p) {
    return is >> p.x >> p.y;
}
```

---

## 考试/测试常见模式

| 场景/关键词 | 答案 |
|-------------|------|
| "cout vs printf" | cout类型安全+可扩展 |
| "IO类继承关系" | ios→istream/ostream→iostream |
| "<<返回什么" | ostream& (支持链式) |
| "文件IO步骤" | 创建流对象→打开→读写→关闭 |
| "ios::app" | 追加模式 |

## 相关笔记

- [[流操作符重载]]
- [[文件IO]]
- [[操作符重载规则]]
