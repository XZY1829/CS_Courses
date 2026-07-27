---
source_pdf: 9-面向对象的IO.ppt
part: 5
keywords: file-io, ifstream, ofstream, fstream, binary, text
---

# 文件IO（★★★）

#cpp #oop #io #file-io

## 概览表（一目了然）

| 条目 | 要点 |
|------|------|
| 输出类 | `ofstream`（写文件） |
| 输入类 | `ifstream`（读文件） |
| 双向类 | `fstream`（读写） |
| 打开模式 | `ios::in`/`ios::out`/`ios::app`/`ios::binary` |
| 基本步骤 | 创建对象→打开→判断→读写→关闭 |
| 文本vs二进制 | 文本有换行转换，二进制原样读写 |

## 文件I/O基本步骤

```
创建流对象 → 打开文件 → 判断成功 → 读写操作 → 关闭文件
```

## 文件输出（写文件）

### 打开文件

```cpp
#include <fstream>

// 方式1：直接构造
ofstream out_file("d:\\data.txt", ios::out);

// 方式2：先构造后打开
ofstream out_file;
out_file.open("d:\\data.txt", ios::out);
```

### 打开模式

| 模式 | 含义 |
|------|------|
| `ios::out` | 写模式（默认），清空已有内容 |
| `ios::app` | 追加模式，保留已有内容 |
| `ios::binary` | 二进制模式 |
| `ios::out \| ios::binary` | 二进制写 |

### 判断是否打开成功

```cpp
if (!out_file.is_open()) {  // 或 out_file.fail() 或 !out_file
    cerr << "无法打开文件" << endl;
    exit(-1);
}
```

### 文本方式输出

```cpp
int x = 12; double y = 12.3;
ofstream out("data.txt");
if (!out) exit(-1);
out << x << ' ' << y << endl;  // 文件内容：12 12.3\n
out.close();
```

### 二进制方式输出

```cpp
ofstream out("data.bin", ios::out | ios::binary);
if (!out) exit(-1);
out.write((char*)&x, sizeof(x));   // 写4字节
out.write((char*)&y, sizeof(y));   // 写8字节
out.close();
```

## 文件输入（读文件）

### 文本方式输入

```cpp
ifstream in("data.txt", ios::in);
if (!in) exit(-1);
int x; double y;
in >> x >> y;        // 自动类型转换
in.close();
```

### 二进制方式输入

```cpp
ifstream in("data.bin", ios::in | ios::binary);
if (!in) exit(-1);
in.read((char*)&x, sizeof(x));
in.read((char*)&y, sizeof(y));
in.close();
```

### 循环读取直到文件末尾

```cpp
ifstream in("data.txt");
if (!in) exit(-1);
int x;
in >> x;
while (!in.fail()) {   // 判断读取是否成功
    // 处理x
    in >> x;           // 继续读下一个
}
in.close();
```

> [!warning] 不要用 `while(!in.eof())`
> `eof()` 在读取**失败后**才变true，可能导致最后一次数据处理错误。
> 正确做法：用 `in.fail()` 或直接判断 `in >> x` 的返回值。

## 文本方式 vs 二进制方式

| | 文本方式 | 二进制方式 |
|--|---------|-----------|
| 存储格式 | 人类可读字符 | 内存原始字节 |
| 换行处理 | Windows: `\n` ↔ `\r\n` | 不做任何转换 |
| 整数-1234567 | 8字节("−1234567") | 4字节(补码) |
| 优点 | 可用记事本查看 | 紧凑、高效 |
| 使用场景 | 配置文件、日志 | 图片、音频、自定义格式 |

## 结构体的文件读写

```cpp
struct Student {
    int no;
    char name[10];
    int scores[5];
};
Student s = {1001, "张三", {90, 95, 85, 75, 88}};

// 二进制写入
ofstream out("students.dat", ios::out | ios::binary);
out.write((char*)&s, sizeof(s));
out.close();

// 二进制读取
Student s2;
ifstream in("students.dat", ios::in | ios::binary);
in.read((char*)&s2, sizeof(s2));
in.close();
```

## 关闭文件的重要性

```cpp
out_file.close();
```

关闭的作用：
1. **刷新缓冲区**：将内存缓冲中的数据写入磁盘
2. **释放资源**：归还操作系统资源
3. 程序正常结束时系统会自动关闭，但**显式关闭**是好习惯

---

## 考试/测试常见模式

| 场景/关键词 | 答案 |
|-------------|------|
| "文件IO三步" | 打开→读写→关闭 |
| "ios::app vs ios::out" | app追加保留内容，out清空重写 |
| "二进制方式" | `ios::binary`，用read/write |
| "判断文件是否打开" | `!file` 或 `file.fail()` 或 `!file.is_open()` |
| "为什么不用eof()循环" | eof在读取失败后才true，多处理一次 |
| "文本方式换行转换" | Windows下 `\n` ↔ `\r\n` |

## 相关笔记

- [[iostream体系]]
- [[流操作符重载]]
