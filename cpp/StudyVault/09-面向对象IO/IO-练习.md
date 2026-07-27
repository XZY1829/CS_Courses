---
source_pdf: 9-面向对象的IO.ppt + 12 函数式程序设计.ppt + 13事件驱动的程序设计.ppt
part: 9-11
keywords: practice, io, functional, event-driven, lambda
---

# IO、函数式与事件驱动 练习（9题）

#practice #cpp #io #functional #event-driven

## 相关概念

- [[iostream体系]]
- [[函数式程序设计]]
- [[事件驱动程序设计]]

---

## 第 1 题 - cout类型安全 [recall]

> cout/cin相比printf/scanf的两个核心优势是什么？

> [!answer]- 查看答案
> 1. **类型安全**：cout自动识别变量类型，不需要格式串，不会出现%d对应double的错误
> 2. **可扩展**：通过重载operator<<，可以输出任何自定义类型；printf只能输出基本类型

---

## 第 2 题 - 流操作符返回值 [application]

> 实现一个Date类的operator<<，支持 `cout << d1 << d2;`

> [!answer]- 查看答案
> ```cpp
> class Date {
>     int year, month, day;
>     friend ostream& operator<<(ostream& os, const Date& d);
> };
> ostream& operator<<(ostream& os, const Date& d) {
>     os << d.year << "/" << d.month << "/" << d.day;
>     return os;  // 必须返回os引用以支持链式
> }
> ```

---

## 第 3 题 - 文件IO操作 [application]

> 写出将一组整数写入文件再读回的完整代码框架。

> [!answer]- 查看答案
> ```cpp
> #include <fstream>
> // 写入
> ofstream fout("data.txt");
> for (int i = 0; i < n; i++)
>     fout << arr[i] << " ";
> fout.close();
> 
> // 读取
> ifstream fin("data.txt");
> int val;
> while (fin >> val)
>     process(val);
> fin.close();
> ```

---

## 第 4 题 - Lambda表达式 [application]

> 写出以下需求的Lambda表达式：
> 1. 判断一个数是否为偶数
> 2. 计算两数之和
> 3. 捕获外部变量threshold，判断x是否大于它

> [!answer]- 查看答案
> ```cpp
> // 1. 判断偶数
> auto isEven = [](int x) { return x % 2 == 0; };
> 
> // 2. 两数之和
> auto add = [](int a, int b) { return a + b; };
> 
> // 3. 捕获外部变量
> int threshold = 10;
> auto isGreater = [threshold](int x) { return x > threshold; };
> // 或 [&threshold] 引用捕获
> ```

---

## 第 5 题 - 纯函数判断 [recall]

> 以下哪些是纯函数？为什么？
> ```cpp
> int f1(int x) { return x * 2; }
> int f2(int x) { static int n=0; return x + n++; }
> int f3(int x) { cout << x; return x; }
> ```

> [!answer]- 查看答案
> - **f1是纯函数** ✓ — 无副作用，相同输入总是相同输出
> - **f2不是** ✗ — 有状态(static n)，每次调用结果不同
> - **f3不是** ✗ — 有副作用(输出到cout改变了外部状态)

---

## 第 6 题 - 命令式vs函数式 [application]

> 用函数式风格（递归、不修改数据）计算数组中所有正数的和。

> [!answer]- 查看答案
> ```cpp
> // 函数式风格
> int sumPositive(int arr[], int n) {
>     if (n == 0) return 0;
>     int last = arr[n-1] > 0 ? arr[n-1] : 0;
>     return last + sumPositive(arr, n-1);
> }
> 
> // 或使用STL+Lambda
> int sum = accumulate(v.begin(), v.end(), 0,
>     [](int acc, int x) { return x > 0 ? acc + x : acc; });
> ```

---

## 第 7 题 - 事件驱动理解 [recall]

> 事件驱动程序的"消息循环"是什么？为什么每个消息处理不宜太长？

> [!answer]- 查看答案
> **消息循环**：程序不断从消息队列中取消息并分发处理的循环：
> ```cpp
> while (GetMessage(&msg, ...))  // 取消息
>     DispatchMessage(&msg);      // 分发处理
> ```
> 
> **处理不宜太长的原因**：消息循环是单线程串行处理。如果某个消息处理耗时过长，程序无法处理后续消息（如鼠标点击、窗口刷新），导致程序"假死"。

---

## 第 8 题 - 高阶函数使用 [application]

> 使用STL的transform函数和Lambda，将vector中每个元素变为其平方。

> [!answer]- 查看答案
> ```cpp
> vector<int> v = {1, 2, 3, 4, 5};
> transform(v.begin(), v.end(), v.begin(),
>     [](int x) { return x * x; });
> // v = {1, 4, 9, 16, 25}
> ```
> transform参数：源起始、源结束、目标起始、转换函数

---

## 第 9 题 - 框架vs库 [analysis]

> "框架"和"库"在程序控制流方面有什么本质区别？这如何体现在事件驱动编程中？

> [!answer]- 查看答案
> | | 库 | 框架 |
> |--|---|------|
> | 控制流 | 你的代码调用库 | 框架调用你的代码 |
> | 主动性 | 程序员决定何时调用 | 框架决定何时调用 |
> | 例子 | printf, sort | MFC, Qt |
> 
> 在事件驱动中：
> - 框架封装了消息循环
> - 程序员只需重写消息处理函数（如OnPaint, OnKeyDown）
> - 框架在适当时机调用你的处理函数 — **好莱坞原则**："Don't call us, we'll call you"
