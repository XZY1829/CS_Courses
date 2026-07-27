---
source_pdf: 17-并发-线程API(1).pdf
part: 7
keywords: pthread, pthread-create, pthread-join, mutex, condition-variable
---

# 线程 API（★★）

#concurrency #thread

## 概览表（一目了然）

| API | 功能 |
|-----|------|
| `pthread_create(thread, attr, start_routine, arg)` | 创建线程 |
| `pthread_join(thread, value_ptr)` | 等待线程完成 |
| `pthread_mutex_lock/unlock` | 锁操作 |
| `pthread_cond_wait/signal` | 条件变量操作 |

## 线程创建

- `thread`：pthread_t 指针
- `start_routine`：函数指针（线程入口）
- `arg`：传给函数的参数（void* 类型，可传任意类型）

> [!warning] 返回值陷阱
> **绝不要**返回栈上变量的地址（函数返回后栈帧销毁 → 悬空指针）。应在**堆上**分配返回值。

## 锁的使用要点

1. 必须**正确初始化**：`PTHREAD_MUTEX_INITIALIZER` 或 `pthread_mutex_init()`
2. 始终**检查错误码**
3. `trylock()`：非阻塞获取；`timedlock()`：超时获取

## 条件变量要点

- `wait()` 需传入**已锁定**的 mutex → 自动释放锁并睡眠 → 唤醒时重新获取锁
- 始终使用 **while 循环**检查条件（防止虚假唤醒）
- **绝不要**用简单 flag+自旋替代条件变量

## 编译

```bash
gcc -o prog prog.c -Wall -pthread
```

---

## 考试/测试常见模式

| 场景/关键词 | 答案 |
|-------------|------|
| "为什么 wait 需要 mutex" | wait 内部会释放锁+睡眠，唤醒后重新获取锁 |
| "为什么用 while 不用 if" | 防止**虚假唤醒**（Mesa 语义） |

## 相关笔记
- [[并发与线程]]
- [[锁]]
- [[条件变量]]
