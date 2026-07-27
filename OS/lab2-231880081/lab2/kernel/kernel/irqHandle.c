#include "x86.h"
#include "device.h"

extern int displayRow;
extern int displayCol;

extern uint32_t keyBuffer[MAX_KEYBUFFER_SIZE];
extern int bufferHead;
extern int bufferTail;


void GProtectFaultHandle(struct TrapFrame *tf);

void KeyboardHandle(struct TrapFrame *tf);

void syscallHandle(struct TrapFrame *tf);

void syscallWrite(struct TrapFrame *tf);

void syscallPrint(struct TrapFrame *tf);

void syscallRead(struct TrapFrame *tf);

void syscallGetChar(struct TrapFrame *tf);

void syscallGetStr(struct TrapFrame *tf);


void irqHandle(struct TrapFrame *tf) { // pointer tf = esp
    /*
     * 中断处理程序
     */
    /* Reassign segment register */
    asm volatile("movw %%ax, %%ds"::"a"(KSEL(SEG_KDATA)));
    //asm volatile("movw %%ax, %%es"::"a"(KSEL(SEG_KDATA)));
    //asm volatile("movw %%ax, %%fs"::"a"(KSEL(SEG_KDATA)));
    //asm volatile("movw %%ax, %%gs"::"a"(KSEL(SEG_KDATA)));
    switch (tf->irq) {
        // TODO: 填好中断处理程序的调用
        case -1: {
            break;
        }
        case 0xd: {
            GProtectFaultHandle(tf);
            break;
        }
        case 0x21://keyboard
            KeyboardHandle(tf);
            break;
        case 0x80://syscall
            syscallHandle(tf);
            break;
        default:
            assert(0);
    }
}

void GProtectFaultHandle(struct TrapFrame *tf) {
    assert(0);
    return;
}

void printCharAtPos(char ch) {
    uint16_t data = ch | (0x0c << 8); // 设置字符和颜色属性
    int pos = (80 * displayRow + displayCol) * 2; // 计算屏幕位置
    asm volatile("movw %0, (%1)" ::"r"(data), "r"(pos + 0xb8000)); // 将字符写入视频内存
}

void KeyboardHandle(struct TrapFrame *tf) {
    uint32_t key = getKeyCode();  // 获取键盘输入的按键编码

    if (key == 0xe) { // 退格符
        // 删除字符，直到行首或缓冲区为空
        if (bufferHead != bufferTail) {
            bufferTail--;
            displayCol--;
            printCharAtPos(' '); // 打印空格覆盖字符
        }
    } else if (key == 0x1c) { // 回车符
        // 处理回车，换行
        displayRow++;
        displayCol = 0;
        keyBuffer[bufferTail] = '\n'; // 在缓冲区添加换行符
        bufferTail = (bufferTail + 1) % MAX_KEYBUFFER_SIZE;
        if (bufferTail == bufferHead) bufferHead = (bufferHead + 1) % MAX_KEYBUFFER_SIZE;
    } else if (key < 0x81 && ((key > 1 && key < 0xe) || (key > 0xf && key != 0x1d && key != 0x2a && key != 0x36 && key != 0x38 && key != 0x3a && key < 0x45))) {
        // 处理正常字符输入
        char c = getChar(key);
        printCharAtPos(c); // 显示字符
        displayCol++;
        keyBuffer[bufferTail] = key; // 将字符存入缓冲区
        bufferTail = (bufferTail + 1) % MAX_KEYBUFFER_SIZE;
        if (bufferTail == bufferHead) bufferHead = (bufferHead + 1) % MAX_KEYBUFFER_SIZE;
    }

    // 处理行列溢出，超过一行就换行
    if (displayCol == 80) {
        displayRow++;
        displayCol = 0;
    }

    if (displayRow == 25) { // 滚动屏幕
        displayRow = 24;
        displayCol = 0;
        scrollScreen();
    }

    // 更新光标位置
    updateCursor(displayRow, displayCol);
}


void syscallHandle(struct TrapFrame *tf) {
    switch (tf->eax) { // syscall number
        case 0:
            syscallWrite(tf);
            break; // for SYS_WRITE
        case 1:
            syscallRead(tf);
            break; // for SYS_READ
        default:
            break;
    }
}

void syscallWrite(struct TrapFrame *tf) {
    switch (tf->ecx) { // file descriptor
        case 0:
            syscallPrint(tf);
            break; // for STD_OUT
        default:
            break;
    }
}

void syscallPrint(struct TrapFrame *tf) {
    int sel = USEL(SEG_UDATA);//TODO: segment selector for user data, need further modification
    char *str = (char *) tf->edx;
    int size = tf->ebx;
    int i = 0;
    uint16_t data = 0;
    int pos = 0;
    asm volatile("movw %0, %%es"::"m"(sel));
    for (i = 0; i < size; i++) {
        // TODO: 完成光标的维护和打印到显存
        char ch;
        // 从用户空间读取字符
        asm volatile("movb %%es:(%1), %0" : "=r"(ch) : "r"(str + i));
        if (ch == '\n') {
            displayRow++;
            displayCol = 0;
        } else {
            data = ch | (0x0c << 8);
            pos = (80 * displayRow + displayCol) * 2;
            asm volatile("movw %0, (%1)"::"r"(data), "r"(pos + 0xb8000));
            displayCol++;
        }
        if (displayCol == 80) {
            displayRow++;
            displayCol = 0;
        }
        if (displayRow == 25) {
            displayRow = 24;
            displayCol = 0;
            scrollScreen();
        }
    }

    updateCursor(displayRow, displayCol);
}

void syscallRead(struct TrapFrame *tf) {
    switch (tf->ecx) { //file descriptor
        case 0:
            syscallGetChar(tf);
            break; // for STD_IN
        case 1:
            syscallGetStr(tf);
            break; // for STD_STR
        default:
            break;
    }
}

void syscallGetChar(struct TrapFrame *tf) {
    // TODO: 自由实现
    asm volatile("sti");
    while (bufferHead == bufferTail || keyBuffer[bufferTail - 1] != '\n')waitForInterrupt();
    asm volatile("cli");
    tf->eax = getChar(keyBuffer[bufferHead]);
    bufferHead = bufferTail;
}

void syscallGetStr(struct TrapFrame *tf) {

    int sel = USEL(SEG_UDATA);
    char *buf = (char*)tf->edx;
    int maxLen = tf->ebx;
    int idx = 0;
    char ch = 0;

    // 启用中断，等待用户输入
    asm volatile("sti");
    while (bufferHead == bufferTail || keyBuffer[bufferTail - 1] != '\n') {
        waitForInterrupt();
    }
    asm volatile("cli");

    // 设置用户态数据段
    asm volatile("movw %0, %%es"::"r"(sel));

    while (idx < maxLen - 1) {
        if (bufferHead == bufferTail) break; // 避免越界
        if (keyBuffer[bufferHead] == '\n') break; // 遇到换行终止

        ch = getChar(keyBuffer[bufferHead]);
        if (ch) {
            asm volatile("movb %1, %%es:(%0)"::"r"(buf + idx), "r"(ch) : "memory");
        }
        bufferHead = (bufferHead + 1) % MAX_KEYBUFFER_SIZE;
        idx++;
    }

    // 添加字符串结束符
    asm volatile("movb $0x00, %%es:(%0)"::"r"(buf + idx) : "memory");
}




