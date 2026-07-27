#include "x86.h"
#include "device.h"

extern ProcessTable pcb[MAX_PCB_NUM];
extern int current;

extern TSS tss;
extern int displayRow;
extern int displayCol;


void GProtectFaultHandle(struct StackFrame *sf);

void syscallHandle(struct StackFrame *sf);

void syscallWrite(struct StackFrame *sf);

void syscallPrint(struct StackFrame *sf);

void syscallFork(struct StackFrame *sf);

void syscallSleep(struct StackFrame *sf);

void syscallExit(struct StackFrame *sf);

void timerHandle(struct StackFrame *sf);


void irqHandle(struct StackFrame *sf) { // pointer sf = esp
    /* Reassign segment register */
    asm volatile("movw %%ax, %%ds"::"a"(KSEL(SEG_KDATA)));
    /* Save esp to stackTop */
    uint32_t tmpStackTop = pcb[current].stackTop;
    pcb[current].prevStackTop = pcb[current].stackTop;
    pcb[current].stackTop = (uint32_t) sf;


    switch (sf->irq) {
        case -1:
            break;
        case 0xd:
            GProtectFaultHandle(sf);
            break;
        case 0x20:
            timerHandle(sf);
            break;
        case 0x80:
            syscallHandle(sf);
            break;
        default:
            assert(0);
    }
    /* Recover stackTop */
    pcb[current].stackTop = tmpStackTop;
}

void GProtectFaultHandle(struct StackFrame *sf) {
    assert(0);
    return;
}


void syscallHandle(struct StackFrame *sf) {
    switch (sf->eax) { // syscall number
        case 0:
            syscallWrite(sf);
            break; // for SYS_WRITE
        case 1:
            syscallFork(sf);
            break;
        case 2:
            break;
        case 3:
            syscallSleep(sf);
            break;
        case 4:
            syscallExit(sf);
            break;

            /* Add Fork,Sleep... */

        default:
            break;
    }
}

void timerHandle(struct StackFrame *sf) {
    //
    // 1. 更新所有 BLOCKED 进程的 sleep 时间
    for (int i = 0; i < MAX_PCB_NUM; i++) {
        if (pcb[i].state == STATE_BLOCKED) {
            if (pcb[i].sleepTime > 0) {
                pcb[i].sleepTime--;
                if (pcb[i].sleepTime == 0) {
                    pcb[i].state = STATE_RUNNABLE;
                }
            }
        }
    }

    // 2. 设置当前进程状态为 RUNNABLE（准备让出 CPU）
    if (pcb[current].state == STATE_RUNNING) {
        if (pcb[current].timeCount != MAX_TIME_COUNT)pcb[current].timeCount++;
        if (pcb[current].timeCount == MAX_TIME_COUNT) {
            pcb[current].timeCount = 0;
            pcb[current].state = STATE_RUNNABLE;
        }
    }

    // 3. 切换上下文
    if (pcb[current].state == STATE_RUNNING) return;


    int i = -1;
    for (i = (current + 1) % MAX_PCB_NUM; i != current; i = (i + 1) % MAX_PCB_NUM) {
        if (pcb[i].state == STATE_RUNNABLE)break;
    }
    if (current == i) {
        if (pcb[i].state == STATE_RUNNABLE)current = i;
        else current = 0;
    } else current = i;
    pcb[current].state = STATE_RUNNING;
    uint32_t tmpStackTop = pcb[current].stackTop;
    tss.esp0 = (uint32_t) &pcb[current].stackTop;
    pcb[current].stackTop = pcb[current].prevStackTop;
    // 恢复被调度进程的栈并返回
    asm volatile("movl %0, %%esp"::"m"(tmpStackTop));
    asm volatile("popl %gs");
    asm volatile("popl %fs");
    asm volatile("popl %es");
    asm volatile("popl %ds");
    asm volatile("popal");
    asm volatile("addl $8, %esp"); // skip error code & irq
    asm volatile("iret");
}


void syscallWrite(struct StackFrame *sf) {
    switch (sf->ecx) { // file descriptor
        case 0:
            syscallPrint(sf);
            break; // for STD_OUT
        default:
            break;
    }
}

void syscallPrint(struct StackFrame *sf) {
    int sel = sf->ds; // segment selector for user data, need further modification
    char *str = (char *) sf->edx;
    int size = sf->ebx;
    int i = 0;
    int pos = 0;
    char character = 0;
    uint16_t data = 0;
    asm volatile("movw %0, %%es"::"m"(sel));
    for (i = 0; i < size; i++) {
        asm volatile("movb %%es:(%1), %0":"=r"(character):"r"(str + i));
        if (character == '\n') {
            displayRow++;
            displayCol = 0;
            if (displayRow == 25) {
                displayRow = 24;
                displayCol = 0;
                scrollScreen();
            }
        } else {
            data = character | (0x0c << 8);
            pos = (80 * displayRow + displayCol) * 2;
            asm volatile("movw %0, (%1)"::"r"(data), "r"(pos + 0xb8000));
            displayCol++;
            if (displayCol == 80) {
                displayRow++;
                displayCol = 0;
                if (displayRow == 25) {
                    displayRow = 24;
                    displayCol = 0;
                    scrollScreen();
                }
            }
        }
        //asm volatile("int $0x20"); //XXX Testing irqTimer during syscall
        //asm volatile("int $0x20":::"memory"); //XXX Testing irqTimer during syscall
    }

    updateCursor(displayRow, displayCol);
    // take care of return value
    return;
}

void syscallFork(struct StackFrame *sf) {
    //

    int child = -1;
    for (int i = 0; i < MAX_PCB_NUM; i++) {
        if (pcb[i].state == STATE_DEAD) {
            child = i;
            break;
        }
    }
    if (child < 0) {
        pcb[current].regs.eax = -1;
        return;
    }

    // 1) 复制 TrapFrame
    pcb[child].regs.edi = sf->edi;
    pcb[child].regs.esi = sf->esi;
    pcb[child].regs.ebp = sf->ebp;
    pcb[child].regs.xxx = sf->xxx;
    pcb[child].regs.ebx = sf->ebx;
    pcb[child].regs.edx = sf->edx;
    pcb[child].regs.ecx = sf->ecx;
    pcb[child].regs.eax = sf->eax;
    pcb[child].regs.irq = sf->irq;
    pcb[child].regs.error = sf->error;
    pcb[child].regs.eip = sf->eip;
    pcb[child].regs.esp = sf->esp;
    pcb[child].regs.eflags = sf->eflags;
    pcb[child].regs.cs = USEL((1 + child * 2));
    pcb[child].regs.ss = USEL((2 + child * 2));
    pcb[child].regs.ds = USEL((2 + child * 2));
    pcb[child].regs.es = USEL((2 + child * 2));
    pcb[child].regs.fs = USEL((2 + child * 2));
    pcb[child].regs.gs = USEL((2 + child * 2));
    // 2) 复制 PCB 其他字段
    pcb[child].pid = child;
    pcb[child].state = STATE_RUNNABLE;
    pcb[child].timeCount = 0;
    pcb[child].sleepTime = 0;

    // 3) 计算栈基址差，拷贝内核栈
    uint32_t pbase = (uint32_t) &pcb[current].stack[0];
    uint32_t cbase = (uint32_t) &pcb[child].stack[0];
    uint32_t diff = cbase - pbase;

    pcb[child].prevStackTop = pcb[current].prevStackTop + diff;
    pcb[child].stackTop = pcb[current].stackTop + diff;

    // 4) 复制用户空间
    for (int j = 0; j < 0x100000; j++) {
        *(uint8_t *) (j + (child + 1) * 0x100000) = *(uint8_t *) (j + (current + 1) * 0x100000);
    }

    // 5) 设置 fork 返回值
    pcb[child].regs.eax = 0;    // 子进程返回 0
    pcb[current].regs.eax = child; // 父进程返回 child PID
    return;
}

void syscallSleep(struct StackFrame *sf) {
    //
    uint32_t ticks = sf->ecx;
    if (ticks == 0) {
        sf->eax = -1;  // 参数非法
        return;
    }

    // 设置当前进程阻塞
    pcb[current].sleepTime = ticks;
    pcb[current].state = STATE_BLOCKED;
    // 主动触发一次时钟中断以进行调度
    asm volatile("int $0x20");
    sf->eax = pcb[current].sleepTime;
    return;
}

void syscallExit(struct StackFrame *sf) {
    //
    pcb[current].state = STATE_DEAD;
    asm volatile("int $0x20");
    sf->eax = 0;
    return;
}



