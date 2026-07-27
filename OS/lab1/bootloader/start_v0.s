/* Real Mode Hello World */
.code16

.global start
start:
    movw %cs, %ax
    movw %ax, %ds
    movw %ax, %es
    movw %ax, %ss

    movw $msg, %si
print_loop:
    movb (%si), %al
    cmpb $0, %al
    je switch_to_protected
    movb $0x0e, %ah
    int $0x10
    inc %si
    jmp print_loop


switch_to_protected:
    # 切换到保护模式
    cli
    data32 addr32 lgdt gdtDesc               # 加载 GDT
    call enable_a20
    movl %cr0, %eax             # 读取 CR0
    orl $0x1, %eax              # 设置 PE（保护模式使能位）
    movl %eax, %cr0             # 写回 CR0，使能保护模式

    # 使用远跳转进入 32 位代码段
    data32 ljmp $0x08, $start32        # 0x08 是 GDT 中代码段的选择子

enable_a20:
    inb $0x64, %al             # 读取8042控制器状态
    testb $0x02, %al           # 检查8042是否准备好接受命令
    jnz  enable_a20            # 如果没有准备好，则继续等待

    movb $0xD1, %al            # 启用A20线命令
    outb %al, $0x64            # 将命令发送到控制器

    inb $0x64, %al             # 读取控制器的状态寄存器
    testb $0x01, %al           # 检查8042是否准备好接受数据
    jnz  enable_a20            # 如果没有准备好，则继续等待

    movb $0xDF, %al            # 启用A20线
    outb %al, $0x60            # 发送数据到键盘控制器
    ret


msg:
    .asciz "Hello, World!"
gdt:
    .word 0x0, 0x0, 0x0, 0x0      # NULL 段（没有实际内存）
    .word 0xFFFF, 0x0, 0x9A00, 0xC0  # 代码段描述符（特权级 0，存在，代码段）
    .word 0xFFFF, 0x0, 0x9200, 0xC0  # 数据段描述符（特权级 0，存在，数据段）
    .word 0xFFFF, 0x0, 0x9200, 0xC0  # 堆栈段描述符（特权级 0，存在，数据段）

gdtDesc:
    .word (gdt - gdtDesc - 1)      # GDT 大小（偏移量 - 1）
    .long gdt                      # GDT 基地址

.code32
start32:
    # 设置段寄存器

    movw $0x10, %ax             # 加载数据段选择子（数据段描述符在 GDT 中的偏移为 0x10）
    movw %ax, %ds               # 设置 DS（数据段）
    movw %ax, %es               # 设置 ES（附加段）
    movw %ax, %fs               # 设置 FS
    movw %ax, %ss               # 设置 SS（堆栈段）

    movw $0x10, %ax             # 加载数据段选择子（堆栈段描述符在 GDT 中的偏移为 0x18）
    movw %ax, %gs               # 设置 GS

    movl $0x8000, %esp          # 设置堆栈指针，指向有效的堆栈位置
    movl $msg32, %esi           # 初始化 %esi 寄存器，指向 msg32 的地址
    movl $0xb8000, %edi         # 显存起始地址
    movb $0x07, %ah             # 字符颜色（灰色）
    
    call bootMain

print_loop32:
    movb (%esi), %al            # 读取字符
    cmpb $0, %al                # 是否到达字符串结尾
    je   bootMain
    movw %ax, (%edi)            # 将字符和颜色写入显存
    addl $2, %edi               # 移动到下一个字符位置
    inc %esi                    # 移动到下一个字符
    jmp print_loop32            # 继续循环

done:
    hlt                         # 停止程序执行
    jmp done                    # 死循环，确保程序终止

.p2align 2
msg32:
    .asciz "Hello,World!"
