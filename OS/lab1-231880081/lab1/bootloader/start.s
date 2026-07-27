
.code16

.global start
start:
    movw %cs, %ax
    movw %ax, %ds
    movw %ax, %es
    movw %ax, %ss


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
    inb $0x92, %al
    orb $0x02, %al
    outb %al, $0x92
    ret

gdt:
    .word 0x0, 0x0, 0x0, 0x0      # NULL 段（没有实际内存）
    .word 0xFFFF, 0x0, 0x9A00, 0xC0  # 代码段描述符（特权级 0，存在，代码段）
    .word 0xFFFF, 0x0, 0x9200, 0xC0  # 数据段描述符（特权级 0，存在，数据段）
    .word 0x07FF, 0x8000, 0x920B, 0x40  # VGA段描述符（特权级 0，存在，数据段）

gdtDesc:
    .word (gdtDesc - gdt - 1)      # GDT 大小（偏移量 - 1）
    .long gdt                      # GDT 基地址



.code32
start32:
    # 设置段寄存器

    movw $0x10, %ax             # 加载数据段选择子（数据段描述符在 GDT 中的偏移为 0x10）
    movw %ax, %ds               # 设置 DS（数据段）
    movw %ax, %es               # 设置 ES（附加段）
    movw %ax, %fs               # 设置 FS
    movw %ax, %ss               # 设置 SS（堆栈段）

    movw $0x18, %ax             # 加载数据段选择子（堆栈段描述符在 GDT 中的偏移为 0x18）
    movw %ax, %gs               # 设置 GS

    movl $0x8000, %esp          # 设置堆栈指针，指向有效的堆栈位置

    jmp bootMain



done:
    hlt                         # 停止程序执行
    jmp done                    # 死循环，确保程序终止

.p2align 2
