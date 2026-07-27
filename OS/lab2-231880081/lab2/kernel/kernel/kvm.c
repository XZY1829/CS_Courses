#include "x86.h"
#include "device.h"

#define SECTSIZE  512

SegDesc gdt[NR_SEGMENTS];       // the new GDT, NR_SEGMENTS=7, defined in x86/memory.h
TSS tss;

void initSeg() { // setup kernel segements
    gdt[SEG_KCODE] = SEG(STA_X | STA_R, 0, 0xffffffff, DPL_KERN);
    gdt[SEG_KDATA] = SEG(STA_W, 0, 0xffffffff, DPL_KERN);
    //gdt[SEG_UCODE] = SEG(STA_X | STA_R, 0,       0xffffffff, DPL_USER);
    gdt[SEG_UCODE] = SEG(STA_X | STA_R, 0x00200000, 0x000fffff, DPL_USER);
    //gdt[SEG_UDATA] = SEG(STA_W,         0,       0xffffffff, DPL_USER);
    gdt[SEG_UDATA] = SEG(STA_W, 0x00200000, 0x000fffff, DPL_USER);
    gdt[SEG_TSS] = SEG16(STS_T32A, &tss, sizeof(TSS) - 1, DPL_KERN);
    gdt[SEG_TSS].s = 0;
    setGdt(gdt, sizeof(gdt)); // gdt is set in bootloader, here reset gdt in kernel

    /*
     * 初始化TSS
     */
    tss.esp0 = 0x1fffff;
    tss.ss0 = KSEL(SEG_KDATA);
    asm volatile("ltr %%ax"::"a" (KSEL(SEG_TSS)));

    /*设置正确的段寄存器*/
    asm volatile("movw %%ax,%%ds"::"a" (KSEL(SEG_KDATA)));
    //asm volatile("movw %%ax,%%es":: "a" (KSEL(SEG_KDATA)));
    //asm volatile("movw %%ax,%%fs":: "a" (KSEL(SEG_KDATA)));
    //asm volatile("movw %%ax,%%gs":: "a" (KSEL(SEG_KDATA)));
    asm volatile("movw %%ax,%%ss"::"a" (KSEL(SEG_KDATA)));

    lLdt(0);

}

void enterUserSpace(uint32_t entry) {

    /*
     * Before enter user space
     * you should set the right segment registers here
     * and use 'iret' to jump to ring3
     */


    asm volatile("movw %%ax, %%es"::"a"(USEL(SEG_UDATA)));
    asm volatile("movw %%ax, %%ds"::"a"(USEL(SEG_UDATA)));
    asm volatile("pushw %0"::"i"(USEL(SEG_UDATA)));
    asm volatile("pushl %0"::"i"(128 << 10));
    asm volatile("pushl %0"::"i"(0x2));
    asm volatile("pushl %0"::"i"(USEL(SEG_UCODE)));
    asm volatile("pushl %0"::"m"(entry));
    asm volatile("iret");
}

void loadUMain(void) {
    // TODO: 参照bootloader加载内核的方式
    int i = 0;
    int offset = 0x1000;
    unsigned int elf = 0x200000;
    uint32_t uMainEntry = 0x200000;

    for (i = 0; i < 200; i++) {
        readSect((void*)(elf + i*512), 201+i);
    }

    uMainEntry=((struct ELFHeader *)elf)->entry;

    for (i = 0; i < 200 * 512; i++) {
        *(unsigned char *)(elf + i) = *(unsigned char *)(elf + i + offset);
    }
    enterUserSpace(uMainEntry);
}


