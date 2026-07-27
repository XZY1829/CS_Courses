#include "boot.h"


#define SECTSIZE 512
#define APP_LOAD_ADDR 0x8c00  // 定义应用程序加载地址
#define APP_SECTOR  1
#define APP_SIZE 2

void bootMain(void) {
    char *app = (char *) APP_LOAD_ADDR;
    int i;
    char *app_mem = (char *) APP_LOAD_ADDR;
    for (i = 0; i < APP_SIZE; ++i) {
        readSect(app_mem + i * SECTSIZE, APP_SECTOR + i);
    }
    void (*app_entry)(void) = (void (*)(void)) APP_LOAD_ADDR;
    app_entry();
}


void waitDisk(void) { // waiting for disk
    while ((inByte(0x1F7) & 0xC0) != 0x40);
}

void readSect(void *dst, int offset) { // reading a sector of disk
    int i;
    waitDisk();
    outByte(0x1F2, 1);
    outByte(0x1F3, offset);
    outByte(0x1F4, offset >> 8);
    outByte(0x1F5, offset >> 16);
    outByte(0x1F6, (offset >> 24) | 0xE0);
    outByte(0x1F7, 0x20);

    waitDisk();
    for (i = 0; i < SECTSIZE / 4; i++) {
        ((int *) dst)[i] = inLong(0x1F0);
    }
}
