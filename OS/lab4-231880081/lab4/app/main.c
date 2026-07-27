#include "lib.h"
#include "types.h"

#define N 5
#define LOOPS 2
sem_t forks[N];

void ph_problem() {
    int i = 0, ret = 0;

    for (int i = 0; i < N; i++) {
        sem_init(&forks[i], 1);
    }

    for (i = 0; i < N - 1; i++) {
        if (ret == 0) {
            ret = fork();
        } else if (ret > 0) {
            break;
        }
    }

    int pid = get_pid();
    for (int i = 0; i < LOOPS; i++) {
        //think
        printf("Ph %d: thinking...\n", pid);
        sleep(128);

        if (pid != 4) {
            //除了ph4，都先拿左边的
            sem_wait(&forks[pid - 1]);
            sem_wait(&forks[(pid) % N]);
        } else {
            //ph4，先拿右边的
            sem_wait(&forks[(pid) % N]);
            sem_wait(&forks[pid - 1]);
        }
        //eat
        printf("Ph %d: eating...\n", pid);
        sleep(128);
        //放回叉
        sem_post(&forks[(pid) % N]);
        sem_post(&forks[pid - 1]);
    }
    sem_destroy(&forks[pid]);
    if (pid != 0) {
        exit();
    }
    return;
}

int uEntry(void) {

    //pcp(4);
    // For lab4.1
    // Test 'scanf'
    int dec = 0;
    int hex = 0;
    char str[6];
    char cha = 0;
    int ret = 0;
    while (1) {
        printf("Input:\" Test %%c Test %%6s %%d %%x\"\n");
        ret = scanf(" Test %c Test %6s %d %x", &cha, str, &dec, &hex);
        printf("Ret: %d; %c, %s, %d, %x.\n", ret, cha, str, dec, hex);
        if (ret == 4)
            break;
    }

    // For lab4.2
    // Test 'Semaphore'

    int i = 4;

    sem_t sem;
    printf("Father Process: Semaphore Initializing.\n");
    ret = sem_init(&sem, 2);
    if (ret == -1) {
        printf("Father Process: Semaphore Initializing Failed.\n");
        exit();
    }

    ret = fork();
    if (ret == 0) {
        while (i != 0) {
            i--;
            printf("Child Process: Semaphore Waiting.\n");
            sem_wait(&sem);
            printf("Child Process: In Critical Area.\n");
        }
        printf("Child Process: Semaphore Destroying.\n");
        sem_destroy(&sem);
        exit();
    } else if (ret != -1) {
        while (i != 0) {
            i--;
            printf("Father Process: Sleeping.\n");
            sleep(128);
            printf("Father Process: Semaphore Posting.\n");
            sem_post(&sem);
        }
        printf("Father Process: Semaphore Destroying.\n");
        sem_destroy(&sem);

    }

    // For lab4.3
    // TODO: You need to design and test the philosopher problem.
    // Producer-Consumer problem and Reader& Writer Problem are optional.
    // Note that you can create your own functions.
    // Requirements are demonstrated in the guide.
    ph_problem();
    return 0;
}
