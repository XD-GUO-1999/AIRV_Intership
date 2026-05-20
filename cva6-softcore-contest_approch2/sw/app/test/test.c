#include <stdio.h>
#include <stdint.h>

static inline int32_t mac4(uint32_t a, uint32_t b)
{
    int32_t r = 0;
    asm volatile(
        "mac4 %[r], %[a], %[b]\n\t"
        : [r] "+r" (r)
        : [a] "r" (a), [b] "r" (b)
    );
    return r;
}

int main(void)
{
    uint32_t a = 0x04030201;   // 1,2,3,4
    uint32_t b = 0x04030201;   // 1,2,3,4

    int32_t r = mac4(a, b);

    printf("r = %d\n", r);     // expected: 30
    return 0;
}