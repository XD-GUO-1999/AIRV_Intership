#include <stdint.h>
#include <stdio.h>

#ifndef __GNUC__
#error "This test expects GCC-style inline assembly"
#endif

// 4-byte aligned, because we use lw to load 4 packed int8/uint8 values
static uint8_t input_data[8] __attribute__((aligned(4))) = {
    1, 2, 3, 4, 5, 6, 7, 8
};

static int8_t weight_data[8] __attribute__((aligned(4))) = {
    1, -1, 2, -2, 3, -3, 4, -4
};

static int32_t reference_mac8(const uint8_t *a, const int8_t *b, int32_t init)
{
    int32_t sum = init;

    for (int i = 0; i < 8; ++i) {
        sum += (int32_t)a[i] * (int32_t)b[i];
    }

    return sum;
}

static inline int32_t mac8i_test_once(const uint8_t *a, const int8_t *b, int32_t init)
{
    int32_t sum = init;

    asm volatile(
        // Explicit operands for mac8i
        "lw t3, 0(%[pa])\n\t"      // t3 = input[0..3]
        "lw t4, 0(%[pb])\n\t"      // t4 = weight[0..3]

        // Implicit operands for mac8i
        "lw t1, 4(%[pa])\n\t"      // t1/x6 = input[4..7]
        "lw t2, 4(%[pb])\n\t"      // t2/x7 = weight[4..7]

        // Temporary debug nops.
        // Keep them if hidden dependency tracking for x6/x7 is not finished yet.
        // Remove them later after scoreboard/forwarding is correct.
        "nop\n\t"
        "nop\n\t"
        "nop\n\t"

        // MAC8I semantic:
        // sum = old_sum + dot4(t3,t4) + dot4(t1,t2)
        "mac8im %[acc], t3, t4\n\t"

        : [acc] "+r" (sum)
        : [pa] "r" (a),
          [pb] "r" (b)
        : "t1", "t2", "t3", "t4", "memory"
    );

    return sum;
}

int main(void)
{
    int32_t init = 100;

    int32_t ref = reference_mac8(input_data, weight_data, init);
    int32_t hw  = mac8i_test_once(input_data, weight_data, init);

    printf("MAC8I test\n");
    printf("init = %ld\n", init);
    printf("ref  = %ld\n", ref);
    printf("hw   = %ld\n", hw);

    if (hw == ref) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }

    return 0;
}