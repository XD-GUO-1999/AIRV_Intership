#include <stdint.h>
#include <stdio.h>

// 保持你原本的纯 C 参考逻辑不变，用于对账
static int32_t reference_mac8(const uint8_t *a, const int8_t *b, int32_t init)
{
    int32_t sum = init;
    for (int i = 0; i < 8; ++i) {
        sum += (int32_t)a[i] * (int32_t)b[i];
    }
    return sum;
}

// 接收 32位指针，完全模拟 MNIST 加载权重时的块读取！
static inline int32_t mac8i_test_with_lw(const uint32_t *pa_packed, const uint32_t *pb_packed, int32_t init)
{
    int32_t sum = init;

    asm volatile(
        // 真正考验 Cache 和 lw 的时刻！
        "lw t3, 0(%[pa])\n\t"      // 隐式 t3 = input[0..3]
        "lw t4, 0(%[pb])\n\t"      // 隐式 t4 = weight[0..3]
        "lw t1, 4(%[pa])\n\t"      // 显式 t1 = input[4..7]
        "lw t2, 4(%[pb])\n\t"      // 显式 t2 = weight[4..7]

        "nop\n\t"
        "nop\n\t"
        "nop\n\t"

        // 调用你的 MAC 算子
        "mac8im %[sum], t1, t2\n\t"

        : [sum] "+r" (sum)
        : [pa] "r" (pa_packed),
          [pb] "r" (pb_packed)
        : "t1", "t2", "t3", "t4", "cc", "memory" 
    );

    return sum;
}

int main(void)
{
    // 1. 定义 32位 对齐的数组（必须在 main 内部作为局部变量）
    uint32_t input_packed[2];
    uint32_t weight_packed[2];

    // 2. 强迫 CPU 执行 Store (sw) 指令，将数据真真切切地写入 L1 D-Cache
    // 这就模拟了 MNIST 推理前，数据已经就绪在内存中的状态
    // input: {4,3,2,1} 和 {8,7,6,5}
    input_packed[0] = 0x04030201; 
    input_packed[1] = 0x08070605; 
    
    // weight: {-2,2,-1,1} 和 {-4,4,-3,3}
    weight_packed[0] = 0xFE02FF01; 
    weight_packed[1] = 0xFC04FD03; 

    int32_t init = 100;

    // 依然用你原来的 uint8_t 指针去跑一遍纯 C 的参考模型
    int32_t ref = reference_mac8((const uint8_t *)input_packed, (const int8_t *)weight_packed, init);
    
    // 跑硬件 MAC8，测试 lw 加载机制
    int32_t hw  = mac8i_test_with_lw(input_packed, weight_packed, init);

    printf("MAC8I Real Memory (lw) Test\n");
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