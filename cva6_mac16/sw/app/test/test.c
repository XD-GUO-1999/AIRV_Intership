#include <stdint.h>
#include <stdio.h>

// 1. 纯 C 参考逻辑：扩展到 16 次迭代，用于最终对账
static int32_t reference_mac16(const uint8_t *a, const int8_t *b, int32_t init)
{
    int32_t sum = init;
    for (int i = 0; i < 16; ++i) {
        sum += (int32_t)a[i] * (int32_t)b[i];
    }
    return sum;
}

// 2. MAC16 硬件测试：4个显式通道传参 + 4个隐式通道 lw 加载
static inline int32_t mac16_test_with_lw(const uint32_t *pa_packed, const uint32_t *pb_packed, int32_t init)
{
    int32_t sum = init;

    asm volatile(
        // 【隐式通道加载】把后半部分的 8 个元素动态加载到 t3 ~ t6 (x28 ~ x31)
        "lw t3, 8(%[pa])\n\t"      // 隐式 t3 = input[8..11]
        "lw t4, 12(%[pa])\n\t"     // 隐式 t4 = input[12..15]
        "lw t5, 8(%[pb])\n\t"      // 隐式 t5 = weight[8..11]
        "lw t6, 12(%[pb])\n\t"     // 隐式 t6 = weight[12..15]

        // 【超级指令发射】调用你的 4 显式魔改算子
        // %[rs1] 和 %[rs2] 传输前半部分像素，%[rs3] 和 %[rs4] 传输前半部分权重
        "mac16 %[sum], %[rs1], %[rs2], %[rs3], %[rs4]\n\t"

        : [sum] "+r" (sum)
        : [rs1] "r" (pa_packed[0]),   // 显式第一个像素块 input[0..3]
          [rs2] "r" (pa_packed[1]),   // 显式第二个像素块 input[4..7]
          [rs3] "r" (pb_packed[0]),   // 显式第一个权重块 weight[0..3]
          [rs4] "r" (pb_packed[1]),   // 显式第二个权重块 weight[4..7]
          [pa]  "r" (pa_packed),      // 传递指针基址供上面 lw 偏移使用
          [pb]  "r" (pb_packed)
        // 告知 GCC 编译器：t3, t4, t5, t6 被硬件隐式征用了，请主动避让
        : "t1", "t2", "t3", "t4", "t5", "t6", "cc", "memory" 
    );

    return sum;
}

int main(void)
{
    // 定义 32位 对齐的数组（容纳 16 个 8-bit 元素，共 4 个 Word）
    uint32_t input_packed[4];
    uint32_t weight_packed[4];

    // 强迫 CPU 执行 Store 指令，将 16 字节的数据真真切切地压入 L1 D-Cache
    // 像素阵列：1 ~ 16 (全正数 uint8_t)
    input_packed[0] = 0x04030201;  // [1, 2, 3, 4]     -> 显式 rs1
    input_packed[1] = 0x08070605;  // [5, 6, 7, 8]     -> 显式 rs2
    input_packed[2] = 0x0C0B0A09;  // [9, 10, 11, 12]  -> 隐式 t3
    input_packed[3] = 0x100F0E0D;  // [13, 14, 15, 16] -> 隐式 t4
    
    // 权重阵列：1 和 -1 (0xFF补码) 穿插交替
    weight_packed[0] = 0x01010101;  // [1, 1, 1, 1]     -> 显式 rs3
    weight_packed[1] = 0xFFFFFFFF;  // [-1, -1, -1, -1] -> 显式 rs4
    weight_packed[2] = 0x01010101;  // [1, 1, 1, 1]     -> 隐式 t5
    weight_packed[3] = 0xFFFFFFFF;  // [-1, -1, -1, -1] -> 隐式 t6

    int32_t init = 100;

    // 1. 跑纯 C 参考模型
    int32_t ref = reference_mac16((const uint8_t *)input_packed, (const int8_t *)weight_packed, init);
    
    // 2. 跑魔改硬件 MAC16
    int32_t hw  = mac16_test_with_lw(input_packed, weight_packed, init);

    // 3. 打印对账结果
    printf("====================================\n");
    printf("MAC16 Super-Instruction (9-Reg) Test\n");
    printf("====================================\n");
    printf("init = %ld\n", init);
    printf("ref  = %ld (Expected: 68)\n", ref);
    printf("hw   = %ld\n", hw);

    if (hw == ref) {
        printf("\n🎉 [SUCCESS] PASS!\n");
    } else {
        printf("\n❌ [ERROR] FAIL! Check your multiplier tree or bypass network.\n");
    }

    return 0;
}