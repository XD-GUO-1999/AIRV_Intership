#include <stdlib.h>
#include <stdio.h>

#include "env.h"
#include "mem_info.h"

#include "conv1.h"
#include "conv2.h"
#include "fc1.h"
#include "fc2.h"


static DATA_T mem[MEMORY_SIZE];

static int max(int lhs, int rhs) {
        return (lhs >= rhs)?lhs:rhs;
    }

static int clamp(int v, int lo, int hi) {
    if(v < lo) {
        return lo;
    }
    else if(v > hi) {
        return hi;
    }
    else {
        return v;
    }
}

static void macsOnRange_no_alined(const UDATA_T* __restrict inputs,
                        const WDATA_T* __restrict weights,
                        SUM_T* __restrict weightedSum,
                        int nb_iterations)
{
    int32_t sum = *weightedSum;
    int iter = 0;

    for (; iter <= nb_iterations - 16; iter += 16) {
        
        sum += inputs[iter + 0] * weights[iter + 0];
        sum += inputs[iter + 1] * weights[iter + 1];
        sum += inputs[iter + 2] * weights[iter + 2];
        sum += inputs[iter + 3] * weights[iter + 3];
        sum += inputs[iter + 24] * weights[iter + 4];
        sum += inputs[iter + 25] * weights[iter + 5];
        sum += inputs[iter + 26] * weights[iter + 6];
        sum += inputs[iter + 27] * weights[iter + 7];
        sum += inputs[iter + 48] * weights[iter + 8];
        sum += inputs[iter + 49] * weights[iter + 9];
        sum += inputs[iter + 50] * weights[iter + 10];
        sum += inputs[iter + 51] * weights[iter + 11];
        sum += inputs[iter + 72] * weights[iter + 12];
        sum += inputs[iter + 73] * weights[iter + 13];
        sum += inputs[iter + 74] * weights[iter + 14];
        sum += inputs[iter + 75] * weights[iter + 15];
    }

    for (; iter < nb_iterations; ++iter)
        {
            sum += inputs[iter] * weights[iter];
        }

     *weightedSum = sum;
}
static void macsOnRange_no_alined_for_fc2(const UDATA_T* __restrict inputs,
                        const WDATA_T* __restrict weights,
                        SUM_T* __restrict weightedSum,
                        int nb_iterations)
{
    int32_t sum = *weightedSum;
    int iter = 0;

        for (; iter <= nb_iterations - 16; iter += 16) {
            sum += inputs[iter + 0] * weights[iter + 0];
            sum += inputs[iter + 1] * weights[iter + 1];
            sum += inputs[iter + 2] * weights[iter + 2];
            sum += inputs[iter + 3] * weights[iter + 3];
            sum += inputs[iter + 4] * weights[iter + 4];
            sum += inputs[iter + 5] * weights[iter + 5];
            sum += inputs[iter + 6] * weights[iter + 6];
            sum += inputs[iter + 7] * weights[iter + 7];
            sum += inputs[iter + 8] * weights[iter + 8];
            sum += inputs[iter + 9] * weights[iter + 9];
            sum += inputs[iter + 10] * weights[iter + 10];
            sum += inputs[iter + 11] * weights[iter + 11];
            sum += inputs[iter + 12] * weights[iter + 12];
            sum += inputs[iter + 13] * weights[iter + 13];
            sum += inputs[iter + 14] * weights[iter + 14];
            sum += inputs[iter + 15] * weights[iter + 15];
            }
            for (; iter < nb_iterations; ++iter)
            {
                sum += inputs[iter] * weights[iter];
            }     
     *weightedSum = sum;
}

/*
 * ============================================================
 * 统一 input buffer 辅助函数
 * ============================================================
 *
 * 硬件侧约定：
 *   - input buffer 最大 400 byte = 100 个 32-bit word = 25 个 16-byte block。
 *   - buf4 的 rd 字段不作为真正写回寄存器，而是作为 mode：
 *
 *         active_blocks = rd + 1
 *
 *   - Conv1: rd = x0  -> active_blocks = 1
 *   - Conv2: rd = x24 -> active_blocks = 25
 *   - FC1:   rd = x23 -> active_blocks = 24
 *   - FC2:   rd = x8  -> active_blocks = 9，剩余 6 个 input 走 scalar
 *
 * 注意：
 *   t3 = x28，t4 = x29。硬件里 rs[3]/rs[4] 固定来自 x28/x29，
 *   所以这里继续用 t3/t4 是匹配的。
 */

/* Conv1 专用：把一个 4x4x1 patch buffer 进去。
 * 这里四个指针分别指向 patch 的四行，每行连续 4 个 uint8。
 * buf4 x0 表示 active_blocks = 1，所以后续每个 mac16buf 都会读 block0。
 */

 static inline void acc_init_sum(SUM_T sum)
{
    asm volatile(
        "acc_init %[sum]\n\t"
        :
        : [sum] "r" (sum)
        : "cc", "memory"
    );
}

static inline void mac16buf_acc_only(const WDATA_T* __restrict weights)
{
    const WDATA_T *p_wt = weights;

    asm volatile(
        "lw t1, 0(%[p_wt]) \n\t"
        "lw t2, 4(%[p_wt]) \n\t"
        "lw t3, 8(%[p_wt]) \n\t"
        "lw t4, 12(%[p_wt]) \n\t"
        "mac16buf x0, t1, t2, t3, t4 \n\t"
        :
        : [p_wt] "r" (p_wt)
        : "t1", "t2", "t3", "t4", "cc", "memory"
    );
}

static inline SUM_T acc_get_sum(void)
{
    SUM_T sum;

    asm volatile(
        "acc_get %[sum]\n\t"
        : [sum] "=r" (sum)
        :
        : "cc", "memory"
    );

    return sum;
}

static inline void buffer4_conv1_patch(
    const UDATA_T* __restrict row0,
    const UDATA_T* __restrict row1,
    const UDATA_T* __restrict row2,
    const UDATA_T* __restrict row3)
{
    asm volatile(
        "lw t1, 0(%[row0]) \n\t"
        "lw t2, 0(%[row1]) \n\t"
        "lw t3, 0(%[row2]) \n\t"
        "lw t4, 0(%[row3]) \n\t"
        "buf4 x0, t1, t2, t3, t4 \n\t"
        :
        : [row0] "r" (row0),
          [row1] "r" (row1),
          [row2] "r" (row2),
          [row3] "r" (row3)
        : "t1", "t2", "t3", "t4", "cc", "memory"
    );
}

/* Conv2 专用：把一个 pixel 的 16 个 channels buffer 进去。
 * Conv2 一个完整 5x5x16 patch 需要连续执行 25 次这个函数。
 * buf4 x24 表示 active_blocks = 25。
 */
static inline void buffer4_contiguous16_conv2(const UDATA_T* __restrict inputs)
{
    const UDATA_T *p_in = inputs;
    asm volatile(
        "lw t1, 0(%[p_in]) \n\t"
        "lw t2, 4(%[p_in]) \n\t"
        "lw t3, 8(%[p_in]) \n\t"
        "lw t4, 12(%[p_in]) \n\t"
        "buf4 x24, t1, t2, t3, t4 \n\t"
        :
        : [p_in] "r" (p_in)
        : "t1", "t2", "t3", "t4", "cc", "memory"
    );
}

/* FC1 专用：FC1 input = 384 byte = 24 个 16-byte block。
 * buf4 x23 表示 active_blocks = 24。
 */
static inline void buffer4_contiguous16_fc1(const UDATA_T* __restrict inputs)
{
    const UDATA_T *p_in = inputs;
    asm volatile(
        "lw t1, 0(%[p_in]) \n\t"
        "lw t2, 4(%[p_in]) \n\t"
        "lw t3, 8(%[p_in]) \n\t"
        "lw t4, 12(%[p_in]) \n\t"
        "buf4 x23, t1, t2, t3, t4 \n\t"
        :
        : [p_in] "r" (p_in)
        : "t1", "t2", "t3", "t4", "cc", "memory"
    );
}

/* FC2 专用：FC2 input = 150 byte。
 * 前 144 byte = 9 个 16-byte block 用 buffer + mac16buf，
 * 最后 6 byte 用普通 scalar 处理。
 * buf4 x8 表示 active_blocks = 9。
 */
static inline void buffer4_contiguous16_fc2(const UDATA_T* __restrict inputs)
{
    const UDATA_T *p_in = inputs;
    asm volatile(
        "lw t1, 0(%[p_in]) \n\t"
        "lw t2, 4(%[p_in]) \n\t"
        "lw t3, 8(%[p_in]) \n\t"
        "lw t4, 12(%[p_in]) \n\t"
        "buf4 x8, t1, t2, t3, t4 \n\t"
        :
        : [p_in] "r" (p_in)
        : "t1", "t2", "t3", "t4", "cc", "memory"
    );
}

/* 只执行 MAC，不更新 input buffer。
 * 这个函数假设：对应的 input block 已经在硬件 input buffer 中。
 * 硬件每执行一次 mac16buf，会根据 active_blocks 自动移动 read counter。
 */
static inline void mac16buf_only(const WDATA_T* __restrict weights,
                                 SUM_T* __restrict weightedSum)
{
    int32_t sum = *weightedSum;
    const WDATA_T *p_wt = weights;

    asm volatile(
        "lw t1, 0(%[p_wt]) \n\t"
        "lw t2, 4(%[p_wt]) \n\t"
        "lw t3, 8(%[p_wt]) \n\t"
        "lw t4, 12(%[p_wt]) \n\t"
        "mac16buf %[sum], t1, t2, t3, t4 \n\t"
        : [sum] "+r" (sum)
        : [p_wt] "r" (p_wt)
        : "t1", "t2", "t3", "t4", "cc", "memory"
    );

    *weightedSum = sum;
}

static void macsOnRange(const UDATA_T* __restrict inputs,
                        const WDATA_T* __restrict weights,
                        SUM_T* __restrict weightedSum,
                        int nb_iterations)
{
    for (int iter = 0; iter < nb_iterations; ++iter) {
        *weightedSum += inputs[iter] * weights[iter];
    }
}

static UDATA_T saturate(SUM_T value, uint32_t sat) {
    return clamp(value, (SUM_T)(0), ((SUM_T)(1) << sat) - 1);
}

static UDATA_T sat(SUM_T weightedSum, int output,
                                           ActivationFunction_T func,
                                           /* const Rescaling_T& __restrict rescaling */
                                           int shift)
{
    switch(func) {
        case Linear:
        case Saturation: {
            break;
        }
        case Rectifier: {
            if(weightedSum <= 0) weightedSum = 0;
            break;
        }
        default:
            printf("Unsupported activation function.\n");
            break;
    }

    return saturate(weightedSum>>shift, NB_BITS);
}

static void convcellPropagate1(
    const UDATA_T* __restrict inputs,
    UDATA_T* __restrict outputs,
    const BDATA_T* __restrict biasses,
    const WDATA_T* __restrict weights,
    int rescaling,
    int NB_CHANNELS, 
    int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
    int NB_OUTPUTS,
    int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
    int PADDING_Y, int PADDING_X,
    int STRIDE_Y, int STRIDE_X,
    int KERNEL_HEIGHT, int KERNEL_WIDTH,
    ActivationFunction_T ACTIVATION,
    // Memory mapping: inputs
    int INPUT_MEM_CONT_OFFSET,
    int INPUT_MEM_CONT_SIZE,
    int INPUT_MEM_WRAP_OFFSET,
    int INPUT_MEM_WRAP_SIZE,
    int INPUT_MEM_STRIDE,
    // Memory mapping: outputs
    int OUTPUT_MEM_CONT_OFFSET,
    int OUTPUT_MEM_CONT_SIZE,
    int OUTPUT_MEM_WRAP_OFFSET,
    int OUTPUT_MEM_WRAP_SIZE,
    int OUTPUT_MEM_STRIDE)
{
    int OUTPUTS_HEIGHT_NOPAD
        = (CHANNELS_HEIGHT - KERNEL_HEIGHT + STRIDE_Y) / STRIDE_Y;
    int OUTPUTS_WIDTH_NOPAD
        = (CHANNELS_WIDTH - KERNEL_WIDTH + STRIDE_X) / STRIDE_X;

    /*
     * Conv1 的 buffer 策略：
     *   - Conv1 kernel = 4x4x1 = 16 byte = 1 个 block。
     *   - 对同一个 output pixel (ox, oy)，input patch 对所有 output filters 都一样。
     *   - 所以 buffer4 应该放在 output 循环外面：一个 patch 只 buffer 一次。
     *   - 后续每个 output filter 只换 weight，然后执行 mac16buf。
     */

    for (int oy = 0; oy < OUTPUTS_HEIGHT; ++oy) {
        const int syMin = (PADDING_Y == 0) ? 0
            : max(PADDING_Y - (oy * STRIDE_Y), 0);
        const int syMax = (PADDING_Y == 0
                && OUTPUTS_HEIGHT == OUTPUTS_HEIGHT_NOPAD) ? KERNEL_HEIGHT
            : clamp(CHANNELS_HEIGHT + PADDING_Y - (oy * STRIDE_Y),
                    0, KERNEL_HEIGHT);
        const int iy = (oy * STRIDE_Y) - PADDING_Y;

        for (int ox = 0; ox < OUTPUTS_WIDTH; ++ox) {
            const int sxMin = (PADDING_X == 0) ? 0
                : max(PADDING_X - (ox * STRIDE_X), 0);
            const int sxMax = (PADDING_X == 0
                    && OUTPUTS_WIDTH == OUTPUTS_WIDTH_NOPAD)
                        ? KERNEL_WIDTH
                : clamp(CHANNELS_WIDTH + PADDING_X - (ox * STRIDE_X),
                        0, KERNEL_WIDTH);
            const int ix = (ox * STRIDE_X) - PADDING_X;

            const int oPos = (ox + OUTPUTS_WIDTH * oy);
            int oOffset = OUTPUT_MEM_STRIDE * oPos;

            if (OUTPUT_MEM_WRAP_SIZE > 0 && oOffset >= OUTPUT_MEM_CONT_SIZE) {
                oOffset += OUTPUT_MEM_WRAP_OFFSET - OUTPUT_MEM_CONT_OFFSET
                            - OUTPUT_MEM_CONT_SIZE;
            }

            /*
             * 只有完整 4x4x1 patch 且地址 4-byte 对齐时，才走 input buffer。
             * 如果遇到 padding、wrap 或 unaligned 地址，就回退到原始 scalar 逻辑。
             */
            bool patch_ok = ((ox & 1) == 0);
            const UDATA_T* row0 = 0;
            const UDATA_T* row1 = 0;
            const UDATA_T* row2 = 0;
            const UDATA_T* row3 = 0;

            if (KERNEL_HEIGHT == 4 && KERNEL_WIDTH == 4 && NB_CHANNELS == 1
                && (sxMax - sxMin) == KERNEL_WIDTH
                && (syMax - syMin) == KERNEL_HEIGHT)
            {
                const int iPos0 = ((sxMin + ix)
                                + CHANNELS_WIDTH * (iy + syMin));
                int iOffset0 = INPUT_MEM_STRIDE * iPos0;

                if (INPUT_MEM_WRAP_SIZE > 0 && iOffset0 >= INPUT_MEM_CONT_SIZE) {
                    iOffset0 += INPUT_MEM_WRAP_OFFSET - INPUT_MEM_CONT_OFFSET
                              - INPUT_MEM_CONT_SIZE;
                }

                const int row_stride = CHANNELS_WIDTH * INPUT_MEM_STRIDE;

                row0 = inputs + iOffset0;
                row1 = inputs + iOffset0 + row_stride;
                row2 = inputs + iOffset0 + 2 * row_stride;
                row3 = inputs + iOffset0 + 3 * row_stride;
            }

            if (patch_ok) {
                // buf4 x0: active_blocks = 1。这个 patch 会被所有 filters 复用。
                buffer4_conv1_patch(row0, row1, row2, row3);
            }

            for (int output = 0; output < NB_OUTPUTS; ++output) {
                SUM_T weightedSum = biasses[output];

                if (patch_ok) {
                    /*
                     * Conv1 一个 filter 的 16 个 weight 连续存放。
                     * input 已经在 buffer 的 block0 里，所以这里只需要换 weight。
                     */
                    const int wOffset = NB_CHANNELS * (sxMin
                        + KERNEL_WIDTH * (syMin + KERNEL_HEIGHT * output));

                    //mac16buf_only(weights + wOffset, &weightedSum);
                    acc_init_sum(weightedSum);
                    mac16buf_acc_only(weights + wOffset);
                    weightedSum = acc_get_sum();
                }
                else {
                    /*
                    * 安全回退路径：保持原始卷积语义。
                    * 但这里不再一个 sx 一个 sx 地处理，而是优先一整行一整行处理。
                    *
                    * 对 Conv1 来说：
                    *   KERNEL_WIDTH = 4
                    *   NB_CHANNELS = 1
                    *   所以一行 = 4 个 input = 4 个 MAC
                    */
                    for (int sy = 0; sy <= KERNEL_HEIGHT - 4; sy += 4) {
                        if ((PADDING_Y != 0 || OUTPUTS_HEIGHT != OUTPUTS_HEIGHT_NOPAD)
                            && sy >= syMax - syMin)
                        {
                            break;
                        }

                        /*
                        * 当前 kernel row 对应的 input 起始位置。
                        * 注意这里从 sxMin 开始，所以如果有 padding，左边无效部分会被跳过。
                        */
                        const int iPos = ((sxMin + ix)
                                        + CHANNELS_WIDTH * (iy + syMin + sy));
                        int iOffset = INPUT_MEM_STRIDE * iPos;

                        bool wrapInRange = false;

                        if (INPUT_MEM_WRAP_SIZE > 0 && iOffset >= INPUT_MEM_CONT_SIZE) {
                            iOffset += INPUT_MEM_WRAP_OFFSET
                                    - INPUT_MEM_CONT_OFFSET
                                    - INPUT_MEM_CONT_SIZE;
                        }
                        else if (INPUT_MEM_WRAP_SIZE > 0
                            && KERNEL_WIDTH > 1
                            && CHANNELS_HEIGHT == 1
                            && iOffset + KERNEL_WIDTH * NB_CHANNELS > INPUT_MEM_CONT_SIZE)
                        {
                            wrapInRange = true;
                        }

                        /*
                        * 当前 kernel row 对应的 weight 起始位置。
                        * weight layout:
                        *   [output][sy][sx][channel]
                        */
                        const int wOffset = NB_CHANNELS * (sxMin
                            + KERNEL_WIDTH * (syMin + sy + KERNEL_HEIGHT * output));

                        /*
                        * 如果 input memory 中这一行是连续的，就整行处理。
                        * Conv1 中 NB_CHANNELS = INPUT_MEM_STRIDE = 1，
                        * 所以正常情况下这里会成立。
                        */
                        if (!wrapInRange && NB_CHANNELS == INPUT_MEM_STRIDE) {
                            macsOnRange_no_alined(
                                inputs + iOffset,
                                weights + wOffset,
                                &weightedSum,
                                KERNEL_WIDTH * NB_CHANNELS * 4
                            );
                        }
                        else {
                            /*
                            * 极端 fallback：
                            * 如果这一行发生 wrap，或者 memory stride 不连续，
                            * 才退回到 sx-by-sx。
                            */
                            for (int sx = 0; sx < KERNEL_WIDTH; ++sx) {
                                if ((PADDING_X != 0 || OUTPUTS_WIDTH != OUTPUTS_WIDTH_NOPAD)
                                    && sx >= sxMax - sxMin)
                                {
                                    break;
                                }

                                int iOffsetInRange = iOffset + sx * INPUT_MEM_STRIDE;

                                if (wrapInRange && iOffsetInRange >= INPUT_MEM_CONT_SIZE) {
                                    iOffsetInRange += INPUT_MEM_WRAP_OFFSET
                                                    - INPUT_MEM_CONT_OFFSET
                                                    - INPUT_MEM_CONT_SIZE;
                                }

                                macsOnRange(
                                    inputs + iOffsetInRange,
                                    weights + wOffset + sx * NB_CHANNELS,
                                    &weightedSum,
                                    NB_CHANNELS
                                );
                            }
                        }
                    }
                }

                outputs[oOffset + output]
                    = sat(weightedSum, output, ACTIVATION, rescaling);
            }
        }
    }
}


static void convcellPropagate2(
    const UDATA_T* __restrict inputs,
    UDATA_T* __restrict outputs,
    const BDATA_T* __restrict biasses,
    const WDATA_T* __restrict weights,
    int rescaling,
    int NB_CHANNELS, 
    int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
    int NB_OUTPUTS,
    int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
    int PADDING_Y, int PADDING_X,
    int STRIDE_Y, int STRIDE_X,
    int KERNEL_HEIGHT, int KERNEL_WIDTH,
    ActivationFunction_T ACTIVATION,
    // Memory mapping: inputs
    int INPUT_MEM_CONT_OFFSET,
    int INPUT_MEM_CONT_SIZE,
    int INPUT_MEM_WRAP_OFFSET,
    int INPUT_MEM_WRAP_SIZE,
    int INPUT_MEM_STRIDE,
    // Memory mapping: outputs
    int OUTPUT_MEM_CONT_OFFSET,
    int OUTPUT_MEM_CONT_SIZE,
    int OUTPUT_MEM_WRAP_OFFSET,
    int OUTPUT_MEM_WRAP_SIZE,
    int OUTPUT_MEM_STRIDE)
{
    int OUTPUTS_HEIGHT_NOPAD
        = (CHANNELS_HEIGHT - KERNEL_HEIGHT + STRIDE_Y) / STRIDE_Y;
    int OUTPUTS_WIDTH_NOPAD
        = (CHANNELS_WIDTH - KERNEL_WIDTH + STRIDE_X) / STRIDE_X;

    /*
     * Conv2 的 buffer 策略：
     *   - Conv2 kernel = 5x5x16 = 400 byte = 25 个 16-byte block。
     *   - 对同一个 output pixel (ox, oy)，完整 400-byte input patch
     *     对 24 个 output filters 都相同。
     *   - 所以先连续执行 25 次 buf4 x24，把完整 input patch 存进硬件 buffer。
     *   - 然后每个 output filter 执行 25 次 mac16buf。
     *   - 硬件 active_blocks=25，所以 25 次 mac16buf 后 read counter 自动回到 0。
     */

    for (int oy = 0; oy < OUTPUTS_HEIGHT; ++oy) {
        const int syMin = (PADDING_Y == 0) ? 0
            : max(PADDING_Y - (oy * STRIDE_Y), 0);
        const int syMax = (PADDING_Y == 0
                && OUTPUTS_HEIGHT == OUTPUTS_HEIGHT_NOPAD) ? KERNEL_HEIGHT
            : clamp(CHANNELS_HEIGHT + PADDING_Y - (oy * STRIDE_Y),
                    0, KERNEL_HEIGHT);
        const int iy = (oy * STRIDE_Y) - PADDING_Y;

        for (int ox = 0; ox < OUTPUTS_WIDTH; ++ox) {
            const int sxMin = (PADDING_X == 0) ? 0
                : max(PADDING_X - (ox * STRIDE_X), 0);
            const int sxMax = (PADDING_X == 0
                    && OUTPUTS_WIDTH == OUTPUTS_WIDTH_NOPAD)
                        ? KERNEL_WIDTH
                : clamp(CHANNELS_WIDTH + PADDING_X - (ox * STRIDE_X),
                        0, KERNEL_WIDTH);
            const int ix = (ox * STRIDE_X) - PADDING_X;

            const int oPos = (ox + OUTPUTS_WIDTH * oy);
            int oOffset = OUTPUT_MEM_STRIDE * oPos;

            if (OUTPUT_MEM_WRAP_SIZE > 0 && oOffset >= OUTPUT_MEM_CONT_SIZE) {
                oOffset += OUTPUT_MEM_WRAP_OFFSET - OUTPUT_MEM_CONT_OFFSET
                            - OUTPUT_MEM_CONT_SIZE;
            }

            /*
             * 先检查这个 5x5x16 patch 是否能安全用 buffer：
             *   1. 每个 pixel 的 16 channels 连续；
             *   2. 每个 16-byte block 4-byte 对齐；
             *   3. 没有跨 memory wrap 边界；
             *   4. 当前 patch 没有被 padding 截断。
             *
             * 检查通过后才真正执行 buf4，避免写了一半 buffer 又 fallback，
             * 导致硬件 write counter 错位。
             */
                for (int sy = 0; sy < KERNEL_HEIGHT; ++sy) {
                    for (int sx = 0; sx < KERNEL_WIDTH; ++sx) {
                        const int iPos = ((sxMin + sx + ix)
                                        + CHANNELS_WIDTH * (iy + syMin + sy));
                        int iOffset = INPUT_MEM_STRIDE * iPos;

                        if (INPUT_MEM_WRAP_SIZE > 0
                            && iOffset + NB_CHANNELS > INPUT_MEM_CONT_SIZE)
                        {
                            break;
                        }

                        if (INPUT_MEM_WRAP_SIZE > 0 && iOffset >= INPUT_MEM_CONT_SIZE) {
                            iOffset += INPUT_MEM_WRAP_OFFSET - INPUT_MEM_CONT_OFFSET
                                      - INPUT_MEM_CONT_SIZE;
                        }

                        const UDATA_T* in_ptr = inputs + iOffset;
                    }
                }

                /*
                 * 填满完整 Conv2 patch。
                 * 连续 25 次 buf4 x24：
                 *   block0, block1, ..., block24
                 * 写完后硬件 write counter 自动回到 0。
                 */
                for (int sy = 0; sy < KERNEL_HEIGHT; ++sy) {
                    for (int sx = 0; sx < KERNEL_WIDTH; ++sx) {
                        const int iPos = ((sxMin + sx + ix)
                                        + CHANNELS_WIDTH * (iy + syMin + sy));
                        int iOffset = INPUT_MEM_STRIDE * iPos;

                        if (INPUT_MEM_WRAP_SIZE > 0 && iOffset >= INPUT_MEM_CONT_SIZE) {
                            iOffset += INPUT_MEM_WRAP_OFFSET - INPUT_MEM_CONT_OFFSET
                                      - INPUT_MEM_CONT_SIZE;
                        }

                        buffer4_contiguous16_conv2(inputs + iOffset);
                    }
                }
            

            for (int output = 0; output < NB_OUTPUTS; ++output) {
                SUM_T weightedSum = biasses[output];

                acc_init_sum(weightedSum);
    
                    /*
                     * input patch 已经在硬件 buffer 中。
                     * 对当前 output filter，只需要按 sy/sx 顺序读取对应 weight block。
                     * 25 次 mac16buf 后，硬件 read counter 自动回到 0，
                     * 所以下一个 output filter 会重新从 block0 开始读。
                     */
                    for (int sy = 0; sy < KERNEL_HEIGHT; ++sy) {
                        for (int sx = 0; sx < KERNEL_WIDTH; ++sx) {
                            const int wOffset = NB_CHANNELS * (sxMin + sx
                                + KERNEL_WIDTH * (syMin + sy + KERNEL_HEIGHT * output));

                            mac16buf_acc_only(weights + wOffset);
                        }
                    }
                weightedSum = acc_get_sum();

                outputs[oOffset + output]
                    = sat(weightedSum, output, ACTIVATION, rescaling);
            }
        }
    }
}


static void fccellPropagateUDATA_T(
    const UDATA_T* __restrict inputs,
    UDATA_T* __restrict outputs,
    const BDATA_T* __restrict biasses,
    const WDATA_T* __restrict weights,
    const int rescaling,
    int NB_CHANNELS, 
    int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
    int NB_OUTPUTS,
    int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
    ActivationFunction_T ACTIVATION,
    // Memory mapping: inputs
    int INPUT_MEM_CONT_OFFSET,
    int INPUT_MEM_CONT_SIZE,
    int INPUT_MEM_WRAP_OFFSET,
    int INPUT_MEM_WRAP_SIZE,
    int INPUT_MEM_STRIDE,
    // Memory mapping: outputs
    int OUTPUT_MEM_CONT_OFFSET,
    int OUTPUT_MEM_CONT_SIZE,
    int OUTPUT_MEM_WRAP_OFFSET,
    int OUTPUT_MEM_WRAP_SIZE,
    int OUTPUT_MEM_STRIDE)
{
    /*
     * FC1 的 buffer 策略：
     *   - FC1 input = Conv2 output = 24x4x4 = 384 byte。
     *   - 384 byte = 24 个 16-byte block。
     *   - 先执行 24 次 buf4 x23，把整个 FC1 input 存进硬件 input buffer。
     *   - 然后每个 output neuron 执行 24 次 mac16buf。
     *   - 硬件 active_blocks=24，所以每个 neuron 算完后 read counter 自动回到 0。
     *
     * 如果输入布局不满足连续/对齐要求，就走原始 scalar fallback。
     */
    const int total_inputs = NB_CHANNELS * CHANNELS_WIDTH * CHANNELS_HEIGHT;


        // 一次性把 FC1 的 384-byte input 全部 buffer 进去。
        for (int block = 0; block < 24; ++block) {
            buffer4_contiguous16_fc1(inputs + block * 16);
        }
        
        for (int och = 0; och < NB_OUTPUTS; och++) {
            SUM_T weightedSum = biasses[och];

            acc_init_sum(weightedSum);

            const int wBase = och * total_inputs;

            for (int block = 0; block < 24; ++block) {
                mac16buf_acc_only(weights + wBase + block * 16);
            }

            weightedSum = acc_get_sum();
            outputs[och] = sat(weightedSum, och, ACTIVATION, rescaling);
        }

        return;


    /*
     * 原始 fallback：
     * 用于非 FC1、非连续布局、或不对齐情况。
     */
}

static void fccellPropagateDATA_T(
    const UDATA_T* __restrict inputs,
    DATA_T* __restrict outputs,
    const BDATA_T* __restrict biasses,
    const WDATA_T* __restrict weights,
    const int rescaling,
    int NB_CHANNELS, 
    int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
    int NB_OUTPUTS,
    int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
    ActivationFunction_T ACTIVATION,
    // Memory mapping: inputs
    int INPUT_MEM_CONT_OFFSET,
    int INPUT_MEM_CONT_SIZE,
    int INPUT_MEM_WRAP_OFFSET,
    int INPUT_MEM_WRAP_SIZE,
    int INPUT_MEM_STRIDE,
    // Memory mapping: outputs
    int OUTPUT_MEM_CONT_OFFSET,
    int OUTPUT_MEM_CONT_SIZE,
    int OUTPUT_MEM_WRAP_OFFSET,
    int OUTPUT_MEM_WRAP_SIZE,
    int OUTPUT_MEM_STRIDE)
{
    /*
     * FC2 的 buffer 策略：
     *   - FC2 input = 150 byte。
     *   - 前 144 byte = 9 个 16-byte block，用 buf4 x8 + mac16buf。
     *   - 最后 6 byte 不是完整 16-byte block，先保留 scalar 处理。
     *   - 每个 output class 执行 9 次 mac16buf 后，硬件 read counter 自动回到 0。
     */
    const int total_inputs = NB_CHANNELS * CHANNELS_WIDTH * CHANNELS_HEIGHT;


    // 只 buffer 前 144 个 input，剩下 6 个 input 在每个 output 中 scalar 累加。
    for (int block = 0; block < 9; ++block) {
        buffer4_contiguous16_fc2(inputs + block * 16);
    }

    for (int och = 0; och < NB_OUTPUTS; och++) {
        SUM_T weightedSum = biasses[och];
        
        acc_init_sum(weightedSum);

        const int wBase = och * total_inputs;
        const WDATA_T* weight_ptr = weights + wBase;
        const bool weight_aligned = (((uintptr_t)weight_ptr & 0x3) == 0);

        if (weight_aligned) {

            for (int block = 0; block < 9; ++block) {
                mac16buf_acc_only(weights + wBase + block * 16);
            }

            weightedSum = acc_get_sum();
            
            // FC2 剩余 6 个 input：index 144..149。
            for (int i = 144; i < total_inputs; ++i) {
                weightedSum += inputs[i] * weights[wBase + i];
            }
        }else{
            macsOnRange_no_alined_for_fc2(
                inputs, 
                weights + wBase, 
                &weightedSum, total_inputs);
        }
        outputs[och] = sat(weightedSum, och, ACTIVATION, rescaling);
    }

    return;
}


//     /*
//      * 原始 fallback：
//      * 用于非 FC2、非连续布局、或不对齐情况。
//      */
//     for (int och = 0; och < NB_OUTPUTS; och++) {
//         SUM_T weightedSum = biasses[och];

//         for (int iy = 0; iy < CHANNELS_HEIGHT; ++iy) {
//             const int iPos = (CHANNELS_WIDTH * iy);
//             int iOffset = INPUT_MEM_STRIDE * iPos;

//             bool wrapInRange = false;

//             if (INPUT_MEM_WRAP_SIZE > 0 && iOffset >= INPUT_MEM_CONT_SIZE) {
//                 iOffset += INPUT_MEM_WRAP_OFFSET - INPUT_MEM_CONT_OFFSET
//                             - INPUT_MEM_CONT_SIZE;
//             }
//             else if (INPUT_MEM_WRAP_SIZE > 0 && CHANNELS_WIDTH > 1
//                 && CHANNELS_HEIGHT == 1
//                 && iOffset + CHANNELS_WIDTH * NB_CHANNELS
//                     > INPUT_MEM_CONT_SIZE)
//             {
//                 wrapInRange = true;
//             }

//             const int wOffset = NB_CHANNELS * CHANNELS_WIDTH
//                                     * (iy + CHANNELS_HEIGHT * och);

//             if (!wrapInRange && INPUT_MEM_STRIDE == NB_CHANNELS) {
//                 macsOnRange(
//                     inputs + iOffset, 
//                     weights + wOffset, 
//                     &weightedSum, NB_CHANNELS * CHANNELS_WIDTH);
//             }
//             else {
//                 for (int ix = 0; ix < CHANNELS_WIDTH; ++ix) {
//                     int iOffsetInRange = iOffset + ix * INPUT_MEM_STRIDE;

//                     if (wrapInRange
//                         && iOffsetInRange >= INPUT_MEM_CONT_SIZE)
//                     {
//                         iOffsetInRange += INPUT_MEM_WRAP_OFFSET
//                                     - INPUT_MEM_CONT_OFFSET
//                                     - INPUT_MEM_CONT_SIZE;
//                     }

//                     macsOnRange(
//                         inputs + iOffsetInRange, 
//                         weights + wOffset + ix * NB_CHANNELS, 
//                         &weightedSum, NB_CHANNELS);
//                 }
//             }
//         }

//         outputs[och] = sat(weightedSum, och, ACTIVATION, rescaling);
//     }
// }

static void maxPropagate1(
    const DATA_T* __restrict inputs,
    int32_t* __restrict outputs,
    DATA_T* output_value,
    int NB_CHANNELS,
    int INPUTS_HEIGHT, int INPUTS_WIDTH,
    // Memory mapping: outputs
    int INPUT_MEM_CONT_OFFSET,
    int INPUT_MEM_CONT_SIZE,
    int INPUT_MEM_WRAP_OFFSET,
    int INPUT_MEM_WRAP_SIZE,
    int INPUT_MEM_STRIDE)
{
    int iMaxInput = 0;
    DATA_T maxInput = SCHAR_MIN;

    for (int iy = 0; iy < INPUTS_HEIGHT; ++iy) {
        for (int ix = 0; ix < INPUTS_WIDTH; ++ix) {
            const int oPos = (ix + INPUTS_WIDTH * iy);
            int iOffset = INPUT_MEM_STRIDE * oPos;

            if (INPUT_MEM_WRAP_SIZE > 0 && iOffset >= INPUT_MEM_CONT_SIZE) {
                iOffset += INPUT_MEM_WRAP_OFFSET - INPUT_MEM_CONT_OFFSET
                            - INPUT_MEM_CONT_SIZE;
            }

            if (NB_CHANNELS > 1) {
                for (int ch = 0; ch < NB_CHANNELS; ++ch) {
                    if (inputs[iOffset + ch] > maxInput) {
                        iMaxInput = ch;
                        maxInput = inputs[iOffset + ch];
                    }
                }

                outputs[oPos] = (int32_t)(iMaxInput);
		*output_value = maxInput;
            }
            else {
                outputs[oPos] = (inputs[iOffset] > 0);
		output_value = inputs[iOffset];
            }
        }
    }
}

void propagate(const UDATA_T* inputs, Target_T* outputs, UDATA_T* maxPropagate_val)
{
#ifdef SAVE_OUTPUTS
    FILE* env_stream = fopen("env_output.txt", "w");
    saveOutputs(ENV_NB_OUTPUTS, ENV_SIZE_Y, ENV_SIZE_X, ENV_MEM_CONT_OFFSET, ENV_MEM_CONT_SIZE, ENV_MEM_WRAP_OFFSET, ENV_MEM_WRAP_SIZE, ENV_MEM_STRIDE, inputs, env_stream, Network::Format::CHW);
    fclose(env_stream);
#endif
    // conv1
    UDATA_T* conv1_output = (UDATA_T*) mem + CONV1_MEM_CONT_OFFSET;

#ifdef BENCHMARK
    const Tick_T start_conv1 = tick();
#endif

    convcellPropagate1(inputs , conv1_output, conv1_biases, conv1_weights, 8,
    CONV1_NB_CHANNELS, CONV1_CHANNELS_HEIGHT, CONV1_CHANNELS_WIDTH, CONV1_NB_OUTPUTS, CONV1_OUTPUTS_HEIGHT, 
    CONV1_OUTPUTS_WIDTH, CONV1_PADDING_Y, CONV1_PADDING_X, CONV1_STRIDE_Y, CONV1_STRIDE_X, CONV1_KERNEL_HEIGHT, 
    CONV1_KERNEL_WIDTH, CONV1_ACTIVATION, ENV_MEM_CONT_OFFSET, ENV_MEM_CONT_SIZE, ENV_MEM_WRAP_OFFSET, 
    ENV_MEM_WRAP_SIZE, ENV_MEM_STRIDE, CONV1_MEM_CONT_OFFSET, CONV1_MEM_CONT_SIZE, CONV1_MEM_WRAP_OFFSET, CONV1_MEM_WRAP_SIZE, CONV1_MEM_STRIDE);

    //convcellPropagate1(inputs , conv1_output, conv1_biases, conv1_weights, CONV1_SCALING);

#ifdef BENCHMARK
    const Tick_T end_conv1 = tick();
    static RunningMean_T conv1_timing = {0.0, 0};
    benchmark("conv1", start_conv1, end_conv1, conv1_timing);
#endif

#ifdef SAVE_OUTPUTS
    FILE* conv1_stream = fopen("conv1_output.txt", "w");
    saveOutputs(CONV1_NB_OUTPUTS, CONV1_OUTPUTS_HEIGHT, CONV1_OUTPUTS_WIDTH, CONV1_MEM_CONT_OFFSET, CONV1_MEM_CONT_SIZE, CONV1_MEM_WRAP_OFFSET, CONV1_MEM_WRAP_SIZE, CONV1_MEM_STRIDE, conv1_output , conv1_stream, Network::Format::CHW);
    fclose(conv1_stream);
#endif




    // conv2
    UDATA_T* conv2_output = (UDATA_T*) mem + CONV2_MEM_CONT_OFFSET;

#ifdef BENCHMARK
    const Tick_T start_conv2 = tick();
#endif

    convcellPropagate2(conv1_output , conv2_output, conv2_biases, conv2_weights, 8,
    CONV2_NB_CHANNELS, CONV2_CHANNELS_HEIGHT, CONV2_CHANNELS_WIDTH, 
    CONV2_NB_OUTPUTS, CONV2_OUTPUTS_HEIGHT, CONV2_OUTPUTS_WIDTH, 
    CONV2_PADDING_Y, CONV2_PADDING_X, CONV2_STRIDE_Y, CONV2_STRIDE_X, 
    CONV2_KERNEL_HEIGHT, CONV2_KERNEL_WIDTH, CONV2_ACTIVATION, CONV1_MEM_CONT_OFFSET, 
    CONV1_MEM_CONT_SIZE, CONV1_MEM_WRAP_OFFSET, CONV1_MEM_WRAP_SIZE, 
    CONV1_MEM_STRIDE, CONV2_MEM_CONT_OFFSET, CONV2_MEM_CONT_SIZE, CONV2_MEM_WRAP_OFFSET, 
    CONV2_MEM_WRAP_SIZE, CONV2_MEM_STRIDE);

    //convcellPropagate2(conv1_output , conv2_output, conv2_biases, conv2_weights, CONV2_SCALING);

#ifdef BENCHMARK
    const Tick_T end_conv2 = tick();
    static RunningMean_T conv2_timing = {0.0, 0};
    benchmark("conv2", start_conv2, end_conv2, conv2_timing);
#endif

#ifdef SAVE_OUTPUTS
    FILE* conv2_stream = fopen("conv2_output.txt", "w");
    saveOutputs(CONV2_NB_OUTPUTS, CONV2_OUTPUTS_HEIGHT, CONV2_OUTPUTS_WIDTH, CONV2_MEM_CONT_OFFSET, CONV2_MEM_CONT_SIZE, CONV2_MEM_WRAP_OFFSET, CONV2_MEM_WRAP_SIZE, CONV2_MEM_STRIDE, conv2_output , conv2_stream, Network::Format::CHW);
    fclose(conv2_stream);
#endif




    // fc1
    UDATA_T* fc1_output = (UDATA_T*) mem + FC1_MEM_CONT_OFFSET;

#ifdef BENCHMARK
    const Tick_T start_fc1 = tick();
#endif

    fccellPropagateUDATA_T(conv2_output , fc1_output, fc1_biases, fc1_weights, 8,
    FC1_NB_CHANNELS, FC1_CHANNELS_HEIGHT, 
    FC1_CHANNELS_WIDTH, FC1_NB_OUTPUTS, 
    FC1_OUTPUTS_HEIGHT, FC1_OUTPUTS_WIDTH, FC1_ACTIVATION, 
    CONV2_MEM_CONT_OFFSET, CONV2_MEM_CONT_SIZE, 
    CONV2_MEM_WRAP_OFFSET, CONV2_MEM_WRAP_SIZE, 
    CONV2_MEM_STRIDE, FC1_MEM_CONT_OFFSET, 
    FC1_MEM_CONT_SIZE, FC1_MEM_WRAP_OFFSET, FC1_MEM_WRAP_SIZE, FC1_MEM_STRIDE);

#ifdef BENCHMARK
    const Tick_T end_fc1 = tick();
    static RunningMean_T fc1_timing = {0.0, 0};
    benchmark("fc1", start_fc1, end_fc1, fc1_timing);
#endif

#ifdef SAVE_OUTPUTS
    FILE* fc1_stream = fopen("fc1_output.txt", "w");
    saveOutputs(FC1_NB_OUTPUTS, FC1_OUTPUTS_HEIGHT, FC1_OUTPUTS_WIDTH, FC1_MEM_CONT_OFFSET, FC1_MEM_CONT_SIZE, FC1_MEM_WRAP_OFFSET, FC1_MEM_WRAP_SIZE, FC1_MEM_STRIDE, fc1_output , fc1_stream, Network::Format::CHW);
    fclose(fc1_stream);
#endif




    // fc2
    DATA_T* fc2_output = (DATA_T*) mem + FC2_MEM_CONT_OFFSET;

#ifdef BENCHMARK
    const Tick_T start_fc2 = tick();
#endif

    fccellPropagateDATA_T(fc1_output , fc2_output, fc2_biases, fc2_weights, 11,
    FC2_NB_CHANNELS, FC2_CHANNELS_HEIGHT, 
    FC2_CHANNELS_WIDTH, FC2_NB_OUTPUTS, 
    FC2_OUTPUTS_HEIGHT, FC2_OUTPUTS_WIDTH, 
    FC2_ACTIVATION, FC1_MEM_CONT_OFFSET, 
    FC1_MEM_CONT_SIZE, FC1_MEM_WRAP_OFFSET, 
    FC1_MEM_WRAP_SIZE, FC1_MEM_STRIDE, 
    FC2_MEM_CONT_OFFSET, FC2_MEM_CONT_SIZE, 
    FC2_MEM_WRAP_OFFSET, FC2_MEM_WRAP_SIZE, FC2_MEM_STRIDE);

#ifdef BENCHMARK
    const Tick_T end_fc2 = tick();
    static RunningMean_T fc2_timing = {0.0, 0};
    benchmark("fc2", start_fc2, end_fc2, fc2_timing);
#endif

#ifdef SAVE_OUTPUTS
    FILE* fc2_stream = fopen("fc2_output.txt", "w");
    saveOutputs(FC2_NB_OUTPUTS, FC2_OUTPUTS_HEIGHT, FC2_OUTPUTS_WIDTH, FC2_MEM_CONT_OFFSET, FC2_MEM_CONT_SIZE, FC2_MEM_WRAP_OFFSET, FC2_MEM_WRAP_SIZE, FC2_MEM_STRIDE, fc2_output , fc2_stream, Network::Format::CHW);
    fclose(fc2_stream);
#endif
//modifcation debug
    // printf("fc2_output = ");
    // for (int i = 0; i < 10; i++) {
    //     printf("%d ", fc2_output[i]);
    // }
    // printf("\n");
    maxPropagate1(fc2_output, outputs, maxPropagate_val, FC2_NB_OUTPUTS, FC2_OUTPUTS_HEIGHT, FC2_OUTPUTS_WIDTH, FC2_MEM_CONT_OFFSET, FC2_MEM_CONT_SIZE, FC2_MEM_WRAP_OFFSET, FC2_MEM_WRAP_SIZE, FC2_MEM_STRIDE);

#ifdef SAVE_OUTPUTS
    FILE* max_stream = fopen("max_output.txt", "w");
    saveOutputs(FC2_NB_OUTPUTS, FC2_OUTPUTS_HEIGHT, FC2_OUTPUTS_WIDTH, FC2_MEM_CONT_OFFSET, FC2_MEM_CONT_SIZE, FC2_MEM_WRAP_OFFSET, FC2_MEM_WRAP_SIZE, FC2_MEM_STRIDE, outputs, max_stream, Network::Format::CHW);
    fclose(max_stream);
#endif

}

/*template<>
float Network::backpropagate(const DATA_T* input, const std::int32_t* labels){
   const float loss = 0.0f;
   return loss;
 }

int Network::gradientCheck(){
   return(0);
}*/

