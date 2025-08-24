/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file detect_mat_mul.cpp
 * \brief
 */
#include "detect_mat_mul.h"
#include "mat_mul_kernel_stress_detect.h"
#include "mat_mul_asw_kernel_stress_detect.h"
#include "mat_mul_optimized_fixpipe_algorithm_stress_detect.h"
#include "mat_mul_bl1_full_load_stress_detect.h"

using namespace AscendC;
using namespace matmul;
#ifndef DTYPE_BIAS
#define DTYPE_BIAS half
#endif

#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

#if defined(FORMAT_X1) && FORMAT_X1 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x1 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x1 = CubeFormat::ND;
#endif

#if defined(FORMAT_X2) && FORMAT_X2 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x2 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x2 = CubeFormat::ND;
#endif

#if defined(FORMAT_Y) && FORMAT_Y == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_y = CubeFormat::NZ;
#else
constexpr CubeFormat format_y = CubeFormat::ND;
#endif

#define MM_STRESS_DETECT_IMPL(templateFunc, ...)                                                                                 \
    do {                                                                                                             \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;                                         \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                             \
        if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 0) {                          \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                            \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateFunc<aType, bType, cType, biasType>(aGM, bGM, cGM, biasGM, tilingData, user);                    \
        } else if (tilingData.matmulRunInfo.transA == 1 && tilingData.matmulRunInfo.transB == 0) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateFunc<aType, bType, cType, biasType>(aGM, bGM, cGM, biasGM, tilingData, user);                    \
        } else if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 1) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                            \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateFunc<aType, bType, cType, biasType>(aGM, bGM, cGM, biasGM, tilingData, user);                    \
        } else {                                                                                                     \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateFunc<aType, bType, cType, biasType>(aGM, bGM, cGM, biasGM, tilingData, user);                    \
        }                                                                                                            \
    } while(0)


#define MM_STRESS_DETECT_IMPL_CLASS(templateClass, ...)                                                                          \
    do {                                                                                                             \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;                                         \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                             \
        TPipe pipe;                                                                                                  \
        if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 0) {                          \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                            \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else if (tilingData.matmulRunInfo.transA == 1 && tilingData.matmulRunInfo.transB == 0) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 1) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                            \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else {                                                                                                     \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        }                                                                                                            \
    } while (0)


extern "C" __global__ __aicore__ void detect_mat_mul(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
    GM_ADDR offsetWGM, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    GET_TILING_DATA(tilingData, tilingGM);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

#if defined(__DAV_C310__)
    // Adaptive Sliding Window Kernel
    if (TILING_KEY_IS(10000000000000000001UL)) {
        MM_STRESS_DETECT_IMPL_CLASS(MatmulStressDetect::MatmulAswKernel, MatmulStressDetect::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    }
#else
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    // 第一个模板使用mix类型的，使得整个算子的coreType在dyn场景都为mix，静态则根据选择的tilingkey决定coreType
    if (TILING_KEY_IS(10000000000000000000UL)) {
        MM_STRESS_DETECT_IMPL_CLASS(MatmulStressDetect::MatmulBaseUnAlignedKernel, MatmulStressDetect::MatmulBaseBlock, MM_CFG_VEC_ND2NZ);
    } else if (TILING_KEY_IS(10000000000000000001UL)) {
        MM_STRESS_DETECT_IMPL_CLASS(MatmulStressDetect::MatmulBaseKernel, MatmulStressDetect::MatmulBaseBlock, MM_CFG_VEC_ND2NZ);
    }
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(10000000000000000000UL)) {
        MM_STRESS_DETECT_IMPL_CLASS(MatmulStressDetect::MatmulBaseUnAlignedKernel, MatmulStressDetect::MatmulBaseBlock, MatmulStressDetect::MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000000000000000021UL)) {
        MM_STRESS_DETECT_IMPL_CLASS(MatmulStressDetect::MatMulSingleCoreSplitKKernel, MatmulStressDetect::MatmulSingleCoreSplitKBaseBlock, MatmulStressDetect::MM_CFG_PRELOAD);
    } else if (TILING_KEY_IS(10000000000000000020UL)) {
        MM_STRESS_DETECT_IMPL_CLASS(MatmulStressDetect::MatMulUnAlignedSingleCoreSplitKKernel, MatmulStressDetect::MatmulSingleCoreSplitKBaseBlock, MatmulStressDetect::MM_CFG_PRELOAD);
    } else if (TILING_KEY_IS(10000000000000000031UL)) {
        MM_STRESS_DETECT_IMPL(MatmulStressDetect::MatMulKernelDeterministicSplitK, aGM, bGM, cGM, biasGM, tilingData, user);
    } else if (TILING_KEY_IS(10000000000000000030UL)) {
        MM_STRESS_DETECT_IMPL(MatmulStressDetect::MatMulUnAlignedKernelDeterministicSplitK, aGM, bGM, cGM, biasGM, tilingData, user);
    } else if (TILING_KEY_IS(10000000000000000001UL)) {
        KERNEL_TASK_TYPE(10000000000000000001UL, KERNEL_TYPE_AIC_ONLY);
        MM_STRESS_DETECT_IMPL_CLASS(MatmulStressDetect::MatmulBaseKernel, MatmulStressDetect::MatmulBaseBlock, MatmulStressDetect::MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000000000000000201UL)) {
        KERNEL_TASK_TYPE(10000000000000000201UL, KERNEL_TYPE_AIC_ONLY);
        MM_STRESS_DETECT_IMPL_CLASS(MatmulStressDetect::MatmulBaseKernelBL1FullLoad, MatmulStressDetect::MatmulBaseBlock, MatmulStressDetect::MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000000000000010201UL)) {
        MM_STRESS_DETECT_IMPL_CLASS(MatmulStressDetect::MatmulBaseUnalignedNKernel, MatmulStressDetect::MatmulBaseBlock, MatmulStressDetect::MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000000000000000200UL)) {
        MM_STRESS_DETECT_IMPL_CLASS(MatmulStressDetect::MatmulBaseUnAlignedKernelBL1FullLoad, MatmulStressDetect::MatmulBaseBlock, MatmulStressDetect::MM_CFG_NO_PRELOAD);
    }
#endif
#endif
}
