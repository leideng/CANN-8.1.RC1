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

/* !
 * \file batch_mat_mul_v3.cpp
 * \brief
 */
#include "batch_mat_mul_v3.h"

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

#define BMMV3_IMPL_CLASS(templateClass, ...)                                                                          \
    do {                                                                                                              \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y, false, LayoutMode::NORMAL>;               \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS, false, LayoutMode::NORMAL>;   \
        TPipe pipe;                                                                                                   \
        if (tilingData.matmulTiling.matmulRunInfo.transA == 0 && tilingData.matmulTiling.matmulRunInfo.transB == 0) { \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false, LayoutMode::NORMAL>;         \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false, LayoutMode::NORMAL>;         \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        } else if (tilingData.matmulTiling.matmulRunInfo.transA == 1 &&                                               \
            tilingData.matmulTiling.matmulRunInfo.transB == 0) {                                                      \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true, LayoutMode::NORMAL>;          \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false, LayoutMode::NORMAL>;         \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        } else if (tilingData.matmulTiling.matmulRunInfo.transA == 0 &&                                               \
            tilingData.matmulTiling.matmulRunInfo.transB == 1) {                                                      \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false, LayoutMode::NORMAL>;         \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true, LayoutMode::NORMAL>;          \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        } else {                                                                                                      \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true, LayoutMode::NORMAL>;          \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true, LayoutMode::NORMAL>;          \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        }                                                                                                             \
    } while (0)

#define BMMV3_IMPL_CLASS_COMMON(templateClass, ...)                                                                   \
    do {                                                                                                              \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;                                          \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                              \
        TPipe pipe;                                                                                                   \
        if (tilingData.matmulTiling.matmulRunInfo.transA == 0 && tilingData.matmulTiling.matmulRunInfo.transB == 0) { \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        } else if (tilingData.matmulTiling.matmulRunInfo.transA == 1 &&                                               \
            tilingData.matmulTiling.matmulRunInfo.transB == 0) {                                                      \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;                              \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        } else if (tilingData.matmulTiling.matmulRunInfo.transA == 0 &&                                               \
            tilingData.matmulTiling.matmulRunInfo.transB == 1) {                                                      \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                              \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        } else {                                                                                                      \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;                              \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                              \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        }                                                                                                             \
    } while (0)

extern "C" __global__ __aicore__ void batch_mat_mul_v3(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
    GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    GET_TILING_DATA(tilingData, tilingGM);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    if (TILING_KEY_IS(10000000000000000001UL)) {
        BMMV3_IMPL_CLASS_COMMON(BatchMatMulCommonKernel, BatchMatMulCommonBaseBlock, MM_CFG_VEC_ND2NZ);
    }
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(10000000000000000000UL)) {
        BMMV3_IMPL_CLASS_COMMON(BatchMatMulUnalignedKernel, MatmulBaseBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000000000000000001UL)) {
        KERNEL_TASK_TYPE(10000000000000000001UL, KERNEL_TYPE_AIC_ONLY);
        BMMV3_IMPL_CLASS_COMMON(BatchMatMulCommonKernel, BatchMatMulCommonBaseBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000000000000000101UL)) {
        KERNEL_TASK_TYPE(10000000000000000101UL, KERNEL_TYPE_AIC_ONLY);
        BMMV3_IMPL_CLASS_COMMON(BatchMatMulCommonKernel, BatchMatMulCommonBaseBlock, MM_CFG_NO_PRELOAD,
                                MatmulCallBackFunc<nullptr, BmmCopyAL1, nullptr>);
    } else if (TILING_KEY_IS(10000000000000000201UL)) {
        KERNEL_TASK_TYPE(10000000000000000201UL, KERNEL_TYPE_AIC_ONLY);
        BMMV3_IMPL_CLASS_COMMON(BatchMatMulCommonKernel, BatchMatMulCommonBaseBlock, MM_CFG_NO_PRELOAD,
                                MatmulCallBackFunc<nullptr, nullptr, BmmCopyBL1>);
    } else if (TILING_KEY_IS(10000000000000001001UL)) { // need to be set
        KERNEL_TASK_TYPE(10000000000000001001UL, KERNEL_TYPE_AIC_ONLY);
        BMMV3_IMPL_CLASS(BatchMatMulMultiBatchKernel, BatchMatMulMultiBatchBaseBlock);
    } else if (TILING_KEY_IS(10000000000000010001UL)) {
        KERNEL_TASK_TYPE(10000000000000010001UL, KERNEL_TYPE_AIC_ONLY);
        BMMV3_IMPL_CLASS(BatchMatMulMultiBatchFullLoadKernel, BatchMatMulMultiBatchFullLoadBlock);
    } else if (TILING_KEY_IS(10000000000000001000UL)) {
        BMMV3_IMPL_CLASS(BatchMatMulUnalignedMultiBatchKernel, BatchMatMulUnalignedMultiBatchBaseBlock);
    }
#endif
}