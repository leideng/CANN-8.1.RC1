/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file gemm_v2.cpp
 * \brief
 */
#include "../mat_mul_v3/mat_mul_base_kernel.h"
using namespace AscendC;
using namespace matmul;
#ifndef DTYPE_BIAS
#define DTYPE_BIAS half
#endif

#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

#if defined(FORMAT_A) && FORMAT_A == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_a = CubeFormat::NZ;
#else
constexpr CubeFormat format_a = CubeFormat::ND;
#endif

#if defined(FORMAT_B) && FORMAT_B == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_b = CubeFormat::NZ;
#else
constexpr CubeFormat format_b = CubeFormat::ND;
#endif

#if defined(FORMAT_C) && FORMAT_C == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_c = CubeFormat::NZ;
#else
constexpr CubeFormat format_c = CubeFormat::ND;
#endif

#define MMV3_IMPL_CLASS(templateClass, ...)                                                                                \
   do {                                                                                                             \
        using cType = MatmulType<AscendC::TPosition::GM, format_c, DTYPE_C>;                                         \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                             \
        TPipe pipe;                                                                                                  \
        if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 0) {                          \
            using aType = MatmulType<AscendC::TPosition::GM, format_a, DTYPE_A, false>;                            \
            using bType = MatmulType<AscendC::TPosition::GM, format_b, DTYPE_B, false>;                            \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                                         \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process(0, 1);                                                                                         \
        } else if (tilingData.matmulRunInfo.transA == 1 && tilingData.matmulRunInfo.transB == 0) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, format_a, DTYPE_A, true>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_b, DTYPE_B, false>;                            \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                                         \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process(0, 1);                                                                                         \
        } else if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 1) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, format_a, DTYPE_A, false>;                            \
            using bType = MatmulType<AscendC::TPosition::GM, format_b, DTYPE_B, true>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                                         \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process(0, 1);                                                                                         \
        } else {                                                                                                     \
            using aType = MatmulType<AscendC::TPosition::GM, format_a, DTYPE_A, true>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_b, DTYPE_B, true>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                                         \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process(0, 1);                                                                                         \
        }                                                                                                            \
    } while (0)

extern "C" __global__ __aicore__ void gemm_v2(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR alpha, GM_ADDR beta, GM_ADDR ref_c,
    GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    SetSysWorkspace(workspaceGM);
    GET_TILING_DATA(tilingData, tilingGM);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

    GM_ADDR biasGM = nullptr;
    GM_ADDR offsetWGM = nullptr;
    // 第一个模板使用mix类型的，使得整个算子的coreType在dyn场景都为mix，静态则根据选择的tilingkey决定coreType
    if (TILING_KEY_IS(10000000000000000001UL)) {
        MMV3_IMPL_CLASS(MatmulBaseKernel, MatmulBaseBlock, MM_CFG_NO_PRELOAD);
    }
}