/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
 * \file transpose_batch_mat_mul.cpp
 * \brief
 */
#include "transpose_batch_mat_mul.h"

using namespace AscendC;
using namespace matmul;
using namespace TransposeBatchMatmulSpace;
#ifndef DTYPE_BIAS
#define DTYPE_BIAS half
#endif

#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

constexpr CubeFormat format_x1 = CubeFormat::ND;
constexpr CubeFormat format_x2 = CubeFormat::ND;
constexpr CubeFormat format_y = CubeFormat::ND;

#define BMMV3_IMPL_CLASS_COMMON(templateClass, Mode, ...)                                                             \
    do {                                                                                                              \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;                                          \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                              \
        TPipe pipe;                                                                                                   \
                                                                                                                      \
        using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                                 \
        using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                                 \
        templateClass<aType, bType, cType, biasType, Mode, __VA_ARGS__> op;                                           \
        op.Init(aGM, bGM, cGM, biasGM, scalesGM, user, &tilingData, &pipe);                                           \
        op.Process();                                                                                                 \
                                                                                                                      \
    } while (0)

extern "C" __global__ __aicore__ void transpose_batch_mat_mul(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR scalesGM, 
                                                              GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    GET_TILING_DATA(tilingData, tilingGM);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (TILING_KEY_IS(10000000000000000001UL)) {
        BMMV3_IMPL_CLASS_COMMON(TransposeBatchMatMulKernel, TBMM_MODE::TRANS_BMM_TRANS, TransposeBatchMatMulBlock,
                                MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000000000000000000UL)) {
        BMMV3_IMPL_CLASS_COMMON(TransposeBatchMatMulKernel, TBMM_MODE::BMM_TRANS, TransposeBatchMatMulBlock,
                                MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000000000000000011UL)) {
        BMMV3_IMPL_CLASS_COMMON(TransposeBatchMatMulKernel, TBMM_MODE::TRANS_BMM_TRANS_TRANS, TransposeBatchMatMulBlock,
                                MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000000000000000010UL)) {
        BMMV3_IMPL_CLASS_COMMON(TransposeBatchMatMulKernel, TBMM_MODE::BMM_TRANS_TRANS, TransposeBatchMatMulBlock,
                                MM_CFG_NO_PRELOAD);
    }
}