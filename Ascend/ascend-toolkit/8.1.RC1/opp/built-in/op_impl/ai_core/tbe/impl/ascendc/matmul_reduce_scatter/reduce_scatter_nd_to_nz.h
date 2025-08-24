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

/*!
 * \file reduce_scatter_nd_to_nz.h
 * \brief
 */
#ifndef REDUCE_SCATTER_ND_TO_NZ_H
#define REDUCE_SCATTER_ND_TO_NZ_H

#define ENALBE_ND2NZ 1

#if defined(__CCE_KT_TEST__)
#define SET_G_CORE_TYPE_IS_AIV thread_local int g_coreType = 2
#define SET_G_CORE_TYPE_IS_AIC thread_local int g_coreType = 1
#define DTYPE_X1 half
#define DTYPE_Y half
#else
#define SET_G_CORE_TYPE_IS_AIV
#define SET_G_CORE_TYPE_IS_AIC
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "mc2_tiling_struct.h"
#include "../mat_mul_v3/mat_mul_nd2nz.h"

namespace AscendC {
using namespace matmul;

template <class T>
__aicore__ inline void MatrixBtoNZMc2(GM_ADDR workspace, GM_ADDR src, const MatmulReduceScatterTilingData* tilingData,
                                      bool isTransposeB, TBuf<TPosition::VECCALC> &tmpBuf)
{
    if (g_coreType == AIV) {
        if (block_idx >= tilingData->tileTiling.usedCoreNum) {
            // 未使用的AIV核同步等待
            ffts_cross_core_sync(PIPE_MTE3, 0x21 + (3 << 8));
            return;
        }
        MatrixBtoNZV2<T>(workspace, src, tilingData->tileTiling, isTransposeB, tmpBuf, tilingData->socParam.baseBN, tilingData->socParam.baseBD);
        // 先AIC等待AIV, 再AIC之间一次同步
        ffts_cross_core_sync(PIPE_MTE3, 0x21 + (3 << 8));  // v侧做完才能做c侧
    } else {
#ifndef __CCE_KT_TEST__
        wait_flag_dev(3);
        ffts_cross_core_sync(PIPE_MTE3, 0x01 + (4 << 8));
        wait_flag_dev(4);
#endif
    }
}
#endif // REDUCE_SCATTER_ND_TO_NZ_H
}