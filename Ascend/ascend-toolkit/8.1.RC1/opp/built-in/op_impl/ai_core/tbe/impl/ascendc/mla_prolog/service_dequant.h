/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License. 
 */

/*!
 * \file kernel_mla_prolog_vec_s1_cub_s2.h
 * \brief
 */

#ifndef SERVICE_DEQUANT_H
#define SERVICE_DEQUANT_H

#include "mla_prolog_comm.h"

namespace MlaProlog {

template <typename T, typename C, typename O>
__aicore__ inline void Dequant(const GlobalTensor<T> &inputGm, const GlobalTensor<C> &scale1Gm,
                               C scale2, uint32_t row, uint32_t col, uint32_t oriRow, uint32_t stride,
                               const GlobalTensor<O> &outputGm, const LocalTensor<uint8_t> &shareTmpUb) {
    int64_t count = row * col;
    int64_t oriCol = col + stride;

    LocalTensor<T> inputLocal = shareTmpUb.ReinterpretCast<T>(); // count * sizeof(T)
    LocalTensor<C> scaleLocal = inputLocal[count + 16].template ReinterpretCast<C>(); // count * sizeof(C)
    LocalTensor<C> computeLocal = scaleLocal[count + 16].template ReinterpretCast<C>(); // count * sizeof(C)
    LocalTensor<O> outputLocal = computeLocal[count + 16].template ReinterpretCast<O>(); // count * sizeof(O)

    DataCopyParams copyParams {
        static_cast<uint16_t>(row),
        static_cast<uint16_t>(col * sizeof(T) / 32U),
        static_cast<uint16_t>(stride * sizeof(T) / 32U),
        0};

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    for (int64_t rowOffset = 0; rowOffset < oriRow; rowOffset += row) {
        int64_t inputOffset = rowOffset * oriCol;
        int64_t outputOffset = rowOffset * col;
        // copy in
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        DataCopy(inputLocal, inputGm[inputOffset], copyParams);
        DataCopy(scaleLocal, scale1Gm[inputOffset], copyParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);

        // compute
        // cast
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Cast(computeLocal, inputLocal, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        // muls & mul
        Muls(computeLocal, computeLocal, scale2, count);
        PipeBarrier<PIPE_V>();
        Mul(computeLocal, computeLocal, scaleLocal, count);
        PipeBarrier<PIPE_V>();
        // cast
        Cast(outputLocal, computeLocal, RoundMode::CAST_RINT, count);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        // copy out
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(outputGm[outputOffset], outputLocal, count);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }   
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

}
#endif
