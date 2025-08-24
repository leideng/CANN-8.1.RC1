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
 * \file service_gather_sin_cos.h
 * \brief
 */

#ifndef SERVICE_GATHER_SIN_COS_H
#define SERVICE_GATHER_SIN_COS_H

#include "mla_prolog_comm.h"

namespace MlaProlog {

// 只支持获取单个batch的信息
__aicore__ inline void GatherTokenIndex(const GlobalTensor<int64_t>& tokenIndexGm, const int64_t batchIndex,
                                        int64_t& tokenIndex) {
    tokenIndex = tokenIndexGm.GetValue(batchIndex);
}

// 只支持对单个batch的处理 row为1
template <typename T, typename O>
__aicore__ inline void GatherSinCos(const GlobalTensor<T>& cosGm, const GlobalTensor<T>& sinGm, int64_t tokenIndex,
                                    int64_t curVecToken, LocalTensor<uint8_t>& shareTmpUb, int64_t row, int64_t col,
                                    LocalTensor<O>& cosLocal, LocalTensor<O>& sinLocal) {
    int64_t offset = col * tokenIndex;
    int64_t curDataSize = col * curVecToken;
    if constexpr (IsSameType<O, float>::value) {
        auto tmpUbBf16 = shareTmpUb.ReinterpretCast<bfloat16_t>();
        DataCopy(tmpUbBf16, cosGm[offset], curDataSize);
        DataCopy(tmpUbBf16[curDataSize], sinGm[offset], curDataSize);
        event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(event);
        WaitFlag<HardEvent::MTE2_V>(event);
        Cast(cosLocal, tmpUbBf16, RoundMode::CAST_NONE, curDataSize);
        Cast(sinLocal, tmpUbBf16[curDataSize], RoundMode::CAST_NONE, curDataSize);
    } else {
        DataCopy(cosLocal, cosGm[offset], curDataSize);
        DataCopy(sinLocal, sinGm[offset], curDataSize);
    }
    pipe_barrier(PIPE_V);
    uint8_t blockNumPerRow = col / (32 / sizeof(O));
    Muls<O>(sinLocal, sinLocal, -1.0f, col >> 1, curVecToken, {1, 1, blockNumPerRow, blockNumPerRow});
    pipe_barrier(PIPE_V);
}

}

#endif