/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file common_quant.h
 * \brief
 */

#ifndef NORM_COMMMON_QUANT_H
#define NORM_COMMMON_QUANT_H

#include "kernel_operator.h"

static constexpr uint32_t BLOCK_SIZE = 32;

using AscendC::HardEvent;

template <typename S, typename O>
__aicore__ inline void GetQuantInfo(S &varScale, O &varOffset, __gm__ uint8_t *scale, __gm__ uint8_t *offset,
                                    AscendC::TBuf<AscendC::TPosition::VECCALC> &buf)
{
    AscendC::GlobalTensor<half> gm_s;
    AscendC::GlobalTensor<int8_t> gm_o;
    gm_s.SetGlobalBuffer((__gm__ half *)scale);
    gm_o.SetGlobalBuffer((__gm__ int8_t *)offset);

    AscendC::LocalTensor<half> scale_buffer = buf.Get<half>();
    DataCopy(scale_buffer, gm_s, BLOCK_SIZE / sizeof(half));
    AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    varScale = 1 / static_cast<S>(scale_buffer.GetValue(0));

    AscendC::LocalTensor<int8_t> offset_buffer = buf.Get<int8_t>();
    AscendC::SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    DataCopy(offset_buffer, gm_o, BLOCK_SIZE / sizeof(int8_t));
    AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    varOffset = static_cast<O>(offset_buffer.GetValue(0));
    AscendC::SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
}

template<typename T>
__aicore__ inline void GetScaleAndOffset(float &varScale, float &varOffset, __gm__ uint8_t *scale,
                                         __gm__ uint8_t *offset, AscendC::TBuf<AscendC::TPosition::VECCALC> &buf)
{
    AscendC::GlobalTensor<T> gm_s;
    gm_s.SetGlobalBuffer((__gm__ T *)scale);
    AscendC::LocalTensor<T> scale_buffer = buf.Get<T>();
    DataCopy(scale_buffer, gm_s, BLOCK_SIZE / sizeof(T));
    if constexpr (AscendC::IsSameType<T, half>::value) {
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        varScale = 1 / (float)(scale_buffer.GetValue(0));
    } else {
        AscendC::LocalTensor<float> tmpFp32 = buf.Get<float>();
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(tmpFp32, scale_buffer, AscendC::RoundMode::CAST_NONE, 1);
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        varScale = 1 / (float)(tmpFp32.GetValue(0));
    }
    AscendC::GlobalTensor<int8_t> gm_o;
    gm_o.SetGlobalBuffer((__gm__ int8_t *)offset);
    AscendC::LocalTensor<int8_t> tmpInt8 = buf.Get<int8_t>();
    DataCopy(tmpInt8, gm_o, BLOCK_SIZE / sizeof(int8_t));
    AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    varOffset = (float)(tmpInt8.GetValue(0));
}
#endif