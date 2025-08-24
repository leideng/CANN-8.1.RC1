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
 * \file dequant_per_channel.cpp
 * \brief
 */
#include "../quant_per_channel/simple_broadcast_base.h"
using AscendC::HardEvent;
using namespace AscendC;

template <bool HAS_OFFSET, bool CUT_N>
class DequantPerChannelKernel : public SimpleBroadcastBase {
public:
    __aicore__ inline DequantPerChannelKernel() {}

    __aicore__ inline void Process(GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR x)
{
        InitBroadcastGlobalTensor<int8_t>(yGm_, y);
        InitNormalGlobalTensor<half>(scaleGm_, scale);
        InitNormalGlobalTensor<int8_t>(offsetGm_, offset);
        InitBroadcastGlobalTensor<half>(xGm_, x);

        InitBroadcastQueue<int8_t>(yQueue_);
        InitNormalQueue<half>(scaleQueue_);
        InitNormalQueue<int8_t>(offsetQueue_);
        InitBroadcastQueue<half>(xQueue_);

        InitNormalTBuf<half>(f16TmpBuf_);
        InitNormalTBuf<half>(f16TmpBuf2_);

        LocalTensor<half> yCastedLocal = f16TmpBuf_.Get<half>();        // cast<int8-half>(y)
        LocalTensor<half> offsetCastedLocal = f16TmpBuf2_.Get<half>();  // cast<int8-half>(offset)

        LocalTensor<half> scaleLocal;
        LocalTensor<int8_t> offsetLocal;
        if constexpr (!CUT_N) {
            CopyInNormal(scaleGm_, scaleQueue_);
            scaleLocal = scaleQueue_.DeQue<half>();
            if constexpr (HAS_OFFSET) {
                CopyInNormal(offsetGm_, offsetQueue_);
                offsetLocal = offsetQueue_.DeQue<int8_t>();
                Cast(offsetCastedLocal, offsetLocal, RoundMode::CAST_NONE, innerCount_);
                PipeBarrier<PIPE_V>();
            }
        }

        do {
            CopyInBroadcast(yGm_, yQueue_);
            LocalTensor<int8_t> yLocal = yQueue_.DeQue<int8_t>();
            LocalTensor<half> xLocal = xQueue_.AllocTensor<half>();

            if constexpr (CUT_N) {
                CopyInNormal(scaleGm_, scaleQueue_);
                scaleLocal = scaleQueue_.DeQue<half>();
                if constexpr (HAS_OFFSET) {
                    CopyInNormal(offsetGm_, offsetQueue_);
                    offsetLocal = offsetQueue_.DeQue<int8_t>();
                    Cast(offsetCastedLocal, offsetLocal, RoundMode::CAST_NONE, innerCount_);
                    PipeBarrier<PIPE_V>();
                }
            }

            do {
                LocalTensor<int8_t> yLocalInner = GetInnerTensor(yLocal);
                LocalTensor<half> xLocalInner = GetInnerTensor(xLocal);
                Cast(yCastedLocal, yLocalInner, RoundMode::CAST_NONE, innerCount_);
                if constexpr (HAS_OFFSET) {
                    PipeBarrier<PIPE_V>();
                    Sub(yCastedLocal, yCastedLocal, offsetCastedLocal, innerCount_);
                }
                PipeBarrier<PIPE_V>();
                Mul(xLocalInner, scaleLocal, yCastedLocal, innerCount_);
            } while (InnerNext());

            xQueue_.EnQue(xLocal);
            yQueue_.FreeTensor(yLocal);
            if constexpr (CUT_N) {
                scaleQueue_.FreeTensor(scaleLocal);
                if constexpr (HAS_OFFSET) {
                    offsetQueue_.FreeTensor(offsetLocal);
                }
            }
            CopyOut(xGm_, xQueue_);
        } while (OuterNext());

        if constexpr (!CUT_N) {
            scaleQueue_.FreeTensor(scaleLocal);
            if constexpr (HAS_OFFSET) {
                offsetQueue_.FreeTensor(offsetLocal);
            }
        }
    }

private:
    GlobalTensor<int8_t> yGm_;
    GlobalTensor<half> scaleGm_;
    GlobalTensor<int8_t> offsetGm_;
    GlobalTensor<half> xGm_;

    TQue<QuePosition::VECIN, BUFFER_NUM> yQueue_;
    TQue<QuePosition::VECIN, 1> scaleQueue_;
    TQue<QuePosition::VECIN, 1> offsetQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> xQueue_;

    TBuf<QuePosition::VECCALC> f16TmpBuf_;
    TBuf<QuePosition::VECCALC> f16TmpBuf2_;
};

template <bool HAS_OFFSET>
class DequantPerTensorKernel : public SimpleBroadcastBase {
public:
    __aicore__ inline DequantPerTensorKernel() {}

    __aicore__ inline void Process(GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR x)
    {
        InitBroadcastGlobalTensor<int8_t>(yGm_, y);
        half scaleValue = *(__gm__ half *)scale;
        half offsetNegValue = 0.0f;
        if constexpr (HAS_OFFSET) {
            offsetNegValue = (half)(-*((__gm__ int8_t *)offset));
        }
        InitBroadcastGlobalTensor<half>(xGm_, x);

        InitBroadcastQueue<int8_t>(yQueue_);
        InitBroadcastQueue<half>(xQueue_);

        InitNormalTBuf<half>(f16TmpBuf_);
        LocalTensor<half> yCastedLocal = f16TmpBuf_.Get<half>();        // cast<int8-half>(y)

        do {
            CopyInBroadcast(yGm_, yQueue_);
            LocalTensor<int8_t> yLocal = yQueue_.DeQue<int8_t>();
            LocalTensor<half> xLocal = xQueue_.AllocTensor<half>();
            do {
                LocalTensor<int8_t> yLocalInner = GetInnerTensor(yLocal);
                LocalTensor<half> xLocalInner = GetInnerTensor(xLocal);
                Cast(yCastedLocal, yLocalInner, RoundMode::CAST_NONE, innerCount_);
                if constexpr (HAS_OFFSET) {
                    PipeBarrier<PIPE_V>();
                    Adds(yCastedLocal, yCastedLocal, offsetNegValue, innerCount_);
                }
                PipeBarrier<PIPE_V>();
                Muls(xLocalInner, yCastedLocal, scaleValue, innerCount_);
            } while (InnerNext());

            xQueue_.EnQue(xLocal);
            yQueue_.FreeTensor(yLocal);

            CopyOut(xGm_, xQueue_);
        } while (OuterNext());
    }

private:
    GlobalTensor<int8_t> yGm_;
    GlobalTensor<half> xGm_;

    TQue<QuePosition::VECIN, BUFFER_NUM> yQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> xQueue_;

    TBuf<QuePosition::VECCALC> f16TmpBuf_;
};

extern "C" __global__ __aicore__ void dequant_per_channel(
    GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR x,GM_ADDR workspace,GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(2000000000)) {
        // per tensor, no offset
        DequantPerTensorKernel<false> op;
        op.Init(&tilingData);
        op.Process(y, scale, offset, x);
    }
    if (TILING_KEY_IS(2100000000)) {
        // per channel, no offset, cut n
        DequantPerChannelKernel<false, true> op;
        op.Init(&tilingData);
        op.Process(y, scale, offset, x);
    }
    if (TILING_KEY_IS(2200000000)) {
        // per channel, no offset, cut b
        DequantPerChannelKernel<false, false> op;
        op.Init(&tilingData);
        op.Process(y, scale, offset, x);
    }
    if (TILING_KEY_IS(2000000001)) {
        // per tensor, has offset
        DequantPerTensorKernel<true> op;
        op.Init(&tilingData);
        op.Process(y, scale, offset, x);
    }
    if (TILING_KEY_IS(2100000001)) {
        // per channel, has offset, cut n
        DequantPerChannelKernel<true, true> op;
        op.Init(&tilingData);
        op.Process(y, scale, offset, x);
    }
    if (TILING_KEY_IS(2200000001)) {
        // per channel, has offset, cut b
        DequantPerChannelKernel<true, false> op;
        op.Init(&tilingData);
        op.Process(y, scale, offset, x);
    }
}