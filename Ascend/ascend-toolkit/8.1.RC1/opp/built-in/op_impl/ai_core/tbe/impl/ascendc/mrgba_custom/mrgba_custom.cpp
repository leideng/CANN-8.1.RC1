/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file mrgba_custom.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "vector_scheduler.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr size_t CHANNEL_RATIO = 3;
constexpr float RATIO = 0.003921568627451;// this value is 1/255 for normalize

class KernelMrgba {
public:
    __aicore__ inline KernelMrgba()
    {
    }

    __aicore__ inline void Init(GM_ADDR rgb, GM_ADDR alpha, GM_ADDR dst, size_t bufferNum, size_t bufferBytes,
                                size_t gmIdx, size_t gmDataLen)
    {
        if (bufferBytes <= 0) {
            return;
        }
        pipe.InitBuffer(inQueueRgb, bufferNum, CHANNEL_RATIO * bufferBytes);
        pipe.InitBuffer(inQueueAlpha, bufferNum, bufferBytes);
        pipe.InitBuffer(outQueueDst, bufferNum, CHANNEL_RATIO * bufferBytes);

        pipe.InitBuffer(bufAlphaF16C1, bufferBytes * sizeof(half));
        pipe.InitBuffer(bufAlphaF16C3, CHANNEL_RATIO * bufferBytes * sizeof(half));
        pipe.InitBuffer(bufRgbF16C3, CHANNEL_RATIO * bufferBytes * sizeof(half));

        rgbGm.SetGlobalBuffer((__gm__ uint8_t *)rgb + CHANNEL_RATIO * gmIdx, CHANNEL_RATIO * gmDataLen);
        alphaGm.SetGlobalBuffer((__gm__ uint8_t *)alpha + gmIdx, gmDataLen);
        dstGm.SetGlobalBuffer((__gm__ uint8_t *)dst + CHANNEL_RATIO * gmIdx, CHANNEL_RATIO * gmDataLen);
    }

    __aicore__ inline void CalcForAlign32(uint32_t idx, size_t len)
    {
        if (len <= 0) {
            return;
        }
        uint32_t alphaIdx = idx;
        uint32_t rgbIdx = CHANNEL_RATIO * idx;
        size_t alphaLen = len;
        size_t rgbLen = CHANNEL_RATIO * len;
        // copyIn
        auto rgbLocal = inQueueRgb.AllocTensor<uint8_t>();
        auto alphaLocal = inQueueAlpha.AllocTensor<uint8_t>();

        DataCopy(alphaLocal, alphaGm[alphaIdx], alphaLen);
        DataCopy(rgbLocal, rgbGm[rgbIdx], rgbLen);
        inQueueRgb.EnQue(rgbLocal);
        inQueueAlpha.EnQue(alphaLocal);

        // compute
        rgbLocal = inQueueRgb.DeQue<uint8_t>();
        alphaLocal = inQueueAlpha.DeQue<uint8_t>();
        auto dstLocal = outQueueDst.AllocTensor<uint8_t>();

        auto alphaLocalF16C1 = bufAlphaF16C1.Get<half>();
        auto alphaBrbaLocalF16C3 = bufAlphaF16C3.Get<half>();
        auto rgbLocalF16C3 = bufRgbF16C3.Get<half>();

        Cast(alphaLocalF16C1, alphaLocal, RoundMode::CAST_NONE, alphaLen);

        const uint32_t alphaLocalBrbaShape[2] = {static_cast<uint32_t>(alphaLen), 3};
        const uint32_t alphaLocalShape[2] = {static_cast<uint32_t>(alphaLen), 1};
        const int32_t broadCastDim = 2;
        BroadCast<half, broadCastDim, 1>(alphaBrbaLocalF16C3, alphaLocalF16C1, alphaLocalBrbaShape, alphaLocalShape);

        half normalizedRatio = RATIO;
        Muls(alphaBrbaLocalF16C3, alphaBrbaLocalF16C3, normalizedRatio, rgbLen);
        Cast(rgbLocalF16C3, rgbLocal, RoundMode::CAST_NONE, rgbLen);
        Mul(rgbLocalF16C3, rgbLocalF16C3, alphaBrbaLocalF16C3, rgbLen);
        Cast(dstLocal, rgbLocalF16C3, RoundMode::CAST_FLOOR, rgbLen);

        outQueueDst.EnQue<uint8_t>(dstLocal);
        inQueueRgb.FreeTensor(rgbLocal);
        inQueueAlpha.FreeTensor(alphaLocal);

        // copyOut
        dstLocal = outQueueDst.DeQue<uint8_t>();
        DataCopy(dstGm[rgbIdx], dstLocal, rgbLen);
        outQueueDst.FreeTensor(dstLocal);
    }

protected:
    TPipe pipe;
    TQue <QuePosition::VECIN, BUFFER_NUM> inQueueRgb;
    TQue <QuePosition::VECIN, BUFFER_NUM> inQueueAlpha;
    TQue <QuePosition::VECOUT, BUFFER_NUM> outQueueDst;
    TBuf <TPosition::VECCALC> bufAlphaF16C1;
    TBuf <TPosition::VECCALC> bufAlphaF16C3;
    TBuf <TPosition::VECCALC> bufRgbF16C3;

    GlobalTensor <uint8_t> rgbGm;
    GlobalTensor <uint8_t> alphaGm;
    GlobalTensor <uint8_t> dstGm;
};

template <typename T>
__aicore__ void run_op(GM_ADDR rgb, GM_ADDR alpha, GM_ADDR dst, GM_ADDR tiling, float ubVarNum)
{
    GET_TILING_DATA(tilingData, tiling);
    uint32_t alphaLen = tilingData.alphaLen;
    VectorScheduler sch(tilingData.alphaLen, GetBlockNum(), BUFFER_NUM, ubVarNum, sizeof(uint8_t));
    KernelMrgba op;
    size_t orgVecIdx = GetBlockIdx() * sch.dataLenPerCore;
    op.Init(rgb, alpha, dst, sch.bufferNum, sch.dataBytesPerLoop, orgVecIdx, sch.dataLen);
    sch.run(&op, sch.dataLen);
}

extern "C" __global__ __aicore__ void mrgba_custom(GM_ADDR rgb, GM_ADDR alpha, GM_ADDR dst, GM_ADDR workspace,
                                                   GM_ADDR tiling)
{
    constexpr float ubVarNum = 100;
    run_op<uint8_t>(rgb, alpha, dst, tiling, ubVarNum);
}