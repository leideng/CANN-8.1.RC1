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
 * \file background_replace.cpp
 * \brief BackgroundReplace 算子 Kernel 入口.
 */

#include "kernel_operator.h"
#include "vector_scheduler.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t CHANNEL_NUM = 3;

template <typename T1, typename T2>
class KernelBackgroundReplaceC1 {
public:
    __aicore__ inline KernelBackgroundReplaceC1()
    {}
    __aicore__ inline void Init(GM_ADDR bkg, GM_ADDR src, GM_ADDR mask, GM_ADDR out, GM_ADDR workspace, size_t bufferNum, 
                                size_t bufferBytes, size_t gmIdx, size_t gmDataLen)
    {
        if (bufferBytes <= 0) {
            return;
        }
        pipe.InitBuffer(inQueueBkg, bufferNum, bufferBytes);
        pipe.InitBuffer(inQueueSrc, bufferNum, bufferBytes);
        pipe.InitBuffer(outQueuedst, bufferNum, bufferBytes);
        if (sizeof(T1) == 1) {
            pipe.InitBuffer(calcBufX1, bufferBytes * sizeof(half));
            pipe.InitBuffer(calcBufX2, bufferBytes * sizeof(half));
            pipe.InitBuffer(inQueueMask, bufferNum, bufferBytes * sizeof(half));
        } else {
            pipe.InitBuffer(inQueueMask, bufferNum, bufferBytes);
        }

        x1Gm.SetGlobalBuffer((__gm__ T1*)bkg + gmIdx, gmDataLen);
        x2Gm.SetGlobalBuffer((__gm__ T1*)src + gmIdx, gmDataLen);
        x3Gm.SetGlobalBuffer((__gm__ T2*)mask + gmIdx, gmDataLen);
        zGm.SetGlobalBuffer((__gm__ T1*)out + gmIdx, gmDataLen);
    }
    __aicore__ void CalcForAlign32(uint32_t idx, size_t len)
    {
        if (len <= 0) {
            return;
        }
        // copyIn
        auto bkgLocal = inQueueBkg.AllocTensor<T1>();
        auto srcLocal = inQueueSrc.AllocTensor<T1>();
        auto maskLocal = inQueueMask.AllocTensor<T2>();
        auto dstLocal = outQueuedst.AllocTensor<T1>();
        DataCopy(bkgLocal, x1Gm[idx], len);
        DataCopy(srcLocal, x2Gm[idx], len);
        DataCopy(maskLocal, x3Gm[idx], len);
        inQueueBkg.EnQue(bkgLocal);
        inQueueSrc.EnQue(srcLocal);
        inQueueMask.EnQue(maskLocal);

        //compute
        bkgLocal = inQueueBkg.DeQue<T1>();
        srcLocal = inQueueSrc.DeQue<T1>();
        maskLocal = inQueueMask.DeQue<T2>();

        if constexpr(sizeof(T1) == 1) {
            LocalTensor<half> bkgTmpLocal = calcBufX1.Get<half>();
            LocalTensor<half> srcTmpLocal = calcBufX2.Get<half>();
            Cast(bkgTmpLocal, bkgLocal, RoundMode::CAST_NONE, len);
            Cast(srcTmpLocal, srcLocal, RoundMode::CAST_NONE, len);

            Mul(srcTmpLocal, srcTmpLocal, maskLocal, len);
            Mul(maskLocal, bkgTmpLocal, maskLocal, len);
            Sub(bkgTmpLocal, bkgTmpLocal, maskLocal, len);
            Add(bkgTmpLocal, bkgTmpLocal, srcTmpLocal, len);

            Cast(dstLocal, bkgTmpLocal, RoundMode::CAST_NONE, len);
        } else {
            Mul(srcLocal, srcLocal, maskLocal, len);
            Mul(maskLocal, bkgLocal, maskLocal, len);
            Sub(bkgLocal, bkgLocal, maskLocal, len);
            Add(dstLocal, bkgLocal, srcLocal, len);
        }

        //CopyOut
        outQueuedst.EnQue(dstLocal);
        inQueueBkg.FreeTensor(bkgLocal);
        inQueueSrc.FreeTensor(srcLocal);
        inQueueMask.FreeTensor(maskLocal);
        dstLocal = outQueuedst.DeQue<T1>();
        DataCopy(zGm[idx], dstLocal, len);
        outQueuedst.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueBkg;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSrc;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueMask;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueuedst;
    TBuf<TPosition::VECCALC> calcBufX1;
    TBuf<TPosition::VECCALC> calcBufX2;
    GlobalTensor<T1> x1Gm;
    GlobalTensor<T1> x2Gm;
    GlobalTensor<T2> x3Gm;
    GlobalTensor<T1> zGm;
};

template <typename T1, typename T2>
class KernelBackgroundReplaceC3 {
public:
    __aicore__ inline KernelBackgroundReplaceC3()
    {}
    __aicore__ inline void Init(GM_ADDR bkg, GM_ADDR src, GM_ADDR mask, GM_ADDR out, GM_ADDR workspace, size_t bufferNum, 
                                size_t bufferBytes, size_t gmIdx, size_t gmDataLen)
    {
        if (bufferBytes <= 0) {
            return;
        }
        size_t bkgBufferBytes = bufferBytes * CHANNEL_NUM; // src
        pipe.InitBuffer(inQueueBkg, bufferNum, bkgBufferBytes);
        pipe.InitBuffer(inQueueSrc, bufferNum, bkgBufferBytes);
        pipe.InitBuffer(outQueuedst, bufferNum, bkgBufferBytes);
        if (sizeof(T1) == 1) {
            pipe.InitBuffer(calcBufX1, bkgBufferBytes * sizeof(half));
            pipe.InitBuffer(calcBufX2, bkgBufferBytes * sizeof(half));
            // broadcast mask
            pipe.InitBuffer(inQueueMask, bufferNum, bufferBytes * sizeof(half));
            pipe.InitBuffer(calcBufX3, bkgBufferBytes * sizeof(half));
        } else {
            // broadcast mask
            pipe.InitBuffer(inQueueMask, bufferNum, bufferBytes);
            pipe.InitBuffer(calcBufX3, bkgBufferBytes);
        }

        x1Gm.SetGlobalBuffer((__gm__ T1*)bkg + gmIdx * CHANNEL_NUM, gmDataLen * CHANNEL_NUM);
        x2Gm.SetGlobalBuffer((__gm__ T1*)src + gmIdx * CHANNEL_NUM, gmDataLen * CHANNEL_NUM);
        x3Gm.SetGlobalBuffer((__gm__ T2*)mask + gmIdx, gmDataLen);
        zGm.SetGlobalBuffer((__gm__ T1*)out + gmIdx * CHANNEL_NUM, gmDataLen * CHANNEL_NUM);
    }

    __aicore__ void CalcForAlign32(uint32_t idx, size_t len)
    {
        if (len <= 0) {
            return;
        }
        size_t srclen = len * CHANNEL_NUM;
        // copyIn
        auto bkgLocal = inQueueBkg.AllocTensor<T1>();
        auto srcLocal = inQueueSrc.AllocTensor<T1>();
        auto maskLocal = inQueueMask.AllocTensor<T2>();

        DataCopy(bkgLocal, x1Gm[idx * CHANNEL_NUM], srclen);
        DataCopy(srcLocal, x2Gm[idx * CHANNEL_NUM], srclen);
        DataCopy(maskLocal, x3Gm[idx], len);
        inQueueBkg.EnQue(bkgLocal);
        inQueueSrc.EnQue(srcLocal);
        inQueueMask.EnQue(maskLocal);

        //compute
        bkgLocal = inQueueBkg.DeQue<T1>();
        srcLocal = inQueueSrc.DeQue<T1>();
        maskLocal = inQueueMask.DeQue<T2>();
        const uint32_t dimNum = 2;
        const uint32_t dstShape[dimNum] = {static_cast<uint32_t>(len), CHANNEL_NUM};
        const uint32_t srcShape[dimNum] = {static_cast<uint32_t>(len), 1};
        LocalTensor<half> maskC3Local = calcBufX3.Get<half>();
        auto dstLocal = outQueuedst.AllocTensor<T1>();
        BroadCast<half, dimNum, 1>(maskC3Local, maskLocal, dstShape, srcShape);
        if constexpr(sizeof(T1) == 1) {
            LocalTensor<half> bkgTmpLocal = calcBufX1.Get<half>();
            LocalTensor<half> srcTmpLocal = calcBufX2.Get<half>();
            Cast(bkgTmpLocal, bkgLocal, RoundMode::CAST_NONE, srclen);
            Cast(srcTmpLocal, srcLocal, RoundMode::CAST_NONE, srclen);

            Mul(srcTmpLocal, srcTmpLocal, maskC3Local, srclen);
            Mul(maskC3Local, bkgTmpLocal, maskC3Local, srclen);
            Sub(bkgTmpLocal, bkgTmpLocal, maskC3Local, srclen);
            Add(bkgTmpLocal, bkgTmpLocal, srcTmpLocal, srclen);

            Cast(dstLocal, bkgTmpLocal, RoundMode::CAST_NONE, srclen);
        } else {
            Mul(srcLocal, srcLocal, maskC3Local, srclen);
            Mul(maskC3Local, bkgLocal, maskC3Local, srclen);
            Sub(bkgLocal, bkgLocal, maskC3Local, srclen);
            Add(dstLocal, bkgLocal, srcLocal, srclen);
        }

        //CopyOut
        
        outQueuedst.EnQue(dstLocal);
        inQueueBkg.FreeTensor(bkgLocal);
        inQueueSrc.FreeTensor(srcLocal);
        inQueueMask.FreeTensor(maskLocal);
        dstLocal = outQueuedst.DeQue<T1>();
        DataCopy(zGm[idx * CHANNEL_NUM], dstLocal, srclen);
        outQueuedst.FreeTensor(dstLocal);
    }
protected:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueBkg;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSrc;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueMask;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueuedst;
    TBuf<TPosition::VECCALC> calcBufX1;
    TBuf<TPosition::VECCALC> calcBufX2;
    TBuf<TPosition::VECCALC> calcBufX3;
    GlobalTensor<T1> x1Gm;
    GlobalTensor<T1> x2Gm;
    GlobalTensor<T2> x3Gm;
    GlobalTensor<T1> zGm;
};

template <typename T1, typename T2>
__aicore__ void run_op(GM_ADDR bkg, GM_ADDR src, GM_ADDR mask, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling, float ubVarNum, bool isRGB=false)
{
    GET_TILING_DATA(tilingData, tiling);
    if (!isRGB) {
        VectorScheduler sch(tilingData.size, GetBlockNum(), BUFFER_NUM, ubVarNum, sizeof(T1));
        size_t orgVecIdx = GetBlockIdx() * sch.dataLenPerCore;
        KernelBackgroundReplaceC1<T1, T2> op;
        op.Init(bkg, src, mask, out, workspace, sch.bufferNum, sch.dataBytesPerLoop, orgVecIdx, sch.dataLen);
        sch.run(&op, sch.dataLen);
    } else {
        VectorScheduler sch(tilingData.size, GetBlockNum(), BUFFER_NUM, ubVarNum, sizeof(T1));
        size_t orgVecIdx = GetBlockIdx() * sch.dataLenPerCore;
        KernelBackgroundReplaceC3<T1, T2> op;
        op.Init(bkg, src, mask, out, workspace, sch.bufferNum, sch.dataBytesPerLoop, orgVecIdx, sch.dataLen);
        sch.run(&op, sch.dataLen);
    }
}

extern "C" __global__ __aicore__ void background_replace(GM_ADDR bkg, GM_ADDR src, GM_ADDR mask, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    if (TILING_KEY_IS(1)) {
        constexpr float ubVarNum = 5;
        run_op<half, half>(bkg, src, mask, out, workspace, tiling, ubVarNum);
    } else if (TILING_KEY_IS(2)) {
        constexpr float ubVarNum = 12;
        run_op<uint8_t, half>(bkg, src, mask, out, workspace, tiling, ubVarNum);
    } else if (TILING_KEY_IS(3)) {
        constexpr float ubVarNum = 100;
        run_op<half, half>(bkg, src, mask, out, workspace, tiling, ubVarNum, true);
    } else if (TILING_KEY_IS(4)) {
        constexpr float ubVarNum = 100;
        run_op<uint8_t, half>(bkg, src, mask, out, workspace, tiling, ubVarNum, true);
    }
}
