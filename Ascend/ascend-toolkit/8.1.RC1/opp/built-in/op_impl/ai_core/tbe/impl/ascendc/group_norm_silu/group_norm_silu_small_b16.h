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
 * \file group_norm_silu_small_b16.h
 * \brief
 */
#ifndef GROUP_NORM_SILU_SMALL_B16_H_
#define GROUP_NORM_SILU_SMALL_B16_H_

#include "group_norm_silu_base.h"

namespace GroupNormSilu {
using namespace AscendC;

template <typename T1, typename T2>
class GroupNormSiluSmallB16 : public GroupNormSiluBase<T1> {
public:
    __aicore__ inline GroupNormSiluSmallB16(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR silu, GM_ADDR mean, GM_ADDR rstd,
                                GM_ADDR workspace, const GroupNormSiluTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInX(const int64_t& outIdx, const int64_t& dIdx, const int64_t& innerIdx,
                                   const int64_t& copyNum);
    __aicore__ inline void CopyInGammaAndBeta();
    __aicore__ inline void CastGammaAndBeta();
    __aicore__ inline void ComputeSumX(const int64_t& num);
    __aicore__ inline void CastMeanAndRstd(const int64_t& copyNum);
    __aicore__ inline void ComputeSilu(const int64_t& loopNum);
    __aicore__ inline void ComputeAlign(const int64_t& loopNum);
    __aicore__ inline void ComputeWithCopyOnce(const int64_t &loopNum);
    __aicore__ inline void ComputeOutputAlign(LocalTensor<float>& gammaUb32, LocalTensor<float>& betaUb32,
                                              const int64_t& cIdx, const float& rstd, const float& mean);
    __aicore__ inline void ComputeOutputTail(LocalTensor<float>& gammaUb32, LocalTensor<float>& betaUb32,
                                             const int64_t& cIdx, const float& rstd, const float& mean);
    __aicore__ inline void AccumulateXandX2OneLoop(const int64_t& outIdx);
    __aicore__ inline void AccumulateXandX2MultipleLoop(const int64_t& outIdx);
    __aicore__ inline void NormaLizeX(const int64_t& xIdx, const float& scale, const float& bias);
    __aicore__ inline void CalcSiluAlign(const int64_t& calcNum);
    __aicore__ inline void CopyOutY(const int64_t& outIdx, const int64_t& dIdx, const int64_t& innerIdx,
                                    const int64_t& copyNum);
    __aicore__ inline void CopyOutMeanAndRstd(const int64_t& copyNum);
    __aicore__ inline void ProcessPerCore(const int64_t &loopNum);
    constexpr static int32_t bufferNum = 2;
    constexpr static float negativeOne = -1.0;
    constexpr static float scalarOne = 1.0;
    constexpr static int64_t blockSize = 32;
    constexpr static int64_t elementsPerBlock = blockSize / sizeof(T1);
    constexpr static int64_t elementsPerBlock32 = 8;
    constexpr static int64_t processSize = 8192;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueGamma;
    TQue<QuePosition::VECIN, 1> inQueueBeta;
    TQue<QuePosition::VECOUT, bufferNum> outQueueSilu;
    TQue<QuePosition::VECOUT, 1> outQueueMean;
    TQue<QuePosition::VECOUT, 1> outQueueRstd;

    TBuf<QuePosition::VECCALC> xBuf32;
    TBuf<QuePosition::VECCALC> x1Buf32;
    TBuf<QuePosition::VECCALC> x2Buf32;
    TBuf<QuePosition::VECCALC> gammaBuf32;
    TBuf<QuePosition::VECCALC> betaBuf32;
    TBuf<QuePosition::VECCALC> meanBuf32;
    TBuf<QuePosition::VECCALC> rstdBuf32;
    TBuf<QuePosition::VECCALC> tmpTensor;

    GlobalTensor<T1> xGm;
    GlobalTensor<T2> gammaGm, betaGm;
    GlobalTensor<T1> siluGm, meanGm, rstdGm;

    int32_t blockIdx = 0;
    int64_t blockOffset = 0;
    int64_t gmXOffset = 0;
    float numRec = 0;
    int64_t shapeCAlign = 0;
    int64_t numPerCoreAlign = 0;
    int64_t groupD = 1;  // group of D when data move x
    int64_t loopD = 1;   // loop num for D when data move x
    int64_t dTail = 1;   // tail for D when data move x
    const GroupNormSiluTilingData* tiling;
};

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSmallB16<T1, T2>::Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR silu,
                                                           GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
                                                           const GroupNormSiluTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    tiling = tilingData;
    xGm.SetGlobalBuffer((__gm__ T1 *)x);
    gammaGm.SetGlobalBuffer((__gm__ T2 *)gamma);
    betaGm.SetGlobalBuffer((__gm__ T2 *)beta);
    siluGm.SetGlobalBuffer((__gm__ T1 *)silu);
    meanGm.SetGlobalBuffer((__gm__ T1 *)mean);
    rstdGm.SetGlobalBuffer((__gm__ T1 *)rstd);

    shapeCAlign = this->CeilDiv(tiling->shapeC, elementsPerBlock) * elementsPerBlock;
    numPerCoreAlign = this->CeilDiv(tiling->numPerCore, elementsPerBlock) * elementsPerBlock;

    pipe.InitBuffer(xBuf32, processSize * sizeof(float));
    pipe.InitBuffer(x1Buf32, processSize * sizeof(float));
    pipe.InitBuffer(x2Buf32, processSize * sizeof(float));
    pipe.InitBuffer(inQueueX, bufferNum, processSize * sizeof(T1));
    pipe.InitBuffer(outQueueSilu, bufferNum, processSize * sizeof(T1));
    pipe.InitBuffer(inQueueGamma, 1, shapeCAlign * sizeof(T2));
    pipe.InitBuffer(inQueueBeta, 1, shapeCAlign * sizeof(T2));
    if (sizeof(T2) != sizeof(float)) {
        pipe.InitBuffer(gammaBuf32, shapeCAlign * sizeof(float));
        pipe.InitBuffer(betaBuf32, shapeCAlign * sizeof(float)); 
    }
    pipe.InitBuffer(outQueueMean, 1, numPerCoreAlign * sizeof(T1));
    pipe.InitBuffer(outQueueRstd, 1, numPerCoreAlign * sizeof(T1));
    pipe.InitBuffer(meanBuf32, numPerCoreAlign * sizeof(float));
    pipe.InitBuffer(rstdBuf32, numPerCoreAlign * sizeof(float));
    pipe.InitBuffer(tmpTensor, elementsPerBlock32 * sizeof(float));
    gmXOffset = blockIdx * tiling->numPerCore * tiling->elemNum;
    numRec = float(1.0) / float(tiling->elemNum);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSmallB16<T1, T2>::Process()
{
    if (blockIdx >= tiling->realCoreNum) {
        return;
    }
    if (blockIdx == tiling->realCoreNum - 1) {  // process last core
        ProcessPerCore(tiling->numLastCore);
    } else {
        ProcessPerCore(tiling->numPerCore);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSmallB16<T1, T2>::ProcessPerCore(const int64_t &loopNum)
{
    CopyInGammaAndBeta();
    if (sizeof(T2) != sizeof(float)) {
        CastGammaAndBeta();
    }
    ComputeSilu(loopNum);
    CastMeanAndRstd(loopNum);
    CopyOutMeanAndRstd(loopNum);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSmallB16<T1, T2>::CopyInGammaAndBeta()
{
    LocalTensor<T2> gamma = inQueueGamma.AllocTensor<T2>();
    this->template CopyInData<T2, false>(gamma, gammaGm, tiling->shapeC);
    inQueueGamma.EnQue(gamma);

    LocalTensor<T2> beta = inQueueBeta.AllocTensor<T2>();
    this->template CopyInData<T2, false>(beta, betaGm, tiling->shapeC);
    inQueueBeta.EnQue(beta);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSmallB16<T1, T2>::CastGammaAndBeta()
{
    LocalTensor<T2> gammaUb = inQueueGamma.DeQue<T2>();
    LocalTensor<T2> betaUb = inQueueBeta.DeQue<T2>();
    Cast(gammaBuf32.Get<float>(), gammaUb, RoundMode::CAST_NONE, tiling->shapeC);
    Cast(betaBuf32.Get<float>(), betaUb, RoundMode::CAST_NONE, tiling->shapeC);
    inQueueGamma.FreeTensor(gammaUb);
    inQueueBeta.FreeTensor(betaUb);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSmallB16<T1, T2>::ComputeSilu(const int64_t &loopNum)
{
    if (tiling->elemNum <= processSize) {
        ComputeWithCopyOnce(loopNum);
    } else {
        groupD = processSize / tiling->hwNum;
        groupD = groupD < 1 ? 1 : groupD;
        groupD = groupD > tiling->shapeD ? tiling->shapeD : groupD;
        loopD = this->CeilDiv(tiling->shapeD, groupD);
        dTail = tiling->shapeD - (loopD - 1) * groupD;
        ComputeAlign(loopNum);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSmallB16<T1, T2>::ComputeWithCopyOnce(const int64_t &loopNum)
{
    LocalTensor<float> gammaUb32;
    LocalTensor<float> betaUb32;
    if (sizeof(T2) != sizeof(float)) {
        gammaUb32 = gammaBuf32.Get<float>();
        betaUb32 = betaBuf32.Get<float>();
    } else {
        gammaUb32 = inQueueGamma.DeQue<float>();
        betaUb32 = inQueueBeta.DeQue<float>();
    }
    for (int64_t outIdx = 0; outIdx < loopNum; outIdx++) {
        AccumulateXandX2OneLoop(outIdx);
        // calc var and rstd;
        float mean = tmpTensor.Get<float>().GetValue(0);
        float rstd = x2Buf32.Get<float>().GetValue(0);
        mean = mean * numRec;
        rstd = float(1.0) / sqrt(rstd * numRec - mean * mean + tiling->epsilon);
        meanBuf32.Get<float>().SetValue(outIdx, mean);
        rstdBuf32.Get<float>().SetValue(outIdx, rstd);
        event_t eventS2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventS2V);
        WaitFlag<HardEvent::S_V>(eventS2V);
        // normalize x and calc silu
        int64_t groupIdx = ((blockIdx * tiling->numPerCore + outIdx) % tiling->numGroups);
        LocalTensor<T1> outSilu = outQueueSilu.AllocTensor<T1>();
        for (int64_t dIdx = 0; dIdx < tiling->shapeD; dIdx++) {
            int64_t cIdx = groupIdx * tiling->shapeD + dIdx;
            float gamma = gammaUb32.GetValue(cIdx);
            float beta = betaUb32.GetValue(cIdx);
            float scale = rstd * gamma;
            float bias = -scale * mean + beta;
            NormaLizeX(dIdx * tiling->hwNum, scale, bias);
        }
        CalcSiluAlign(tiling->loopTail);
        Cast(outSilu, x1Buf32.Get<float>(), this->GetRoundMode(), tiling->loopTail);
        pipe_barrier(PIPE_V);
        outQueueSilu.EnQue(outSilu);
        CopyOutY(outIdx, 0, 0, tiling->loopTail);
    }
    if (sizeof(T2) == sizeof(float)) {
        inQueueGamma.FreeTensor(gammaUb32);
        inQueueBeta.FreeTensor(betaUb32);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSmallB16<T1, T2>::ComputeAlign(const int64_t &loopNum)
{
    LocalTensor<float> gammaUb32;
    LocalTensor<float> betaUb32;
    if (sizeof(T2) != sizeof(float)) {
        gammaUb32 = gammaBuf32.Get<float>();
        betaUb32 = betaBuf32.Get<float>();
    } else {
        gammaUb32 = inQueueGamma.DeQue<float>();
        betaUb32 = inQueueBeta.DeQue<float>();
    }
    for (int64_t outIdx = 0; outIdx < loopNum; outIdx++) {
        AccumulateXandX2MultipleLoop(outIdx);
        // calc var and rstd;
        float mean = tmpTensor.Get<float>().GetValue(0);
        float rstd = x2Buf32.Get<float>().GetValue(0);
        mean = mean * numRec;
        rstd = float(1.0) / sqrt(rstd * numRec - mean * mean + tiling->epsilon);
        meanBuf32.Get<float>().SetValue(outIdx, mean);
        rstdBuf32.Get<float>().SetValue(outIdx, rstd);
        // normalize x and calc silu
        int64_t groupIdx = ((blockIdx * tiling->numPerCore + outIdx) % tiling->numGroups);
        for (int64_t dIdx = 0; dIdx < loopD; dIdx++) {
            int64_t cIdx = groupIdx * tiling->shapeD + dIdx * groupD;
            if (dIdx != loopD - 1) {
                CopyInX(outIdx, dIdx * groupD, 0, tiling->hwNum * groupD);
                ComputeOutputAlign(gammaUb32, betaUb32, cIdx, rstd, mean);
                CopyOutY(outIdx, dIdx * groupD, 0, tiling->hwNum * groupD);
            } else {
                CopyInX(outIdx, dIdx * groupD, 0, tiling->hwNum * dTail);
                ComputeOutputTail(gammaUb32, betaUb32, cIdx, rstd, mean);
                CopyOutY(outIdx, dIdx * groupD, 0, tiling->hwNum * dTail);
            }
        }
    }
    if (sizeof(T2) == sizeof(float)) {
        inQueueGamma.FreeTensor(gammaUb32);
        inQueueBeta.FreeTensor(betaUb32);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSmallB16<T1, T2>::AccumulateXandX2OneLoop(const int64_t &outIdx)
{
    LocalTensor<float> xUb32 = xBuf32.Get<float>();
    LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();
    LocalTensor<float> x2Ub32 = x2Buf32.Get<float>();
    LocalTensor<float> tmpMean = tmpTensor.Get<float>();

    CopyInX(outIdx, 0, 0, tiling->loopTail);
    LocalTensor<T1> xUb = inQueueX.DeQue<T1>();
    Cast(x1Ub32, xUb, RoundMode::CAST_NONE, tiling->loopTail);
    pipe_barrier(PIPE_V);
    inQueueX.FreeTensor(xUb);
    Mul(x2Ub32, x1Ub32, x1Ub32, tiling->loopTail);

    // accumulate x and x^2
    ReduceSum<float>(tmpMean, x1Ub32, xUb32, tiling->loopTail);
    ReduceSum<float>(x2Ub32, x2Ub32, xUb32, tiling->loopTail);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSmallB16<T1, T2>::AccumulateXandX2MultipleLoop(const int64_t &outIdx)
{
    LocalTensor<float> xUb32 = xBuf32.Get<float>();
    LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();
    LocalTensor<float> x2Ub32 = x2Buf32.Get<float>();
    LocalTensor<float> tmpMean = tmpTensor.Get<float>();

    // process the first loop
    CopyInX(outIdx, 0, 0, processSize);
    LocalTensor<T1> xUb = inQueueX.DeQue<T1>();
    Cast(x1Ub32, xUb, RoundMode::CAST_NONE, processSize);
    inQueueX.FreeTensor(xUb);
    Mul(x2Ub32, x1Ub32, x1Ub32, processSize);
    // process the middle loops
    for (int64_t innerIdx = 1; innerIdx < tiling->loopNum - 1; innerIdx++) {
        CopyInX(outIdx, 0, innerIdx, processSize);
        ComputeSumX(processSize);
    }
    // process the last loop
    if (tiling->loopNum > 1) {  // process tail data
        CopyInX(outIdx, 0, tiling->loopNum - 1, tiling->loopTail);
        ComputeSumX(tiling->loopTail);
    }
    // accumulate x and x^2
    ReduceSum<float>(tmpMean, x1Ub32, xUb32, processSize);
    ReduceSum<float>(x2Ub32, x2Ub32, xUb32, processSize);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSmallB16<T1, T2>::CopyInX(const int64_t& outIdx, const int64_t& dIdx,
                                                              const int64_t& innerIdx, const int64_t& copyNum)
{
    LocalTensor<T1> xUb = inQueueX.AllocTensor<T1>();
    DataCopy(xUb, xGm[gmXOffset + outIdx * tiling->elemNum + dIdx * tiling->hwNum + innerIdx * processSize], copyNum);
    inQueueX.EnQue(xUb);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSmallB16<T1, T2>::ComputeSumX(const int64_t &num)
{
    LocalTensor<T1> xUb = inQueueX.DeQue<T1>();
    LocalTensor<float> xUb32 = xBuf32.Get<float>();
    LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();
    LocalTensor<float> x2Ub32 = x2Buf32.Get<float>();

    Cast(xUb32, xUb, RoundMode::CAST_NONE, num);
    inQueueX.FreeTensor(xUb);
    Add(x1Ub32, x1Ub32, xUb32, num);
    Mul(xUb32, xUb32, xUb32, num);
    Add(x2Ub32, x2Ub32, xUb32, num);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSmallB16<T1, T2>::NormaLizeX(const int64_t& xIdx,
                                                                 const float& scale,
                                                                 const float& bias)
{
    LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();
    LocalTensor<float> x2Ub32 = x2Buf32.Get<float>();
    // normalize x
    Muls(x1Ub32[xIdx], x1Ub32[xIdx], scale, tiling->hwNum);
    pipe_barrier(PIPE_V);
    Adds(x1Ub32[xIdx], x1Ub32[xIdx], bias, tiling->hwNum);
    pipe_barrier(PIPE_V);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSmallB16<T1, T2>::CalcSiluAlign(const int64_t& calcNum)
{
    LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();
    LocalTensor<float> x2Ub32 = x2Buf32.Get<float>();
    // calc silu
    if (tiling->activateSilu) {
        Muls(x2Ub32, x1Ub32, negativeOne, calcNum);
        pipe_barrier(PIPE_V);
        Exp(x2Ub32, x2Ub32, calcNum);
        pipe_barrier(PIPE_V);
        Adds(x2Ub32, x2Ub32, scalarOne, calcNum);
        pipe_barrier(PIPE_V);
        Div(x1Ub32, x1Ub32, x2Ub32, calcNum);
        pipe_barrier(PIPE_V);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSmallB16<T1, T2>::ComputeOutputAlign(LocalTensor<float>& gammaUb32,
                                                                         LocalTensor<float>& betaUb32,
                                                                         const int64_t& cIdx,
                                                                         const float& rstd, const float& mean)
{
    LocalTensor<T1> xUb = inQueueX.DeQue<T1>();
    LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();
    LocalTensor<T1> outSilu = outQueueSilu.AllocTensor<T1>();
    LocalTensor<float> x2Ub32 = x2Buf32.Get<float>();

    Cast(x1Ub32, xUb, RoundMode::CAST_NONE, tiling->hwNum * groupD);
    pipe_barrier(PIPE_V);
    inQueueX.FreeTensor(xUb);
    for (int64_t gIdx = 0; gIdx < groupD; gIdx++) {
        float gamma = gammaUb32.GetValue(cIdx + gIdx);
        float beta = betaUb32.GetValue(cIdx + gIdx);
        float scale = rstd * gamma;
        float bias = -scale * mean + beta;
        NormaLizeX(gIdx * tiling->hwNum, scale, bias);
    }
    CalcSiluAlign(tiling->hwNum * groupD);
    Cast(outSilu, x1Ub32, this->GetRoundMode(), tiling->hwNum * groupD);
    pipe_barrier(PIPE_V);
    outQueueSilu.EnQue(outSilu);  
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSmallB16<T1, T2>::ComputeOutputTail(LocalTensor<float>& gammaUb32,
                                                                        LocalTensor<float>& betaUb32,
                                                                        const int64_t& cIdx,
                                                                        const float& rstd, const float& mean)
{
    LocalTensor<T1> xUb = inQueueX.DeQue<T1>();
    LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();
    LocalTensor<T1> outSilu = outQueueSilu.AllocTensor<T1>();
    LocalTensor<float> x2Ub32 = x2Buf32.Get<float>();

    Cast(x1Ub32, xUb, RoundMode::CAST_NONE, tiling->hwNum * dTail);
    pipe_barrier(PIPE_V);
    inQueueX.FreeTensor(xUb);
    for (int64_t gIdx = 0; gIdx < dTail; gIdx++) {
        float gamma = gammaUb32.GetValue(cIdx + gIdx);
        float beta = betaUb32.GetValue(cIdx + gIdx);
        float scale = rstd * gamma;
        float bias = -scale * mean + beta;
        NormaLizeX(gIdx * tiling->hwNum, scale, bias);
    }
    CalcSiluAlign(tiling->hwNum * dTail);
    Cast(outSilu, x1Ub32, this->GetRoundMode(), tiling->hwNum * dTail);
    pipe_barrier(PIPE_V);
    outQueueSilu.EnQue(outSilu);  
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSmallB16<T1, T2>::CopyOutY(const int64_t& outIdx, const int64_t& dIdx,
                                                               const int64_t& innerIdx, const int64_t& copyNum)
{
    LocalTensor<T1> outSilu = outQueueSilu.DeQue<T1>();
    DataCopy(siluGm[gmXOffset + outIdx * tiling->elemNum + dIdx * tiling->hwNum +
                    innerIdx * processSize], outSilu, copyNum);
    outQueueSilu.FreeTensor(outSilu);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSmallB16<T1, T2>::CastMeanAndRstd(const int64_t& copyNum)
{
    LocalTensor<float> meanUb = meanBuf32.Get<float>();
    LocalTensor<float> rstdUb = rstdBuf32.Get<float>();

    LocalTensor<T1> meanOut = outQueueMean.AllocTensor<T1>();
    Cast(meanOut, meanUb, this->GetRoundMode(), copyNum);
    outQueueMean.EnQue(meanOut);
    LocalTensor<T1> rstdOut = outQueueRstd.AllocTensor<T1>();
    Cast(rstdOut, rstdUb, this->GetRoundMode(), copyNum);
    outQueueRstd.EnQue(rstdOut);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSmallB16<T1, T2>::CopyOutMeanAndRstd(const int64_t& copyNum)
{
    LocalTensor<T1> meanOut = outQueueMean.DeQue<T1>();
    LocalTensor<T1> rstdOut = outQueueRstd.DeQue<T1>();
#if __CCE_AICORE__ == 220
    // when support DataCopyPad, use DataCopyPad
    uint16_t dataCount = static_cast<uint16_t>(copyNum);
    uint16_t blockCount = 1;
    uint16_t blockLen = dataCount * sizeof(T1);
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    DataCopyParams dataCopyParams{blockCount, blockLen, srcStride, dstStride};
    DataCopyPad(meanGm[blockIdx * tiling->numPerCore], meanOut, dataCopyParams);
    DataCopyPad(rstdGm[blockIdx * tiling->numPerCore], rstdOut, dataCopyParams);
#endif
    outQueueMean.FreeTensor(meanOut);
    outQueueRstd.FreeTensor(rstdOut);
}
}  // namespace GroupNormSilu

#endif  // GROUP_NORM_SILU_SMALL_B16_H_