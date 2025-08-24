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
 * \file apply_came_part4_float32.h
 * \brief
 */
#ifndef _APPLY_CAME_PART4_FLOAT32_H_
#define _APPLY_CAME_PART4_FLOAT32_H_

#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class ApplyCamePart4Float32 {
public:
    __aicore__ inline ApplyCamePart4Float32() {}
    __aicore__ inline void Init(GM_ADDR paramIn, GM_ADDR m, GM_ADDR rIn, GM_ADDR cIn, 
                                GM_ADDR weight_decay, GM_ADDR lr, GM_ADDR beta3, GM_ADDR sum_r, 
                                GM_ADDR sum_u_r, GM_ADDR sum_u_c, GM_ADDR sum_u_rc, GM_ADDR global_shape,
                                GM_ADDR paramOut, GM_ADDR rOut, GM_ADDR cOut,
                                GM_ADDR workspace, const ApplyCamePart4TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const ApplyCamePart4TilingData* tilingData);
    __aicore__ inline void ProcessR();
    __aicore__ inline void CopyInR(int iter, int64_t rNumPerLoop);
    __aicore__ inline void CopyInSumur(int iter, int64_t rNumPerLoop);
    __aicore__ inline void CopyInSumuc(int iter, int64_t rNumPerLoop);
    __aicore__ inline void ComputeR(int64_t NumPerLoop);
    __aicore__ inline void CopyOutR(int64_t iter, int64_t rNumPerLoop);
    __aicore__ inline void ProcessC();
    __aicore__ inline void CopyInC(int iter, int64_t rNumPerLoop);
    __aicore__ inline void ComputeC(int64_t NumPerLoop);
    __aicore__ inline void CopyOutC(int64_t iter, int64_t rNumPerLoop);
    __aicore__ inline void CalcScalarInput();

    __aicore__ inline void ProcessParam();
    __aicore__ inline void InitProcessParam();
    __aicore__ inline void InitForCalcParam();
    __aicore__ inline void ProcessPerCoreParam();
    __aicore__ inline void CopyInParamr(int loopIdx, int64_t dataCount);
    __aicore__ inline void CopyInParam(int rLoopIdx, int cLoopIdx, int64_t curRNumInLoop, int64_t curCNumInLoop);
    __aicore__ inline void CalcParam(int rLoopIdx, int cLoopIdx, int64_t curRNumInLoop, int64_t curCNumInLoop);
    __aicore__ inline void CopyInParamc(int loopIdx, int64_t dataCount);
    __aicore__ inline void ComputeParam(int64_t curRNumInLoop, int64_t curCNumInLoop);
    __aicore__ inline void CopyOutParam(int rLoopIdx, int cLoopIdx, int64_t curRNumInLoop, int64_t curCNumInLoop);
    __aicore__ inline void BroadcastR(LocalTensor<T> &dst, LocalTensor<T> &src, int64_t numR, int64_t numC);
    __aicore__ inline void TransposeR(LocalTensor<T> &dst, LocalTensor<T> &src, int64_t numR, int64_t numC);
    __aicore__ inline void MulRC(LocalTensor<T> &dst, LocalTensor<T> &r, LocalTensor<T> &c, int64_t numR, int64_t numC);
    __aicore__ inline void CopyInNotAlignedParamc(int loopIdx, int64_t dataCount);
    __aicore__ inline void ProcessTailCoreParam();
    __aicore__ inline void CopyInNotAlignedParamr(int loopIdx, int64_t dataCount);
    __aicore__ inline void CopyInm(int rLoopIdx, int cLoopIdx, int64_t curRNumInLoop, int64_t curCNumInLoop);
    __aicore__ inline void CalcRcCycleMode(LocalTensor<T> &dst, LocalTensor<T> &src,
                                           LocalTensor<T> &srcScalar,
                                           int64_t curRNumInLoop, int64_t curCNumInLoop);
    __aicore__ inline void CopyInNotAlignedParam(int rLoopIdx, int cLoopIdx,
                                                int64_t curRNumInLoop, int64_t curCNumInLoop);
    __aicore__ inline void CopyInNotAlignedm(int rLoopIdx, int cLoopIdx,
                                            int64_t curRNumInLoop, int64_t curCNumInLoop);

    template <typename T1, typename T2> __aicore__ inline T1 CeilDiv(T1 a, T2 b) {
        if (b == 0) {
            return 0;
        }

        return (a + b - 1) / b;
    };

    template <typename T1, typename T2> __aicore__ inline T1 Max(T1 a, T2 b) {
        return a > b ? a : b;
    };

private:
    __aicore__ inline void GetConfusionTransposeTiling(int64_t numR, int64_t numC, const uint32_t stackBufferSize,
                                                       const uint32_t typeSize, ConfusionTransposeTiling &tiling);

private:
    TPipe pipe_;
    // VECIN
    TQue<QuePosition::VECIN, 1> rcInQue_;     // set ub for r & c
    TQue<QuePosition::VECIN, 1> sumurcQue_;   // set ub for sum_u_r & sum_u_c
    // VECOUT
    TQue<QuePosition::VECOUT, 1> rcOutQue_;
    TQue<QuePosition::VECOUT, 1> paramOutQue_;

    TBuf<QuePosition::VECCALC> scalarBuf_; //  save scalar input

    TQue<QuePosition::VECIN, 1> inQueR_;
    TQue<QuePosition::VECIN, 1> inQueC_;
    TQue<QuePosition::VECIN, 1> inQuem_;
    TQue<QuePosition::VECIN, 1> inQueScalar_; // all scalar input
    TQue<QuePosition::VECIN, 1> inQueParam_; // shape: (n, m)

    TQue<QuePosition::VECOUT, 1> outQueParam_; // shape: (n, m)
    TQue<QuePosition::VECOUT, 1> outQueSumR_; // scalar

    // each tbuf may be used for multiple purposes
    TBuf<QuePosition::VECCALC> ub1Buf_; // 1. rMatrix after broadcast from r
    TBuf<QuePosition::VECCALC> ub2Buf_; // 1. rMatrix after transpose; 2. r * c
    TBuf<QuePosition::VECCALC> ub3Buf_; // grad

    TQue<QuePosition::VECIN, 1> reduceSumWorkQueue_;

    // multi-core sync
    GlobalTensor<int32_t> syncGlobal_;
    TQue<QuePosition::VECIN, 1> syncWorkQueue_;

    GlobalTensor<T> rInGm_;
    GlobalTensor<T> rOutGm_;
    GlobalTensor<T> cInGm_;
    GlobalTensor<T> cOutGm_;
    GlobalTensor<T> sumurGm_;
    GlobalTensor<T> sumucGm_;

    GlobalTensor<T> mGm_;
    GlobalTensor<T> paramInGm_;
    GlobalTensor<T> paramOutGm_;

    GlobalTensor<T> sumRWorkspace_;

    // scalar gm
    GlobalTensor<T> weightDecayGm_;
    GlobalTensor<T> lrGm_;
    GlobalTensor<T> beta3Gm_;
    GlobalTensor<T> sumurcGm_;
    GlobalTensor<T> sumRGm_;
    GlobalTensor<int64_t> globalShapeGm_;
    
    float beta3_;
    float sumR_;
    float sum_u_rc_;
    float N_;
    float M_;
    float lr_;
    float weight_decay_;
    T rcCoefficient_;

    int64_t blockIdx_;

    GM_ADDR rIn_;
    GM_ADDR rOut_;
    GM_ADDR sum_u_r_;
    GM_ADDR cIn_;
    GM_ADDR cOut_;
    GM_ADDR sum_u_c_;
    GM_ADDR sum_r_;
    GM_ADDR global_shape_;

    LocalTensor<T> rRcLocalTensor_;
    
    int64_t workLocalNeedSize_;

    int64_t rNumPerCore_;             // 非尾核处理的r的个数
    int64_t cNumPerCore_;

    int64_t rCoreNumToUse_;           // r使用的核数
    int64_t rLoopCount_;              // 非尾核，r的loops数
    int64_t rNumPerLoop_;             // 非尾核，r的每次loop处理的元素个数
    int64_t rNumTailPerLoop_;         // 尾核，r的每次loop处理的个数
    int64_t rLoopCountTailCore_;      // 尾核，r的loop数
    int64_t rNumTailLoopLast_;
    int64_t cCoreNumToUse_;           // c使用的核数
    int64_t cLoopCount_;              // 非尾核，c的loops数
    int64_t cNumPerLoop_;             // 非尾核，c的每次loop处理的元素个数
    int64_t cNumTailPerLoop_;         // 尾核，c的每次loop处理的个数
    int64_t cLoopCountTailCore_;      // 尾核，c的loop数
    int64_t cNumTailLoopLast_;
    int64_t rRcNumTailLoopTailCore_;

    int64_t rRcCoreNumToUse_;
    int64_t rRcNumPerCore_;
    int64_t rRcLoopCount_;
    int64_t rRcLoopCountTailCore_;
    int64_t rRcNumPerLoop_;
    int64_t rRcNumTailLoop_;
    int64_t cRcLoopCount_;
    int64_t cRcNumPerLoop_;
    int64_t cRcNumTailLoop_;
    int64_t mShape_;
    int64_t nShape_;
    int64_t rRcNumOnTailCore_;
    int64_t handleMax_;               // 计算r/c时，最大处理元素个数
    int64_t totalCoreNum_;

    // block offset
    int64_t rBlockOffset_;
    int64_t cBlockOffset_;
    int64_t rRcBlockOffset_;

    const int64_t NUM_PER_BLOCK = ONE_BLK_SIZE / sizeof(T);
    const int64_t INT64_PER_BLOCK = ONE_BLK_SIZE / sizeof(int64_t);
    const int32_t CONFUSION_TRANSPOSE_ALIGHNED_NUM = 16;
    const int32_t CALC_SIZE  = 256;
    const int32_t HALF_CALC_SIZE = CALC_SIZE / 2;
};

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::Init(GM_ADDR paramIn, GM_ADDR m, GM_ADDR rIn, GM_ADDR cIn, 
                                                      GM_ADDR weight_decay, GM_ADDR lr, GM_ADDR beta3, GM_ADDR sum_r, 
                                                      GM_ADDR sum_u_r, GM_ADDR sum_u_c, GM_ADDR sum_u_rc, GM_ADDR global_shape,
                                                      GM_ADDR paramOut, GM_ADDR rOut, GM_ADDR cOut,
                                                      GM_ADDR workspace, const ApplyCamePart4TilingData* tilingData)
{
    // init tiling data
    ParseTilingData(tilingData);

    rInGm_.SetGlobalBuffer((__gm__ T*)rIn + GetBlockIdx() * rNumPerCore_);
    cInGm_.SetGlobalBuffer((__gm__ T*)cIn + GetBlockIdx() * cNumPerCore_);
    sumurGm_.SetGlobalBuffer((__gm__ T*)sum_u_r + GetBlockIdx() * rNumPerCore_);
    sumucGm_.SetGlobalBuffer((__gm__ T*)sum_u_c + GetBlockIdx() * cNumPerCore_);
    rOutGm_.SetGlobalBuffer((__gm__ T*)rOut + GetBlockIdx() * rNumPerCore_);
    cOutGm_.SetGlobalBuffer((__gm__ T*)cOut + GetBlockIdx() * cNumPerCore_);

    // scalar input
    weightDecayGm_.SetGlobalBuffer((__gm__ T*)weight_decay);
    lrGm_.SetGlobalBuffer((__gm__ T*)lr);
    beta3Gm_.SetGlobalBuffer((__gm__ T*)beta3);
    sumurcGm_.SetGlobalBuffer((__gm__ T*)sum_u_rc);
    globalShapeGm_.SetGlobalBuffer((__gm__ int64_t*)global_shape);
    mGm_.SetGlobalBuffer((__gm__ T*)m);
    paramInGm_.SetGlobalBuffer((__gm__ T*)paramIn);
    paramOutGm_.SetGlobalBuffer((__gm__ T*)paramOut);

    sumRWorkspace_.SetGlobalBuffer((__gm__ T*)workspace + (totalCoreNum_ * 32) / sizeof(T));

    // init 
    int64_t rBufferLength = (rNumPerLoop_ > rRcNumPerLoop_) ? rNumPerLoop_ : rRcNumPerLoop_;
    int64_t cBufferLength = (cNumPerLoop_ > cRcNumPerLoop_) ? cNumPerLoop_ : cRcNumPerLoop_;
    rBufferLength = NUM_PER_BLOCK * CeilDiv(rBufferLength, NUM_PER_BLOCK);
    cBufferLength = NUM_PER_BLOCK * CeilDiv(cBufferLength, NUM_PER_BLOCK);
    // init ub for r & c
    pipe_.InitBuffer(rcInQue_, 1, (handleMax_ * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE);
    // init ub for sum_u_r & sum_u_c
    pipe_.InitBuffer(sumurcQue_, 1, (handleMax_ * sizeof(float) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE);
    // init output for r & c
    pipe_.InitBuffer(rcOutQue_, 1, (handleMax_ * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE);
    pipe_.InitBuffer(inQueR_, 1, rBufferLength * sizeof(T));
    pipe_.InitBuffer(inQueC_, 1, cBufferLength * sizeof(T));

    // save GM_ADDR
    rIn_ = rIn;
    rOut_ = rOut;
    sum_u_r_ = sum_u_r;
    cIn_ = cIn;
    cOut_ = cOut;
    sum_u_c_ = sum_u_c;
    global_shape_ = global_shape;
    sum_r_ = sum_r;
    blockIdx_ = GetBlockIdx();
    // copy scalar value
    CalcScalarInput();
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::CalcScalarInput()
{
    pipe_.InitBuffer(scalarBuf_, ONE_BLK_SIZE);
    // copy weight_decay
    LocalTensor<T> inputLocal = scalarBuf_.Get<T>();
    DataCopyPad(inputLocal, weightDecayGm_, {1, sizeof(float), 0, 0, 0}, {false, 0, 0, 0});
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    weight_decay_ = inputLocal.GetValue(0);
    // copy lr
    DataCopyPad(inputLocal, lrGm_, {1, sizeof(float), 0, 0, 0}, {false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    lr_ = inputLocal.GetValue(0);
    // copy beta3
    DataCopyPad(inputLocal, beta3Gm_, {1, sizeof(float), 0, 0, 0}, {false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    beta3_ = inputLocal.GetValue(0);
    // copy sum_u_rc
    DataCopyPad(inputLocal, sumurcGm_, {1, sizeof(float), 0, 0, 0}, {false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    sum_u_rc_ = inputLocal.GetValue(0);
    // copy sum_r
    if (sum_r_ != nullptr) {
        sumRGm_.SetGlobalBuffer((__gm__ T*)sum_r_);
        DataCopyPad(inputLocal, sumRGm_, {1, sizeof(float), 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        sumR_ = inputLocal.GetValue(0);
    }
    // copy global_shape
    if (global_shape_ != nullptr) {
        LocalTensor<int64_t> int64Local = scalarBuf_.Get<int64_t>();
        DataCopy(int64Local, globalShapeGm_, INT64_PER_BLOCK);
        PipeBarrier<PIPE_ALL>();
        LocalTensor<float> floatLocal = scalarBuf_.Get<float>();
        Cast(floatLocal, int64Local, RoundMode::CAST_ROUND, INT64_PER_BLOCK);
        PipeBarrier<PIPE_ALL>();
        N_ = floatLocal.GetValue(0);
        M_ = floatLocal.GetValue(1);
    } else {
        LocalTensor<int64_t> int64Local = scalarBuf_.Get<int64_t>();
        int64Local.SetValue(0, nShape_);
        int64Local.SetValue(1, mShape_);
        PipeBarrier<PIPE_ALL>();
        LocalTensor<float> floatLocal = scalarBuf_.Get<float>();
        Cast(floatLocal, int64Local, RoundMode::CAST_ROUND, INT64_PER_BLOCK);
        PipeBarrier<PIPE_ALL>();
        N_ = floatLocal.GetValue(0);
        M_ = floatLocal.GetValue(1);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::ParseTilingData(const ApplyCamePart4TilingData* tilingData)
{
    mShape_ = tilingData->m;
    nShape_ = tilingData->n;
    rRcNumPerCore_ = tilingData->rRcNumPerCore;
    rRcCoreNumToUse_ = tilingData->rRcCoreNumToUse;
    rRcNumOnTailCore_ = tilingData->rRcNumOnTailCore;
    rLoopCount_ = tilingData->rLoopCount;
    rNumPerLoop_ = tilingData->rNumPerLoop;
    rLoopCountTailCore_ = tilingData->rLoopCountTailCore;
    cLoopCount_ = tilingData->cLoopCount;
    cNumPerLoop_ = tilingData->cNumPerLoop;
    cLoopCountTailCore_ = tilingData->cLoopCountTailCore;
    rRcLoopCount_ = tilingData->rRcLoopCount;
    rRcNumPerLoop_ = tilingData->rRcNumPerLoop;
    rRcLoopCountTailCore_ = tilingData->rRcLoopCountTailCore;
    rRcNumTailLoop_ = tilingData->rRcNumTailLoop;
    rRcNumTailLoopTailCore_ = tilingData->rRcNumTailLoopTailCore;
    cRcLoopCount_ = tilingData->cRcLoopCount;
    cRcNumPerLoop_ = tilingData->cRcNumPerLoop;
    cRcNumTailLoop_ = tilingData->cRcNumTailLoop;
    totalCoreNum_ = tilingData->totalCoreNum;
    handleMax_ = tilingData->handleMax;
    rNumPerCore_ = tilingData->rNumPerCore;
    rCoreNumToUse_ = tilingData->rCoreNumToUse;
    rNumTailPerLoop_ = tilingData->rNumTailPerLoop;
    rNumTailLoopLast_ = tilingData->rNumTailLoopLast;
    cNumPerCore_ = tilingData->cNumPerCore;
    cCoreNumToUse_ = tilingData->cCoreNumToUse;
    cNumTailPerLoop_ = tilingData->cNumTailPerLoop;
    cNumTailLoopLast_ = tilingData->cNumTailLoopLast;
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::Process()
{
    if (g_coreType == AIC) {
        return;
    }

    ProcessR();

    ProcessC();
    
    // multi-core sync
    SyncAll();

    ProcessParam();
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::ProcessR()
{
    if (GetBlockIdx() >= rCoreNumToUse_) {
        return;
    }

    // main core 
    if (GetBlockIdx() != rCoreNumToUse_ - 1) {
        for (int64_t i = 0; i < rLoopCount_; i++) {
            CopyInR(i, rNumPerLoop_);
            CopyInSumur(i, rNumPerLoop_);
            ComputeR(rNumPerLoop_);
            CopyOutR(i, rNumPerLoop_);
        }
    // tail core
    } else {
        for (int64_t i = 0; i < rLoopCountTailCore_; i++) {
            CopyInR(i, rNumTailPerLoop_);
            CopyInSumur(i, rNumTailPerLoop_);
            ComputeR(rNumTailPerLoop_);
            CopyOutR(i, rNumTailPerLoop_);
        }
        // hanlde case smaller than ubSize separately
        if (rNumTailLoopLast_ != 0) {
            rInGm_.SetGlobalBuffer((__gm__ T*)rIn_ + rLoopCountTailCore_ * rNumTailPerLoop_);
            rOutGm_.SetGlobalBuffer((__gm__ T*)rOut_ + rLoopCountTailCore_ * rNumTailPerLoop_);
            sumurGm_.SetGlobalBuffer((__gm__ T*)sum_u_r_ + rLoopCountTailCore_ * rNumTailPerLoop_);
            CopyInR(0, rNumTailLoopLast_);
            CopyInSumur(0, rNumTailLoopLast_);
            ComputeR(rNumTailLoopLast_);
            CopyOutR(0, rNumTailLoopLast_);
        }
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::CopyInR(int iter, int64_t rNumPerLoop)
{
    // copy r gm -> ub
    LocalTensor<T> rInput = rcInQue_.AllocTensor<T>();
    int64_t ONCE_COPY_NUM = ONE_BLK_SIZE / sizeof(T);
    DataCopy(rInput, rInGm_[iter * rNumPerLoop], (rNumPerLoop + ONCE_COPY_NUM - 1) / ONCE_COPY_NUM * ONCE_COPY_NUM);
    rcInQue_.EnQue(rInput);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::CopyInSumur(int iter, int64_t rNumPerLoop)
{
    // copy sum_u_r/sum_u_c gm -> ub
    LocalTensor<T> sumurInput = sumurcQue_.AllocTensor<T>();
    int64_t ONCE_COPY_NUM = ONE_BLK_SIZE / sizeof(T);
    DataCopy(sumurInput, sumurGm_[iter * rNumPerLoop], (rNumPerLoop + ONCE_COPY_NUM - 1) / ONCE_COPY_NUM * ONCE_COPY_NUM);
    sumurcQue_.EnQue(sumurInput);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::ComputeR(int64_t rNumPerLoop)
{
    LocalTensor<T> rInput = rcInQue_.DeQue<T>();
    LocalTensor<T> sumurInput = sumurcQue_.DeQue<T>();
    LocalTensor<T> Output = rcOutQue_.AllocTensor<T>();

    // calc: beta3 * r/c
    Muls(Output, rInput, beta3_, rNumPerLoop);
    PipeBarrier<PIPE_V>();
    // calc: beta3 * r + (1 - beta3) * sum_u_r
    float scalarVal = (1 - beta3_) / M_;
    Axpy(Output, sumurInput, scalarVal, rNumPerLoop);
    rcOutQue_.EnQue<T>(Output);
    rcInQue_.FreeTensor(rInput);
    sumurcQue_.FreeTensor(sumurInput);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::CopyOutR(int64_t iter, int64_t rNumPerLoop)
{
    LocalTensor<T> Output = rcOutQue_.DeQue<T>();
    uint16_t blockCount = 1;
    uint16_t blockLen = rNumPerLoop * sizeof(T);
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    DataCopyParams dataCopyParams {blockCount, blockLen, srcStride, dstStride};
    DataCopyPad(rOutGm_[iter * rNumPerLoop], Output, dataCopyParams);
    rcOutQue_.FreeTensor(Output);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::ProcessC()
{
    if (GetBlockIdx() >= cCoreNumToUse_) {
        return;
    }
    // main core 
    if (GetBlockIdx() != cCoreNumToUse_ - 1) {
        for (int64_t i = 0; i < cLoopCount_; i++) {
            CopyInC(i, cNumPerLoop_);
            CopyInSumuc(i, cNumPerLoop_);
            ComputeC(cNumPerLoop_);
            CopyOutC(i, cNumPerLoop_);
        }
    // tail core
    } else {
        for (int64_t i = 0; i < cLoopCountTailCore_; i++) {
            CopyInC(i, cNumTailPerLoop_);
            CopyInSumuc(i, cNumTailPerLoop_);
            ComputeC(cNumTailPerLoop_);
            CopyOutC(i, cNumTailPerLoop_);
        }
        // hanlde case smaller than ubSize separately
        if (cNumTailLoopLast_ != 0) {
            cInGm_.SetGlobalBuffer((__gm__ T*)cIn_ + cLoopCountTailCore_ * cNumTailPerLoop_);
            cOutGm_.SetGlobalBuffer((__gm__ T*)cOut_ + cLoopCountTailCore_ * cNumTailPerLoop_);
            sumucGm_.SetGlobalBuffer((__gm__ T*)sum_u_c_ + cLoopCountTailCore_ * cNumTailPerLoop_);
            CopyInC(0, cNumTailLoopLast_);
            CopyInSumuc(0, cNumTailLoopLast_);
            ComputeC(cNumTailLoopLast_);
            CopyOutC(0, cNumTailLoopLast_);
        }
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::CopyInC(int iter, int64_t cNumPerLoop)
{
    // copy c gm -> ub
    LocalTensor<T> cInput = rcInQue_.AllocTensor<T>();
    int64_t ONCE_COPY_NUM = ONE_BLK_SIZE / sizeof(T);
    DataCopy(cInput, cInGm_[iter * cNumPerLoop], (cNumPerLoop + ONCE_COPY_NUM - 1) / ONCE_COPY_NUM * ONCE_COPY_NUM);
    rcInQue_.EnQue(cInput);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::CopyInSumuc(int iter, int64_t cNumPerLoop)
{
    // copy sum_u_c gm -> ub
    LocalTensor<T> sumucInput = sumurcQue_.AllocTensor<T>();
    int64_t ONCE_COPY_NUM = ONE_BLK_SIZE / sizeof(T);
    DataCopy(sumucInput, sumucGm_[iter * cNumPerLoop], (cNumPerLoop + ONCE_COPY_NUM - 1) / ONCE_COPY_NUM * ONCE_COPY_NUM);
    sumurcQue_.EnQue(sumucInput);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::ComputeC(int64_t cNumPerLoop)
{
    LocalTensor<T> cInput = rcInQue_.DeQue<T>();
    LocalTensor<T> sumucInput = sumurcQue_.DeQue<T>();
    LocalTensor<T> Output = rcOutQue_.AllocTensor<T>();

    // calc: beta3 * c
    Muls(Output, cInput, beta3_, cNumPerLoop);
    PipeBarrier<PIPE_V>();

    // calc: beta3 * c + (1 - beta3) * sum_u_c
    float scalarVal = (1 - beta3_) / N_;
    Axpy(Output, sumucInput, scalarVal, cNumPerLoop);
    
    rcOutQue_.EnQue<T>(Output);
    rcInQue_.FreeTensor(cInput);
    sumurcQue_.FreeTensor(sumucInput);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::CopyOutC(int64_t iter, int64_t cNumPerLoop)
{
    LocalTensor<T> Output = rcOutQue_.DeQue<T>();
    uint16_t blockCount = 1;
    uint16_t blockLen = cNumPerLoop * sizeof(T);
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    DataCopyParams dataCopyParams {blockCount, blockLen, srcStride, dstStride};
    DataCopyPad(cOutGm_[iter * cNumPerLoop], Output, dataCopyParams);
    rcOutQue_.FreeTensor(Output);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::ProcessParam() {
    if (blockIdx_ >= rRcCoreNumToUse_) {
        return;
    }

    InitProcessParam();

    // calc denominator, not in cycle
    T denominator = beta3_ * sumR_  / N_ + (1 - beta3_) * sum_u_rc_ / (M_ * N_);
    rcCoefficient_ = 1 / denominator;

    rRcBlockOffset_ = blockIdx_ * rRcNumPerCore_;

    const bool isTailCore = (blockIdx_ == rRcCoreNumToUse_ - 1);
    if (!isTailCore) {
        this->ProcessPerCoreParam();
    } else {
        this->ProcessTailCoreParam();
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::ProcessTailCoreParam()
{
    // const size_t typeSize = sizeof(T);
    const int64_t loopCount = rRcLoopCountTailCore_;
    int64_t curRNumInLoop = rRcNumPerLoop_;
    int64_t curCNumInLoop = 0;
    int64_t rLoopIdx = 0;
    // non-tail r loop
    for (rLoopIdx = 0; rLoopIdx < loopCount - 1; ++rLoopIdx) {
        CopyInParamr(rLoopIdx, curRNumInLoop);
        int64_t cLoopIdx = 0;
        // non-tail c loop
        for (cLoopIdx = 0; cLoopIdx < cRcLoopCount_ - 1; ++cLoopIdx) {
            curCNumInLoop = cRcNumPerLoop_;
            CalcParam(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
        }

        // tail c loop
        curCNumInLoop = cRcNumTailLoop_;
        CalcParam(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
        inQueR_.FreeTensor(rRcLocalTensor_);
    }

    // tail r loop
    curRNumInLoop = rRcNumTailLoopTailCore_;
    if (curRNumInLoop % NUM_PER_BLOCK == 0) {
        CopyInParamr(rLoopIdx, curRNumInLoop);
    } else {
        CopyInNotAlignedParamr(rLoopIdx, curRNumInLoop);
    }
    int64_t cLoopIdx = 0;
    for (cLoopIdx = 0; cLoopIdx < cRcLoopCount_ - 1; ++cLoopIdx) {
        curCNumInLoop = cRcNumPerLoop_;
        CalcParam(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
    }

    // tail loop
    curCNumInLoop = cRcNumTailLoop_;
    CalcParam(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
    inQueR_.FreeTensor(rRcLocalTensor_);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::CopyInNotAlignedParamr(int loopIdx, int64_t dataCount)
{
    uint16_t blockCount = 1;
    uint16_t blockLen = dataCount * sizeof(T);
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    DataCopyParams dataCopyParams {blockCount, blockLen, srcStride, dstStride};
    uint8_t leftPadding = 0;
    uint8_t rightPadding = NUM_PER_BLOCK - dataCount % NUM_PER_BLOCK;
    uint64_t paddingValue = 0;
    DataCopyPadParams padParams {true, leftPadding, rightPadding, paddingValue};

    LocalTensor<T> rLocal = inQueR_.AllocTensor<T>();
    int64_t offset = rRcBlockOffset_ + loopIdx * rRcNumPerLoop_;
    DataCopyPad(rLocal, rOutGm_[offset], dataCopyParams, padParams);
    inQueR_.EnQue(rLocal);
    rRcLocalTensor_ = inQueR_.DeQue<T>();
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::ProcessPerCoreParam()
{
    const int64_t loopCount = rRcLoopCount_;

    int64_t curRNumInLoop = rRcNumPerLoop_;
    int64_t curCNumInLoop = 0;
    int64_t rLoopIdx = 0;
    // non-tail r loop
    for (rLoopIdx = 0; rLoopIdx < loopCount - 1; ++rLoopIdx) {
        CopyInParamr(rLoopIdx, curRNumInLoop);
        int64_t cLoopIdx = 0;
        // non-tail c loop
        for (cLoopIdx = 0; cLoopIdx < cRcLoopCount_ - 1; ++cLoopIdx) {
            curCNumInLoop = cRcNumPerLoop_;
            CalcParam(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
        }

        // tail c loop
        curCNumInLoop = cRcNumTailLoop_;
        CalcParam(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
        inQueR_.FreeTensor(rRcLocalTensor_);
    }

    // tail r loop
    curRNumInLoop = rRcNumTailLoop_;
    if (curRNumInLoop % NUM_PER_BLOCK == 0) {
        CopyInParamr(rLoopIdx, curRNumInLoop);
    } else {
        CopyInNotAlignedParamr(rLoopIdx, curRNumInLoop);
    }

    int64_t cLoopIdx = 0;
    for (cLoopIdx = 0; cLoopIdx < cRcLoopCount_ - 1; ++cLoopIdx) {
        curCNumInLoop = cRcNumPerLoop_;
        CalcParam(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
    }

    // tail c loop
    curCNumInLoop = cRcNumTailLoop_;
    CalcParam(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
    inQueR_.FreeTensor(rRcLocalTensor_);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::CalcParam(int rLoopIdx, int cLoopIdx, int64_t curRNumInLoop, int64_t curCNumInLoop)
{
    if ((curRNumInLoop % NUM_PER_BLOCK == 0) && (curCNumInLoop % NUM_PER_BLOCK == 0)) {
        CopyInParamc(cLoopIdx, curCNumInLoop);
        CopyInParam(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
        CopyInm(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
        ComputeParam(curRNumInLoop, curCNumInLoop);
        CopyOutParam(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
        return;
    }

    CopyInNotAlignedParamc(cLoopIdx, curCNumInLoop);
    CopyInNotAlignedm(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
    CopyInNotAlignedParam(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
    int64_t alignedCNum = NUM_PER_BLOCK * CeilDiv(curCNumInLoop, NUM_PER_BLOCK);
    ComputeParam(curRNumInLoop, alignedCNum);
    CopyOutParam(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::CopyInNotAlignedParam(int rLoopIdx, int cLoopIdx,
                                                               int64_t curRNumInLoop, int64_t curCNumInLoop) {
    LocalTensor<T> paramLocal = inQueParam_.AllocTensor<T>();
    int64_t startRowNum = rRcBlockOffset_ + rLoopIdx * rRcNumPerLoop_;
    int64_t startColumnNum = cLoopIdx * cRcNumPerLoop_;
    int64_t startOffset = startRowNum * mShape_ + startColumnNum;
    int64_t alignedCNumInLoop = NUM_PER_BLOCK * CeilDiv(curCNumInLoop, NUM_PER_BLOCK);
    for (int64_t i = 0; i < curRNumInLoop; ++i) {
        int64_t srcOffset = startOffset + i * mShape_;
        int64_t dstOffset = i * alignedCNumInLoop;
        uint16_t blockCount = 1;
        uint16_t blockLen = curCNumInLoop * sizeof(T);
        uint16_t srcStride = 0;
        uint16_t dstStride = 0;
        DataCopyParams dataCopyParams {blockCount, blockLen, srcStride, dstStride};
        uint8_t leftPadding = 0;
        uint8_t rightPadding = alignedCNumInLoop - curCNumInLoop;
        uint64_t paddingValue = 0;
        DataCopyPadParams padParams {true, leftPadding, rightPadding, paddingValue};
        DataCopyPad(paramLocal[dstOffset], paramInGm_[srcOffset], dataCopyParams, padParams);
    }
    inQueParam_.EnQue(paramLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::CopyInNotAlignedm(int rLoopIdx, int cLoopIdx,
                                                            int64_t curRNumInLoop, int64_t curCNumInLoop) {
    LocalTensor<T> mLocal = inQuem_.AllocTensor<T>();
    int64_t startRowNum = rRcBlockOffset_ + rLoopIdx * rRcNumPerLoop_;
    int64_t startColumnNum = cLoopIdx * cRcNumPerLoop_;
    int64_t startOffset = startRowNum * mShape_ + startColumnNum;
    int64_t alignedCNumInLoop = NUM_PER_BLOCK * CeilDiv(curCNumInLoop, NUM_PER_BLOCK);
    for (int64_t i = 0; i < curRNumInLoop; ++i) {
        int64_t srcOffset = startOffset + i * mShape_;
        int64_t dstOffset = i * alignedCNumInLoop;

        uint16_t blockCount = 1;
        uint16_t blockLen = curCNumInLoop * sizeof(T);
        uint16_t srcStride = 0;
        uint16_t dstStride = 0;
        DataCopyParams dataCopyParams {blockCount, blockLen, srcStride, dstStride};
        uint8_t leftPadding = 0;
        uint8_t rightPadding = alignedCNumInLoop - curCNumInLoop;
        uint64_t paddingValue = 0;
        DataCopyPadParams padParams {true, leftPadding, rightPadding, paddingValue};
        DataCopyPad(mLocal[dstOffset], mGm_[srcOffset], dataCopyParams, padParams);
    }
    inQuem_.EnQue(mLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::ComputeParam(int64_t curRNumInLoop, int64_t curCNumInLoop)
{
    int64_t alignedCNum = NUM_PER_BLOCK * CeilDiv(curCNumInLoop, NUM_PER_BLOCK);
    curCNumInLoop = alignedCNum;
    // r*c
    LocalTensor<T> &rLocal = rRcLocalTensor_;
    LocalTensor<T> cLocal = inQueC_.DeQue<T>();
    LocalTensor<T> mLocal = inQuem_.DeQue<T>();
    LocalTensor<T> paramInLocal = inQueParam_.DeQue<T>();
    LocalTensor<T> paramOutLocal = outQueParam_.AllocTensor<T>();
    int64_t dataCount = curCNumInLoop * curRNumInLoop;

    LocalTensor<T> ub1 = ub1Buf_.Get<T>();
    LocalTensor<T> ub2 = ub2Buf_.Get<T>();
    LocalTensor<T> ub3 = ub3Buf_.Get<T>();

    // ub3: r*c
    if ((curRNumInLoop % HALF_CALC_SIZE != 0)
        || (curCNumInLoop % HALF_CALC_SIZE != 0)) {
            CalcRcCycleMode(ub3, cLocal, rLocal, curRNumInLoop, curCNumInLoop);
    } else {
        // ub1: r --> broadcast
        BroadcastR(ub1, rLocal, curRNumInLoop, curCNumInLoop);
        PipeBarrier<PIPE_V>();
        // ub2: r --> broadcast and transpose
        TransposeR(ub2, ub1, curRNumInLoop, curCNumInLoop);
        PipeBarrier<PIPE_V>();
        // ub3: r*c
        MulRC(ub3, ub2, cLocal, curRNumInLoop, curCNumInLoop);
    }
    PipeBarrier<PIPE_V>();
    // ub1: rc*coefficient
    Muls(ub1, ub3, rcCoefficient_, dataCount);

    PipeBarrier<PIPE_V>();

    // ub2 = sqrt(1/S)
    Sqrt(ub2, ub1, dataCount);
    PipeBarrier<PIPE_V>();
    // lr*m
    Muls(mLocal, mLocal, lr_, dataCount);
    PipeBarrier<PIPE_V>();
    // (m*lr) / ub2
    Div(ub2, mLocal, ub2, dataCount);
    PipeBarrier<PIPE_V>();

    // (1 - lr * weight_decay_)
    float mScalar = 1 - lr_ * weight_decay_;
    // paramInLocal: (1 - lr * weight_decay) * param
    Muls(paramInLocal, paramInLocal, mScalar, dataCount);
    PipeBarrier<PIPE_V>();

    // paramInLocal - ub2
    Sub(paramOutLocal, paramInLocal, ub2, dataCount);
    PipeBarrier<PIPE_V>();

    outQueParam_.EnQue<T>(paramOutLocal);

    inQueC_.FreeTensor(cLocal);
    inQuem_.FreeTensor(mLocal);
    inQueParam_.FreeTensor(paramInLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::CalcRcCycleMode(LocalTensor<T> &dst, LocalTensor<T> &src,
                                                                 LocalTensor<T> &srcScalar,
                                                                 int64_t curRNumInLoop, int64_t curCNumInLoop)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventId);
    WaitFlag<HardEvent::MTE2_S>(eventId);
    uint32_t dstOffset = 0;
    for (int rLoopIdx = 0; rLoopIdx < curRNumInLoop; ++rLoopIdx) {
        dstOffset = rLoopIdx * curCNumInLoop;
        Muls(dst[dstOffset], src, srcScalar.GetValue(rLoopIdx), curCNumInLoop); // curCNumInLoop<(256/typeSize) * 255
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::CopyOutParam(int rLoopIdx, int cLoopIdx, int64_t curRNumInLoop, int64_t curCNumInLoop)
{
    LocalTensor<T> paramLocal = outQueParam_.DeQue<T>();
    int64_t startRowNum = rRcBlockOffset_ + rLoopIdx * rRcNumPerLoop_;
    int64_t startColumnNum = cLoopIdx * cRcNumPerLoop_;
    int64_t startOffset = startRowNum * mShape_ + startColumnNum;
    if (curCNumInLoop % NUM_PER_BLOCK == 0) {
        for (int64_t i = 0; i < curRNumInLoop; ++i) {
            int64_t dstOffset = startOffset + i * mShape_;
            int64_t srcOffset = i * curCNumInLoop;
            DataCopy(paramOutGm_[dstOffset], paramLocal[srcOffset], curCNumInLoop);
        }
    } else {
        int64_t alignedCNum = NUM_PER_BLOCK * CeilDiv(curCNumInLoop, NUM_PER_BLOCK);
        uint16_t blockCount = 1;
        uint16_t blockLen = curCNumInLoop * sizeof(T);
        uint16_t srcStride = 0;
        uint16_t dstStride = 0;
        DataCopyParams dataCopyParams {blockCount, blockLen, srcStride, dstStride};
        for (int64_t i = 0; i < curRNumInLoop; ++i) {
            int64_t dstOffset = startOffset + i * mShape_;
            int64_t srcOffset = i * alignedCNum;
            DataCopyPad(paramOutGm_[dstOffset], paramLocal[srcOffset], dataCopyParams);
        }
    }

    outQueParam_.FreeTensor(paramLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::CopyInNotAlignedParamc(int loopIdx, int64_t dataCount)
{
    uint16_t blockCount = 1;
    uint16_t blockLen = dataCount * sizeof(T);
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    DataCopyParams dataCopyParams {blockCount, blockLen, srcStride, dstStride};
    uint8_t leftPadding = 0;
    uint8_t rightPadding = NUM_PER_BLOCK - dataCount % NUM_PER_BLOCK;
    uint64_t  paddingValue = 0;
    DataCopyPadParams padParams {true, leftPadding, rightPadding, paddingValue};

    LocalTensor<T> cLocal = inQueC_.AllocTensor<T>();
    int64_t offset = loopIdx * cRcNumPerLoop_;
    DataCopyPad(cLocal, cOutGm_[offset], dataCopyParams, padParams);
    inQueC_.EnQue(cLocal);
}


template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::CopyInm(int rLoopIdx, int cLoopIdx, int64_t curRNumInLoop, int64_t curCNumInLoop)
{
    LocalTensor<T> mLocal = inQuem_.AllocTensor<T>();
    int64_t startRowNum = rRcBlockOffset_ + rLoopIdx * rRcNumPerLoop_;
    int64_t startColumnNum = cLoopIdx * cRcNumPerLoop_;
    int64_t startOffset = startRowNum * mShape_ + startColumnNum;
    int64_t tmp = CeilDiv(curCNumInLoop, NUM_PER_BLOCK);
    curCNumInLoop = tmp * NUM_PER_BLOCK;
    for (int64_t i = 0; i < curRNumInLoop; ++i) {
        int64_t srcOffset = startOffset + i * mShape_;
        int64_t dstOffset = i * curCNumInLoop;
        DataCopy(mLocal[dstOffset], mGm_[srcOffset], curCNumInLoop);
    }
    inQuem_.EnQue(mLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::CopyInParam(int rLoopIdx, int cLoopIdx, int64_t curRNumInLoop, int64_t curCNumInLoop)
{
    LocalTensor<T> paramInLocal = inQueParam_.AllocTensor<T>();
    int64_t startRowNum = rRcBlockOffset_ + rLoopIdx * rRcNumPerLoop_;
    int64_t startColumnNum = cLoopIdx * cRcNumPerLoop_;
    int64_t startOffset = startRowNum * mShape_ + startColumnNum;
    int64_t tmp = CeilDiv(curCNumInLoop, NUM_PER_BLOCK);
    curCNumInLoop = tmp * NUM_PER_BLOCK;
    for (int64_t i = 0; i < curRNumInLoop; ++i) {
        int64_t srcOffset = startOffset + i * mShape_;
        int64_t dstOffset = i * curCNumInLoop;
        DataCopy(paramInLocal[dstOffset], paramInGm_[srcOffset], curCNumInLoop);
    }
    inQueParam_.EnQue(paramInLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::MulRC(LocalTensor<T> &dst,
                                                LocalTensor<T> &r, LocalTensor<T> &c,
                                                int64_t numR, int64_t numC)
{
    // r:(numR, numC)  c:(numC,)
    int64_t rcStride = (numC * sizeof(T)) / ONE_BLK_SIZE;
    BinaryRepeatParams binaryRepeatParams;
    binaryRepeatParams.dstBlkStride = 1;
    binaryRepeatParams.src0BlkStride = 1;
    binaryRepeatParams.src1BlkStride = 1;
    binaryRepeatParams.dstRepStride = rcStride;
    binaryRepeatParams.src0RepStride = rcStride;
    binaryRepeatParams.src1RepStride = 0;
    uint64_t mask = numC;
    uint8_t repeatTimes = numR;
    Mul(dst, r, c, mask, repeatTimes, binaryRepeatParams);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::BroadcastR(LocalTensor<T> &dst, LocalTensor<T> &src, int64_t numR, int64_t numC)
{
    for (int i = 0; i < numC; ++i) {
        DataCopy(dst[i * numR], src, numR);
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::TransposeR(LocalTensor<T> &dst, LocalTensor<T> &src, int64_t numR, int64_t numC)
{
    ConfusionTransposeTiling tiling;
    GetConfusionTransposeTiling(numR, numC, src.GetSize(), sizeof(T), tiling);
    ConfusionTranspose(dst, src, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::CopyInParamc(int loopIdx, int64_t dataCount) {
    LocalTensor<T> cLocal = inQueC_.AllocTensor<T>();
    int64_t offset = loopIdx * cRcNumPerLoop_;
    DataCopy(cLocal, cOutGm_[offset], dataCount);
    inQueC_.EnQue(cLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::CopyInParamr(int loopIdx, int64_t dataCount)
{
    LocalTensor<T> rLocal = inQueR_.AllocTensor<T>();
    int64_t offset = rRcBlockOffset_ + loopIdx * rRcNumPerLoop_;
    DataCopy(rLocal, rOutGm_[offset], dataCount);
    inQueR_.EnQue(rLocal);
    rRcLocalTensor_ = inQueR_.DeQue<T>();
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::InitForCalcParam()
{
    int64_t num = rRcNumPerLoop_ * cRcNumPerLoop_;
    num = CeilDiv(num, NUM_PER_BLOCK) * NUM_PER_BLOCK;
    int64_t uPerLoopSize = num * sizeof(float);
    pipe_.InitBuffer(inQuem_, 1, uPerLoopSize);
    pipe_.InitBuffer(inQueParam_, 1, uPerLoopSize);
    pipe_.InitBuffer(outQueParam_, 1, uPerLoopSize);
    pipe_.InitBuffer(ub1Buf_, uPerLoopSize);
    pipe_.InitBuffer(ub2Buf_, uPerLoopSize);
    pipe_.InitBuffer(ub3Buf_, uPerLoopSize);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::InitProcessParam()
{
    rOutGm_.SetGlobalBuffer((__gm__ T*)rOut_);
    cOutGm_.SetGlobalBuffer((__gm__ T*)cOut_);
    InitForCalcParam();

    if (sum_r_ == nullptr) {
        LocalTensor<T> inputLocal = scalarBuf_.Get<T>();
        DataCopyPad(inputLocal, sumRWorkspace_, {1, sizeof(float), 0, 0, 0}, {false, 0, 0, 0});
        event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        sumR_ = inputLocal.GetValue(0);
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart4Float32<T>::GetConfusionTransposeTiling(int64_t numR, int64_t numC, const uint32_t stackBufferSize,
                                                                             const uint32_t typeSize, ConfusionTransposeTiling &tiling)
{
    (void)stackBufferSize;
    uint32_t blockSize = ONE_BLK_SIZE /typeSize;
    uint32_t height = numR;
    uint32_t width = numC;
    uint32_t highBlock = height / BLOCK_CUBE;
    uint32_t stride = height * blockSize * typeSize / ONE_BLK_SIZE;
    uint32_t repeat = width / blockSize;

    tiling.param0 = blockSize;
    tiling.param1 = height;
    tiling.param2 = width;
    tiling.param3 = highBlock;
    tiling.param4 = stride;
    tiling.param5 = repeat;
}
#endif
