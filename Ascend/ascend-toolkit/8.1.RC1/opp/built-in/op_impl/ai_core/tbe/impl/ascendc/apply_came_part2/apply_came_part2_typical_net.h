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
 * \file apply_came_part2_typical_net.h
 * \brief
 */
#ifndef _APPLY_CAME_PART2_TYPICAL_NET_H_
#define _APPLY_CAME_PART2_TYPICAL_NET_H_

#include "kernel_operator.h"
#include "apply_came_part2_common.h"

using namespace AscendC;

template <typename T>
class ApplyCamePart2Typical {
public:
    __aicore__ inline ApplyCamePart2Typical() {};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR sumGradR, GM_ADDR sumGradC,
                                GM_ADDR sumGradRc, GM_ADDR rIn, GM_ADDR cIn, GM_ADDR beta2,
                                GM_ADDR sumR, GM_ADDR globalShape,
                                GM_ADDR rOut, GM_ADDR cOut, GM_ADDR u, GM_ADDR sumSquareU,
                                GM_ADDR workspace, const ApplyCamePart2TilingData* tilingData) {
        blockIdx_ = GetBlockIdx();
        // get tiling data
        CopyTilingData(tilingData);

        isInputSumR_ = (sumR != nullptr);
        isInputGlobalShape = (globalShape != nullptr);

        workspace_ = workspace;
        sumUOffset_ = 0;

        // set gm addr
        gradGm_.SetGlobalBuffer((__gm__ T*)grad);
        sumGradRInGm_.SetGlobalBuffer((__gm__ T*)sumGradR);
        sumGradCInGm_.SetGlobalBuffer((__gm__ T*)sumGradC);
        sumGradRcGm_.SetGlobalBuffer((__gm__ T*)sumGradRc);
        rInGm_.SetGlobalBuffer((__gm__ T*)rIn);
        cInGm_.SetGlobalBuffer((__gm__ T*)cIn);
        beta2Gm_.SetGlobalBuffer((__gm__ T*)beta2);
        sumRGm_.SetGlobalBuffer((__gm__ T*)sumR);
        globalShapeGm_.SetGlobalBuffer((__gm__ int64_t*)globalShape);

        rOutGm_.SetGlobalBuffer((__gm__ T*)rOut);
        cOutGm_.SetGlobalBuffer((__gm__ T*)cOut);
        uGm_.SetGlobalBuffer((__gm__ T*)u);
        sumSquareUGm_.SetGlobalBuffer((__gm__ T*)sumSquareU);
        sumRWorkspace_.SetGlobalBuffer((__gm__ float*)workspace_);
        sumSquareUWorkspace_.SetGlobalBuffer((__gm__ float*)workspace_ + WORKSPACE_ALIGNED_SIZE / FLOAT_SIZE);
        const int64_t num = 2;
        // init que for in and out
        pipe_.InitBuffer(inQue_, BUFFER_NUM, BUFFER_SIZE * num);
        pipe_.InitBuffer(outQue_, BUFFER_NUM, BUFFER_SIZE * num);
        pipe_.InitBuffer(ub1Buf_, BUFFER_SIZE);
        pipe_.InitBuffer(ub2Buf_, BUFFER_SIZE);
        pipe_.InitBuffer(ub3Buf_, BUFFER_SIZE);

        int64_t sumSquareUWsNum = CeilAlign(tilingData_.rRcCoreNumToUse * (tilingData_.rRcLoopCount + 1) * (tilingData_.cRcLoopCount + 1), 128);
        // set sum_square_u as 0
        if (GetBlockIdx() == 0) {
            InitOutput<float>(sumSquareUWorkspace_, sumSquareUWsNum, (float)0);
        }
        // wait core
        SyncAll();
    }

    __aicore__ inline void Process() {
        CopyInScalar();
        ProcessR();
        ProcessC();

        SyncAll();
        ProcessU();
        SyncAll();
    }

private:
    __aicore__ inline void CopyTilingData(const ApplyCamePart2TilingData* tilingData) {
        tilingData_.n = tilingData->n;
        tilingData_.m = tilingData->m;
        tilingData_.rNumPerCore = tilingData->rNumPerCore;
        tilingData_.rNumOnTailCore = tilingData->rNumOnTailCore;
        tilingData_.rCoreNumToUse = tilingData->rCoreNumToUse;
        tilingData_.cNumPerCore = tilingData->cNumPerCore;
        tilingData_.cNumOnTailCore = tilingData->cNumOnTailCore;
        tilingData_.cCoreNumToUse = tilingData->cCoreNumToUse;

        tilingData_.rRcNumPerCore = tilingData->rRcNumPerCore;
        tilingData_.rRcCoreNumToUse = tilingData->rRcCoreNumToUse;
        tilingData_.rRcNumOnTailCore = tilingData->rRcNumOnTailCore;

        tilingData_.rLoopCount = tilingData->rLoopCount;
        tilingData_.rNumPerLoop = tilingData->rNumPerLoop;
        tilingData_.rLoopCountTailCore = tilingData->rLoopCountTailCore;
        tilingData_.rNumTailLoop = tilingData->rNumTailLoop;
        tilingData_.rNumTailLoopTailCore = tilingData->rNumTailLoopTailCore;

        tilingData_.cLoopCount = tilingData->cLoopCount;
        tilingData_.cNumPerLoop = tilingData->cNumPerLoop;
        tilingData_.cLoopCountTailCore = tilingData->cLoopCountTailCore;
        tilingData_.cNumTailLoop = tilingData->cNumTailLoop;
        tilingData_.cNumTailLoopTailCore = tilingData->cNumTailLoopTailCore;

        tilingData_.rRcLoopCount = tilingData->rRcLoopCount;
        tilingData_.rRcNumPerLoop = tilingData->rRcNumPerLoop;
        tilingData_.rRcLoopCountTailCore = tilingData->rRcLoopCountTailCore;
        tilingData_.rRcNumTailLoop = tilingData->rRcNumTailLoop;
        tilingData_.rRcNumTailLoopTailCore = tilingData->rRcNumTailLoopTailCore;

        tilingData_.cRcLoopCount = tilingData->cRcLoopCount;
        tilingData_.cRcNumPerLoop = tilingData->cRcNumPerLoop;
        tilingData_.cRcNumTailLoop = tilingData->cRcNumTailLoop;

        tilingData_.ubSize = tilingData->ubSize;
        tilingData_.totalCoreNum = tilingData->totalCoreNum;
    }

    __aicore__ inline void CopyInScalar() {
        auto floatLocal = ub1Buf_.Get<float>();
        auto int64Local = ub2Buf_.Get<int64_t>();
        DataCopyPad(floatLocal, sumGradRcGm_, {1, sizeof(float), 0, 0, 0}, {false, 0, 0, 0});
        DataCopyPad(floatLocal[BETA2_OFFSET], beta2Gm_, {1, sizeof(float), 0, 0, 0}, {false, 0, 0, 0});
        if (isInputSumR_) {
            DataCopyPad(floatLocal[SUMR_OFFSET], sumRGm_, {1, sizeof(float), 0, 0, 0}, {false, 0, 0, 0});
        } else {
            DataCopyPad(floatLocal[SUMR_OFFSET], sumRWorkspace_, {1, sizeof(float), 0, 0, 0}, {false, 0, 0, 0});
        } 
        if (isInputGlobalShape) {
            DataCopy(int64Local, globalShapeGm_, INT64_NUM_BLOCK);
        }

        event_t eventMte2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventMte2S);
        WaitFlag<HardEvent::MTE2_S>(eventMte2S);

        sumGradRc_ = floatLocal.GetValue(0);
        beta2_ = floatLocal.GetValue(BETA2_OFFSET);
        sumR_ = floatLocal.GetValue(SUMR_OFFSET);

        pipe_barrier(PIPE_ALL);
        if (isInputGlobalShape) {
            Cast(floatLocal, int64Local, RoundMode::CAST_ROUND, INT64_NUM_BLOCK);
        } else {
            int64Local.SetValue(0, tilingData_.n);
            int64Local.SetValue(1, tilingData_.m);
            pipe_barrier(PIPE_ALL);
            Cast(floatLocal, int64Local, RoundMode::CAST_ROUND, INT64_NUM_BLOCK);
        }
        pipe_barrier(PIPE_ALL);
        N_ = floatLocal.GetValue(0);
        M_ = floatLocal.GetValue(1);

        sumGradRWeight_ = (1 - beta2_) / M_;
        sumGradCWeight_ = (1 - beta2_) / N_;
    }

    __aicore__ inline void ProcessR() {
        if (blockIdx_ >= tilingData_.rCoreNumToUse) {
            return;
        }

        rBlockOffset_ = blockIdx_ * tilingData_.rNumPerCore;
        int64_t dataCount = (blockIdx_ != tilingData_.rCoreNumToUse - 1)
                            ? tilingData_.rNumPerCore : tilingData_.rNumOnTailCore;
        CopyIn(rInGm_, sumGradRInGm_, rBlockOffset_, dataCount);
        auto rLocal = inQue_.DeQue<float>();

        auto rInLocal = rLocal;
        auto sumGradRLocal = rLocal[BUFFER_SIZE_FP32];
        auto rOutLocal = outQue_.AllocTensor<float>();

        Muls(rOutLocal, rInLocal, beta2_, dataCount);
        pipe_barrier(PIPE_V);
        Axpy(rOutLocal, sumGradRLocal, sumGradRWeight_, dataCount);
        outQue_.EnQue(rOutLocal);
        CopyOut(rOutGm_, rBlockOffset_, dataCount);

        inQue_.FreeTensor(rLocal);
    }

    __aicore__ inline void ProcessC() {
        if (blockIdx_ >= tilingData_.cCoreNumToUse) {
            return;
        }

        int64_t dataCount = (blockIdx_ != tilingData_.cCoreNumToUse - 1)
                            ? tilingData_.cNumPerCore : tilingData_.cNumOnTailCore;

        cBlockOffset_ = blockIdx_ * tilingData_.cNumPerCore;
        CopyIn(cInGm_, sumGradCInGm_, cBlockOffset_, dataCount);
        Compute(dataCount, sumGradCWeight_);
        CopyOut(cOutGm_, cBlockOffset_, dataCount);
    }

    __aicore__ inline void CopyIn(GlobalTensor<float> &inGm,
                                  GlobalTensor<float> &inGradGm,
                                  int64_t offset, int64_t dataCount) {
        auto inLocal = inQue_.AllocTensor<float>();
        DataCopy(inLocal, inGm[offset], dataCount);
        DataCopy(inLocal[BUFFER_SIZE_FP32], inGradGm[offset], dataCount);
        inQue_.EnQue(inLocal);
    }

    __aicore__ inline void Compute(int64_t dataCount, float weight) {
        auto rLocal = inQue_.DeQue<float>();
        auto rInLocal = rLocal;
        auto sumGradRLocal = rLocal[BUFFER_SIZE_FP32];
        auto rOutLocal = outQue_.AllocTensor<float>();

        Muls(rOutLocal, rInLocal, beta2_, dataCount);
        inQue_.FreeTensor(rLocal);
        pipe_barrier(PIPE_V);
        Axpy(rOutLocal, sumGradRLocal, weight, dataCount);
        outQue_.EnQue(rOutLocal);
    }

    __aicore__ inline void CopyOut(GlobalTensor<float> outGm,
                                   int64_t offset,
                                   int64_t dataCount) {
        auto rLocal = outQue_.DeQue<float>();
        DataCopy(outGm[offset], rLocal, dataCount);
        outQue_.FreeTensor(rLocal);
    }

    __aicore__ inline void ProcessU() {
        if (blockIdx_ >= tilingData_.rRcCoreNumToUse) {
            return;
        }

        const int64_t mSize = tilingData_.cRcNumPerLoop;
        const int64_t rLoopNum = (blockIdx_ != tilingData_.rRcCoreNumToUse - 1)
                                 ? tilingData_.rRcNumPerCore : tilingData_.rRcNumOnTailCore;;

        int64_t rOffset = blockIdx_ * tilingData_.rRcNumPerCore;
        int64_t rLoopNumAlign = CeilDiv(rLoopNum, FLOAT_NUM_BLOCK) * FLOAT_NUM_BLOCK;

        auto rLocal = inQue_.AllocTensor<float>();
        DataCopy(rLocal, rOutGm_[rOffset], rLoopNumAlign);
        DataCopy(rLocal[BUFFER_SIZE_FP32], cOutGm_, mSize);
        inQue_.EnQue(rLocal);

        rLocal = inQue_.DeQue<float>();
        auto cLocal = rLocal[BUFFER_SIZE_FP32];
        rcWeight_ = 1 / (beta2_ * sumR_ / N_ + (1 - beta2_) * sumGradRc_/ (M_ * N_));

        auto cTmp = ub1Buf_.Get<float>();
        auto rTmp = ub2Buf_.Get<float>();
        auto sumSquareU = ub3Buf_.Get<float>();
        Duplicate(sumSquareU, (float)0, mSize);
        Muls(cTmp, cLocal, rcWeight_, mSize);
        Muls(rTmp, rLocal, (float)1.0, rLoopNumAlign);
        inQue_.FreeTensor(rLocal);

        for (int64_t i = 0; i < rLoopNum && (rOffset + i) < tilingData_.n; ++i) {
            int64_t offset = (rOffset + i) * mSize;
            CopyInGrad(offset, mSize);
            ComputeU(cTmp, sumSquareU, rTmp.GetValue(i), mSize);
            CopyOut(uGm_, offset, mSize);
        }
        ComputeSumSquareU(sumSquareU);
    }
    __aicore__ inline void CopyInGrad(int64_t offset, int64_t dataCount) {
        auto rLocal = inQue_.AllocTensor<float>();
        DataCopy(rLocal, gradGm_[offset], dataCount);
        inQue_.EnQue(rLocal);
    }

    __aicore__ inline void ComputeU(LocalTensor<float>& cTmp, LocalTensor<float>& sumSquareU,
                                    float rValue, int64_t dataCount) {
        auto uOutLocal = outQue_.AllocTensor<float>();
        Muls(uOutLocal, cTmp, rValue, dataCount);
        pipe_barrier(PIPE_V);
        Sqrt(uOutLocal, uOutLocal, dataCount);
        pipe_barrier(PIPE_V);

        auto gradLocal = inQue_.DeQue<float>();
        Div(uOutLocal, gradLocal, uOutLocal, dataCount);
        pipe_barrier(PIPE_V);
        Mul(gradLocal, uOutLocal, uOutLocal, dataCount);
        outQue_.EnQue(uOutLocal);
        Add(sumSquareU, sumSquareU, gradLocal, dataCount);
        inQue_.FreeTensor(gradLocal);
    }

    __aicore__ inline void ComputeSumSquareU(LocalTensor<float>& sumGradU) {
        const int64_t mSize = tilingData_.cRcNumPerLoop;
        auto outLocal = outQue_.AllocTensor<float>();
        ReduceSum(outLocal, sumGradU, ub1Buf_.Get<float>(), mSize);
        outQue_.EnQue(outLocal);
        outLocal = outQue_.DeQue<float>();

        int64_t blockOffset = blockIdx_ * (tilingData_.rRcLoopCount + 1) * (tilingData_.cRcLoopCount + 1);
        uint16_t blockCount = 1;
        uint16_t blockLen = sizeof(float);
        uint16_t srcStride = 0;
        uint16_t dstStride = 0;
        DataCopyParams copyParams {blockCount, blockLen, srcStride, dstStride};
        DataCopyPad(sumSquareUWorkspace_[blockOffset + sumUOffset_], outLocal, copyParams);
        sumUOffset_ = sumUOffset_ + 1;
        outQue_.FreeTensor(outLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue_;

    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue_;
    TQue<QuePosition::VECOUT, 1> workQue_;

    // each tbuf may be used for multiple purposes
    TBuf<QuePosition::VECCALC> ub1Buf_; // 1. rMatrix after broadcast from r
    TBuf<QuePosition::VECCALC> ub2Buf_; // 1. rMatrix after transpose; 2. r * c
    TBuf<QuePosition::VECCALC> ub3Buf_; // grad

    GlobalTensor<T> rInGm_;
    GlobalTensor<T> sumGradRInGm_;
    GlobalTensor<T> cInGm_;
    GlobalTensor<T> sumGradCInGm_;
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> rOutGm_;
    GlobalTensor<T> cOutGm_;
    GlobalTensor<T> sumSquareUGm_;
    GlobalTensor<T> uGm_;

    GlobalTensor<T> sumGradRcGm_;
    GlobalTensor<T> beta2Gm_;
    GlobalTensor<T> sumRGm_;
    GlobalTensor<int64_t> globalShapeGm_;

    GlobalTensor<T> sumRWorkspace_;
    GlobalTensor<float> sumSquareUWorkspace_;

    GM_ADDR workspace_;

    bool isInputSumR_; // is sum_r in input null?
    bool isInputGlobalShape;

    // scalar
    float sumGradRc_;
    float beta2_;
    float sumR_;
    float M_;
    float N_;

    float sumGradRWeight_;
    float sumGradCWeight_;
    float rcWeight_;

    int64_t blockIdx_;

    // tiling data
    ApplyCamePart2TilingData tilingData_;

    // block offset
    int64_t rBlockOffset_;
    int64_t cBlockOffset_;
    int64_t rRcBlockOffset_;
    int64_t sumUOffset_;

    const int64_t NUM_PER_BLOCK = BLOCK_SIZE / sizeof(T);

    static constexpr int64_t TYPICAL_BUFFER_NUM = 2;
    static constexpr int64_t FLOAT_NUM_BLOCK = 8;
    static constexpr int64_t INT64_NUM_BLOCK = 4;

    static constexpr int64_t BUFFER_SIZE = 16 * 1024;
    static constexpr int64_t BUFFER_SIZE_FP32 = 4 * 1024;

    static constexpr int64_t BETA2_OFFSET = 8;
    static constexpr int64_t SUMR_OFFSET = 16;
};
#endif // _APPLY_CAME_PART2_TYPICAL_NET_H_
