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
 * \file apply_came_part2.cpp
 * \brief
 */
#include "apply_came_part2.h"
#include "apply_came_part2_float16.h"
#include "apply_came_part2_typical_net.h"
#include "apply_came_part2_float16_typical_net.h"
#include "apply_came_part2_post.h"
#include "apply_came_part2_pre.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void apply_came_part2(GM_ADDR grad, GM_ADDR sumGradR, GM_ADDR sumGradC,
                                                       GM_ADDR sumGradRc, GM_ADDR rIn, GM_ADDR cIn, GM_ADDR beta2,
                                                       GM_ADDR sumR, GM_ADDR globalShape,
                                                       GM_ADDR rOut, GM_ADDR cOut, GM_ADDR u, GM_ADDR sumSquareU,
                                                       GM_ADDR workspace, GM_ADDR tiling)
{
  if (workspace == nullptr) {
    return;
  }
  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }

  GET_TILING_DATA(tilingData, tiling);

  if (TILING_KEY_IS(101)) {
    if (sumR == nullptr) {
        ApplyCamePart2Pre<float> applyCamePart2Pre;
        applyCamePart2Pre.Init(rIn, userWS, &tilingData);
        applyCamePart2Pre.Process();
    }
    ApplyCamePart2<float> applyCamePart2;
    applyCamePart2.Init(grad, sumGradR, sumGradC, sumGradRc, rIn, cIn, beta2, sumR, globalShape,
                        rOut, cOut, u, sumSquareU, userWS, &tilingData);
    applyCamePart2.Process();
  }

  if (TILING_KEY_IS(102)) {
    if (sumR == nullptr) {
        ApplyCamePart2Pre<half> applyCamePart2Pre;
        applyCamePart2Pre.Init(rIn, userWS, &tilingData);
        applyCamePart2Pre.Process();
    }
    ApplyCamePart2Float16<half> op;
    op.Init(grad, sumGradR, sumGradC, sumGradRc, rIn, cIn, beta2, sumR, globalShape,
            rOut, cOut, u, sumSquareU, userWS, &tilingData);
    op.Process();
  }

  if (TILING_KEY_IS(103)) {
    if (sumR == nullptr) {
        ApplyCamePart2Pre<bfloat16_t> applyCamePart2Pre;
        applyCamePart2Pre.Init(rIn, userWS, &tilingData);
        applyCamePart2Pre.Process();
    }
    ApplyCamePart2Float16<bfloat16_t> op;
    op.Init(grad, sumGradR, sumGradC, sumGradRc, rIn, cIn, beta2, sumR, globalShape,
            rOut, cOut, u, sumSquareU, userWS, &tilingData);
    op.Process();
  }

  if (TILING_KEY_IS(201)) {
    if (sumR == nullptr) {
        ApplyCamePart2Pre<float> applyCamePart2Pre;
        applyCamePart2Pre.Init(rIn, userWS, &tilingData);
        applyCamePart2Pre.Process();
    }
    ApplyCamePart2Typical<float> op;
    op.Init(grad, sumGradR, sumGradC, sumGradRc, rIn, cIn, beta2, sumR, globalShape,
            rOut, cOut, u, sumSquareU, userWS, &tilingData);
    op.Process();
  }

  if (TILING_KEY_IS(202)) {
    if (sumR == nullptr) {
        ApplyCamePart2Pre<half> applyCamePart2Pre;
        applyCamePart2Pre.Init(rIn, userWS, &tilingData);
        applyCamePart2Pre.Process();
    }
    ApplyCamePart2Float16Typical<half> op;
    op.Init(grad, sumGradR, sumGradC, sumGradRc, rIn, cIn, beta2, sumR, globalShape,
            rOut, cOut, u, sumSquareU, userWS, &tilingData);
    op.Process();
  }

  if (TILING_KEY_IS(203)) {
    if (sumR == nullptr) {
        ApplyCamePart2Pre<bfloat16_t> applyCamePart2Pre;
        applyCamePart2Pre.Init(rIn, userWS, &tilingData);
        applyCamePart2Pre.Process();
    }
    ApplyCamePart2Float16Typical<bfloat16_t> op;
    op.Init(grad, sumGradR, sumGradC,
            sumGradRc, rIn, cIn, beta2,
            sumR, globalShape,
            rOut, cOut, u, sumSquareU,
            userWS, &tilingData);
    op.Process();
  }
  ApplyCamePart2Post<float> opPost;
  opPost.Init(sumSquareU, userWS, &tilingData);
  opPost.Process();
}

template <typename T>
__aicore__ inline int64_t ApplyCamePart2<T>::GetMaxCoreNumToUse() {
    int64_t tmpMax = (tilingData_.rCoreNumToUse > tilingData_.cCoreNumToUse)
                     ? tilingData_.rCoreNumToUse : tilingData_.cCoreNumToUse;
    return (tmpMax > tilingData_.rRcCoreNumToUse) ? tmpMax : tilingData_.rRcCoreNumToUse;
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::Init(GM_ADDR grad, GM_ADDR sumGradR, GM_ADDR sumGradC,
                                               GM_ADDR sumGradRc, GM_ADDR rIn, GM_ADDR cIn, GM_ADDR beta2,
                                               GM_ADDR sumR, GM_ADDR globalShape,
                                               GM_ADDR rOut, GM_ADDR cOut, GM_ADDR u, GM_ADDR sumSquareU,
                                               GM_ADDR workspace, const ApplyCamePart2TilingData* tilingData) {
    blockIdx_ = GetBlockIdx();
    // get tiling data
    CopyTilingData(tilingData);

    int i = 0;
    sumUOffset_ = 0;
    workspace_ = workspace;
    isInputSumR_ = (sumR != nullptr);
    isInputGlobalShape = (globalShape != nullptr);

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

    InitTque();

    int64_t num = tilingData_.rRcNumPerLoop * tilingData_.cRcNumPerLoop;
    num = CeilDiv(num, NUM_PER_BLOCK) * NUM_PER_BLOCK;
    int64_t uPerLoopSize = num * sizeof(T);
    pipe_.InitBuffer(inQueGrad_, BUFFER_NUM, uPerLoopSize);
    pipe_.InitBuffer(outQueU_, BUFFER_NUM, uPerLoopSize);
    pipe_.InitBuffer(outQueSumSquareU_, BUFFER_NUM, ONE_BLK_SIZE);

    pipe_.InitBuffer(ub1Buf_, uPerLoopSize);
    pipe_.InitBuffer(ub2Buf_, uPerLoopSize);
    pipe_.InitBuffer(ub3Buf_, uPerLoopSize);

    int64_t sumSquareUWsNum = CeilAlign(tilingData_.rRcCoreNumToUse * (tilingData_.rRcLoopCount + 1) * (tilingData_.cRcLoopCount + 1), 128);
    // set sum_square_u as 0
    if (GetBlockIdx() == 0) {
        InitOutput<float>(sumSquareUWorkspace_, sumSquareUWsNum, (float)0);
    }
    // wait core
    SyncAll();
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::InitTque() {
    int64_t rBufferLength = (tilingData_.rNumPerLoop > tilingData_.rRcNumPerLoop)
                            ? tilingData_.rNumPerLoop : tilingData_.rRcNumPerLoop;
    int64_t cBufferLength = (tilingData_.cNumPerLoop > tilingData_.cRcNumPerLoop)
                            ? tilingData_.cNumPerLoop : tilingData_.cRcNumPerLoop;
    rBufferLength = NUM_PER_BLOCK * CeilDiv(rBufferLength, NUM_PER_BLOCK);
    cBufferLength = NUM_PER_BLOCK * CeilDiv(cBufferLength, NUM_PER_BLOCK);

    // init que for in and out
    pipe_.InitBuffer(inQueR_, BUFFER_NUM, rBufferLength * sizeof(T));
    pipe_.InitBuffer(inQueC_, BUFFER_NUM, cBufferLength * sizeof(T));
    pipe_.InitBuffer(inQueSumGradR_, BUFFER_NUM, rBufferLength * sizeof(T));
    pipe_.InitBuffer(inQueSumGradC_, BUFFER_NUM, cBufferLength * sizeof(T));
    pipe_.InitBuffer(outQueR_, BUFFER_NUM, rBufferLength * sizeof(T));
    pipe_.InitBuffer(outQueC_, BUFFER_NUM, cBufferLength * sizeof(T));
    pipe_.InitBuffer(outQueSumR_, BUFFER_NUM, ONE_BLK_SIZE);

    CalcWorkLocal();
    pipe_.InitBuffer(reduceSumWorkQueue_, 1, workLocalNeedSize_ * sizeof(T));
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::CalcWorkLocal()
{
    // can be moved to tiling
    int64_t elementsPerBlock = BLOCK_SIZE / sizeof(T);
    int64_t elementsPerRepeat = CALC_SIZE / sizeof(T);
    int64_t handleNum = tilingData_.rRcNumPerLoop * tilingData_.cRcNumPerLoop;
    int64_t firstMaxRepeat = handleNum / elementsPerRepeat;
    int64_t iter1OutputCount = firstMaxRepeat;
    int64_t iter1AlignEnd = (iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
    workLocalNeedSize_ = iter1AlignEnd;
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::CopyTilingData(const ApplyCamePart2TilingData* tilingData) {
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

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::CopyInScalar() {
    pipe_.InitBuffer(scalarBuf_, ONE_BLK_SIZE);
    // copy sum_grad_rc
    LocalTensor<T> inputLocal = scalarBuf_.Get<T>();
    DataCopyPad(inputLocal, sumGradRcGm_, {1, sizeof(T), 0, 0, 0}, {false, 0, 0, 0});
    event_t eventMte2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventMte2S);
    WaitFlag<HardEvent::MTE2_S>(eventMte2S);
    sumGradRc_ = inputLocal.GetValue(0);
    // copy beta3
    DataCopyPad(inputLocal, beta2Gm_, {1, sizeof(T), 0, 0, 0}, {false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(eventMte2S);
    WaitFlag<HardEvent::MTE2_S>(eventMte2S);
    beta2_ = inputLocal.GetValue(0);
    // copy global_shape
    if (isInputGlobalShape) {
        LocalTensor<int64_t> int64Local = scalarBuf_.Get<int64_t>();
        DataCopy(int64Local, globalShapeGm_, INT64_PER_BLOCK);
        pipe_barrier(PIPE_ALL);
        LocalTensor<float> floatLocal = scalarBuf_.Get<float>();
        Cast(floatLocal, int64Local, RoundMode::CAST_ROUND, INT64_PER_BLOCK);
        pipe_barrier(PIPE_ALL);
        N_ = floatLocal.GetValue(0);
        M_ = floatLocal.GetValue(1);
    } else {
        LocalTensor<int64_t> int64Local = scalarBuf_.Get<int64_t>();
        int64Local.SetValue(0, tilingData_.n);
        int64Local.SetValue(1, tilingData_.m);
        pipe_barrier(PIPE_ALL);
        LocalTensor<float> floatLocal = scalarBuf_.Get<float>();
        Cast(floatLocal, int64Local, RoundMode::CAST_ROUND, INT64_PER_BLOCK);
        pipe_barrier(PIPE_ALL);
        N_ = floatLocal.GetValue(0);
        M_ = floatLocal.GetValue(1);
    }
    // copy sum_r
    if (isInputSumR_) {
        DataCopyPad(inputLocal, sumRGm_, {1, sizeof(T), 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(eventMte2S);
        WaitFlag<HardEvent::MTE2_S>(eventMte2S);
        sumR_ = inputLocal.GetValue(0);
    } else {
        DataCopyPad(inputLocal, sumRWorkspace_, {1, sizeof(T), 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(eventMte2S);
        WaitFlag<HardEvent::MTE2_S>(eventMte2S);
        sumR_ = inputLocal.GetValue(0);
    }
    pipe_barrier(PIPE_ALL);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::Process() {
    CopyInScalar();

    // calc r
    ProcessR();

    // calc c
    ProcessC();

    // must synch all core
    SyncAll();

    // calc u and sum_square_u tmp(include r*c)
    ProcessU();

    // must sync all core
    SyncAll();
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::ProcessPerCoreR() {
    const int64_t loopCount = tilingData_.rLoopCount;
    // non-tail loop
    int i = 0;
    for (i = 0; i < loopCount - 1; ++i) {
        int64_t dataCount = tilingData_.rNumPerLoop;
        CopyInR(i, dataCount, false);
        ComputeR(dataCount);
        CopyOutR(i, dataCount);
    }

    // tail loop
    int64_t dataCount = tilingData_.rNumTailLoop;
    CopyInR(i, dataCount, true);
    ComputeR(dataCount);
    CopyOutR(i, dataCount);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::ProcessTailCoreR() {
    const int64_t loopCount = tilingData_.rLoopCountTailCore;
    // non-tail loop
    int i = 0;
    for (i = 0; i < loopCount - 1; ++i) {
        int64_t dataCount = tilingData_.rNumPerLoop;
        CopyInR(i, dataCount, false);
        ComputeR(dataCount);
        CopyOutR(i, dataCount);
    }

    // tail loop
    int64_t dataCount = tilingData_.rNumTailLoopTailCore;
    CopyInR(i, dataCount, true);
    ComputeR(dataCount);
    CopyOutR(i, dataCount);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::ProcessR() {
    if (blockIdx_ >= tilingData_.rCoreNumToUse) {
        return;
    }

    rBlockOffset_ = blockIdx_ * tilingData_.rNumPerCore;
    if (blockIdx_ != tilingData_.rCoreNumToUse - 1) {
        ProcessPerCoreR();
    } else {
        ProcessTailCoreR();
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::CopyInR(int loopIdx, int64_t dataCount, bool tail) {
    LocalTensor<T> rLocal = inQueR_.AllocTensor<T>();
    LocalTensor<T> sumGradRLocal = inQueSumGradR_.AllocTensor<T>();
    int64_t offset = rBlockOffset_ + loopIdx * tilingData_.rNumPerLoop;
    if (tail) {
        uint16_t blockCount = 1;
        uint32_t blockLen = dataCount * sizeof(T);
        uint32_t srcStride = 0;
        uint32_t dstStride = 0;
        uint32_t rsv = 0;
        DataCopyExtParams dataCopyExtParams {blockCount, blockLen, srcStride, dstStride, rsv};
        DataCopyPadExtParams<T> dataCopyPadExtParams {false, 0, 0, 0};
        DataCopyPad(rLocal, rInGm_[offset], dataCopyExtParams, dataCopyPadExtParams);
        DataCopyPad(sumGradRLocal, sumGradRInGm_[offset], dataCopyExtParams, dataCopyPadExtParams);
    } else {
        DataCopy(rLocal, rInGm_[offset], dataCount);
        DataCopy(sumGradRLocal, sumGradRInGm_[offset], dataCount);
    }
    inQueR_.EnQue(rLocal);
    inQueSumGradR_.EnQue(sumGradRLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::ComputeR(int64_t dataCount) {
    LocalTensor<T> rLocal = inQueR_.DeQue<T>();
    LocalTensor<T> sumGradRLocal = inQueSumGradR_.DeQue<T>();
    LocalTensor<T> rOutLocal = outQueR_.AllocTensor<T>();

    Muls(rOutLocal, rLocal, beta2_, dataCount);
    float scalar = (1 - beta2_) / M_;
    pipe_barrier(PIPE_V);
    Axpy(rOutLocal, sumGradRLocal, scalar, dataCount);

    outQueR_.EnQue(rOutLocal);
    inQueR_.FreeTensor(rLocal);
    inQueSumGradR_.FreeTensor(sumGradRLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::CopyOutR(int loopIdx, int64_t dataCount) {
    LocalTensor<T> rLocal = outQueR_.DeQue<T>();
    int64_t offset = rBlockOffset_ + loopIdx * tilingData_.rNumPerLoop;
    uint16_t blockCount = 1;
    uint16_t blockLen = dataCount * sizeof(T);
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    DataCopyParams dataCopyParams {blockCount, blockLen, srcStride, dstStride};
    DataCopyPad(rOutGm_[offset], rLocal, dataCopyParams);
    outQueR_.FreeTensor(rLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::ProcessPerCoreC() {
    const int64_t loopCount = tilingData_.cLoopCount;
    // non-tail loop
    int i = 0;
    for (i = 0; i < loopCount - 1; ++i) {
        int64_t dataCount = tilingData_.cNumPerLoop;
        CopyInC(i, dataCount, false);
        ComputeC(dataCount);
        CopyOutC(i, dataCount);
    }

    // tail loop
    int64_t dataCount = tilingData_.cNumTailLoop;
    CopyInC(i, dataCount, true);
    ComputeC(dataCount);
    CopyOutC(i, dataCount);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::ProcessTailCoreC() {
    const int64_t loopCount = tilingData_.cLoopCountTailCore;
    // non-tail loop
    int i = 0;
    for (i = 0; i < loopCount - 1; ++i) {
        int64_t dataCount = tilingData_.cNumPerLoop;
        CopyInC(i, dataCount, false);
        ComputeC(dataCount);
        CopyOutC(i, dataCount);
    }

    // tail loop
    int64_t dataCount = tilingData_.cNumTailLoopTailCore;
    CopyInC(i, dataCount, true);
    ComputeC(dataCount);
    CopyOutC(i, dataCount);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::ProcessC() {
    if (blockIdx_ >= tilingData_.cCoreNumToUse) {
        return;
    }
    cBlockOffset_ = blockIdx_ * tilingData_.cNumPerCore;
    if (blockIdx_ != tilingData_.cCoreNumToUse - 1) {
        ProcessPerCoreC();
    } else {
        ProcessTailCoreC();
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::CopyInC(int loopIdx, int64_t dataCount, bool tail) {
    LocalTensor<T> cLocal = inQueC_.AllocTensor<T>();
    LocalTensor<T> sumGradCLocal = inQueSumGradC_.AllocTensor<T>();
    int64_t offset = cBlockOffset_ + loopIdx * tilingData_.cNumPerLoop;
    if (tail) {
        uint16_t blockCount = 1;
        uint32_t blockLen = dataCount * sizeof(T);
        uint32_t srcStride = 0;
        uint32_t dstStride = 0;
        uint32_t rsv = 0;
        DataCopyExtParams dataCopyExtParams {blockCount, blockLen, srcStride, dstStride, rsv};
        DataCopyPadExtParams<T> dataCopyPadExtParams {false, 0, 0, 0};
        DataCopyPad(cLocal, cInGm_[offset], dataCopyExtParams, dataCopyPadExtParams);
        DataCopyPad(sumGradCLocal, sumGradCInGm_[offset], dataCopyExtParams, dataCopyPadExtParams);
    } else {
        DataCopy(cLocal, cInGm_[offset], dataCount);
        DataCopy(sumGradCLocal, sumGradCInGm_[offset], dataCount);
    }
    inQueC_.EnQue(cLocal);
    inQueSumGradC_.EnQue(sumGradCLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::ComputeC(int64_t dataCount) {
    LocalTensor<T> cLocal = inQueC_.DeQue<T>();
    LocalTensor<T> sumGradCLocal = inQueSumGradC_.DeQue<T>();
    LocalTensor<T> cOutLocal = outQueC_.AllocTensor<T>();

    Muls(cOutLocal, cLocal, beta2_, dataCount);
    float scalar = (1 - beta2_) / N_;
    pipe_barrier(PIPE_V);
    Axpy(cOutLocal, sumGradCLocal, scalar, dataCount);

    outQueC_.EnQue(cOutLocal);
    inQueC_.FreeTensor(cLocal);
    inQueSumGradC_.FreeTensor(sumGradCLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::CopyOutC(int loopIdx, int64_t dataCount) {
    LocalTensor<T> cLocal = outQueC_.DeQue<T>();
    int64_t offset = cBlockOffset_ + loopIdx * tilingData_.cNumPerLoop;
    uint16_t blockCount = 1;
    uint16_t blockLen = dataCount * sizeof(T);
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    DataCopyParams dataCopyParams {blockCount, blockLen, srcStride, dstStride};
    DataCopyPad(cOutGm_[offset], cLocal, dataCopyParams);
    outQueC_.FreeTensor(cLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::CalcU(int rLoopIdx, int cLoopIdx,
                                                int64_t curRNumInLoop, int64_t curCNumInLoop) {
    if ((curRNumInLoop % NUM_PER_BLOCK == 0) && (curCNumInLoop % NUM_PER_BLOCK == 0)) {
        CopyInUc(cLoopIdx, curCNumInLoop);
        CopyInGrad(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
        ComputeU(curRNumInLoop, curCNumInLoop);
        CopyOutU(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
        return;
    }

    CopyInNotAlignedUc(cLoopIdx, curCNumInLoop);
    CopyInNotAlignedGrad(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
    ComputeU(curRNumInLoop, curCNumInLoop);
    CopyOutU(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::ProcessPerCoreU() {
    const int64_t loopCount = tilingData_.rRcLoopCount;

    int64_t curRNumInLoop = tilingData_.rRcNumPerLoop;
    int64_t curCNumInLoop = 0;
    int64_t rLoopIdx = 0;
    // non-tail r loop
    for (rLoopIdx = 0; rLoopIdx < loopCount - 1; ++rLoopIdx) {
        CopyInUr(rLoopIdx, curRNumInLoop);
        int64_t cLoopIdx = 0;
        // non-tail c loop
        for (cLoopIdx = 0; cLoopIdx < tilingData_.cRcLoopCount - 1; ++cLoopIdx) {
            curCNumInLoop = tilingData_.cRcNumPerLoop;
            CalcU(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
        }

        // tail c loop
        curCNumInLoop = tilingData_.cRcNumTailLoop;
        CalcU(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);

        inQueR_.FreeTensor(rRcLocalTensor_);
    }

    // tail r loop
    curRNumInLoop = tilingData_.rRcNumTailLoop;
    if (curRNumInLoop % NUM_PER_BLOCK == 0) {
        CopyInUr(rLoopIdx, curRNumInLoop);
    } else {
        CopyInNotAlignedUr(rLoopIdx, curRNumInLoop);
    }

    int64_t cLoopIdx = 0;
    for (cLoopIdx = 0; cLoopIdx < tilingData_.cRcLoopCount - 1; ++cLoopIdx) {
        curCNumInLoop = tilingData_.cRcNumPerLoop;
        CalcU(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
    }

    // tail c loop
    curCNumInLoop = tilingData_.cRcNumTailLoop;
    CalcU(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);

    inQueR_.FreeTensor(rRcLocalTensor_);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::ProcessTailCoreU() {
    const int64_t loopCount = tilingData_.rRcLoopCountTailCore;

    int64_t curRNumInLoop = tilingData_.rRcNumPerLoop;
    int64_t curCNumInLoop = 0;
    int64_t rLoopIdx = 0;
    // non-tail r loop
    for (rLoopIdx = 0; rLoopIdx < loopCount - 1; ++rLoopIdx) {
        CopyInUr(rLoopIdx, curRNumInLoop);
        int64_t cLoopIdx = 0;
        // non-tail c loop
        for (cLoopIdx = 0; cLoopIdx < tilingData_.cRcLoopCount - 1; ++cLoopIdx) {
            curCNumInLoop = tilingData_.cRcNumPerLoop;
            CalcU(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
        }

        // tail c loop
        curCNumInLoop = tilingData_.cRcNumTailLoop;
        CalcU(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
        inQueR_.FreeTensor(rRcLocalTensor_);
    }

    // tail r loop
    curRNumInLoop = tilingData_.rRcNumTailLoopTailCore;
    if (curRNumInLoop % NUM_PER_BLOCK == 0) {
        CopyInUr(rLoopIdx, curRNumInLoop);
    } else {
        CopyInNotAlignedUr(rLoopIdx, curRNumInLoop);
    }
    int64_t cLoopIdx = 0;
    for (cLoopIdx = 0; cLoopIdx < tilingData_.cRcLoopCount - 1; ++cLoopIdx) {
        curCNumInLoop = tilingData_.cRcNumPerLoop;
        CalcU(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
    }

    // tail loop
    curCNumInLoop = tilingData_.cRcNumTailLoop;
    CalcU(rLoopIdx, cLoopIdx, curRNumInLoop, curCNumInLoop);
    inQueR_.FreeTensor(rRcLocalTensor_);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::ProcessU() {
    if (blockIdx_ >= tilingData_.rRcCoreNumToUse) {
        return;
    }

    // calc denominator, not in cycle
    T denominator = beta2_ * sumR_  / N_ + (1 - beta2_) * sumGradRc_ / (M_ * N_);
    rcCoefficient_ = 1 / denominator;

    rRcBlockOffset_ = blockIdx_ * tilingData_.rRcNumPerCore;

    const bool isTailCore = (blockIdx_ == tilingData_.rRcCoreNumToUse - 1);
    if (!isTailCore) {
        this->ProcessPerCoreU();
    } else {
        this->ProcessTailCoreU();
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::CopyInGrad(int rLoopIdx, int cLoopIdx,
                                                     int64_t curRNumInLoop, int64_t curCNumInLoop) {
    LocalTensor<T> gradLocal = inQueGrad_.AllocTensor<T>();
    int64_t startRowNum = rRcBlockOffset_ + rLoopIdx * tilingData_.rRcNumPerLoop;
    int64_t startColumnNum = cLoopIdx * tilingData_.cRcNumPerLoop;
    int64_t startOffset = startRowNum * tilingData_.m + startColumnNum;
    int64_t tmp = CeilDiv(curCNumInLoop, NUM_PER_BLOCK);
    curCNumInLoop = tmp * NUM_PER_BLOCK;
    for (int64_t i = 0; i < curRNumInLoop; ++i) {
        int64_t srcOffset = startOffset + i * tilingData_.m;
        int64_t dstOffset = i * curCNumInLoop;
        // aligned
        DataCopy(gradLocal[dstOffset], gradGm_[srcOffset], curCNumInLoop);
    }
    inQueGrad_.EnQue(gradLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::CopyOutU(int rLoopIdx, int cLoopIdx,
                                                   int64_t curRNumInLoop, int64_t curCNumInLoop) {
    LocalTensor<T> uLocal = outQueU_.DeQue<T>();
    int64_t startRowNum = rRcBlockOffset_ + rLoopIdx * tilingData_.rRcNumPerLoop;
    int64_t startColumnNum = cLoopIdx * tilingData_.cRcNumPerLoop;
    int64_t startOffset = startRowNum * tilingData_.m + startColumnNum;
    if (curCNumInLoop % NUM_PER_BLOCK == 0) {
        for (int64_t i = 0; i < curRNumInLoop; ++i) {
            int64_t dstOffset = startOffset + i * tilingData_.m;
            int64_t srcOffset = i * curCNumInLoop;
            // aligned
            DataCopy(uGm_[dstOffset], uLocal[srcOffset], curCNumInLoop);
        }
    } else {
        int64_t alignedCNum = NUM_PER_BLOCK * CeilDiv(curCNumInLoop, NUM_PER_BLOCK);
        uint16_t blockCount = 1;
        uint16_t blockLen = curCNumInLoop * sizeof(T);
        uint16_t srcStride = 0;
        uint16_t dstStride = 0;
        DataCopyParams dataCopyParams {blockCount, blockLen, srcStride, dstStride};
        for (int64_t i = 0; i < curRNumInLoop; ++i) {
            int64_t dstOffset = startOffset + i * tilingData_.m;
            int64_t srcOffset = i * alignedCNum;
            DataCopyPad(uGm_[dstOffset], uLocal[srcOffset], dataCopyParams);
        }
    }

    outQueU_.FreeTensor(uLocal);

    LocalTensor<T> sumSquareUOutLocal = outQueSumSquareU_.DeQue<T>();

    int64_t blockOffset = blockIdx_ * (tilingData_.rRcLoopCount + 1) * (tilingData_.cRcLoopCount + 1);
    uint16_t blockCount = 1;
    uint16_t blockLen = sizeof(float);
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    DataCopyParams copyParams {blockCount, blockLen, srcStride, dstStride};
    DataCopyPad(sumSquareUWorkspace_[blockOffset + sumUOffset_], sumSquareUOutLocal, copyParams);
    sumUOffset_ = sumUOffset_ + 1;
    outQueSumSquareU_.FreeTensor(sumSquareUOutLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::CopyInNotAlignedGrad(int rLoopIdx, int cLoopIdx,
                                                               int64_t curRNumInLoop, int64_t curCNumInLoop) {
    LocalTensor<T> gradLocal = inQueGrad_.AllocTensor<T>();
    int64_t startRowNum = rRcBlockOffset_ + rLoopIdx * tilingData_.rRcNumPerLoop;
    int64_t startColumnNum = cLoopIdx * tilingData_.cRcNumPerLoop;
    int64_t startOffset = startRowNum * tilingData_.m + startColumnNum;
    int64_t alignedCNumInLoop = NUM_PER_BLOCK * CeilDiv(curCNumInLoop, NUM_PER_BLOCK);
    for (int64_t i = 0; i < curRNumInLoop; ++i) {
        int64_t srcOffset = startOffset + i * tilingData_.m;
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
        DataCopyPad(gradLocal[dstOffset], gradGm_[srcOffset], dataCopyParams, padParams);
    }

    inQueGrad_.EnQue(gradLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::CopyInNotAlignedUr(int loopIdx, int64_t dataCount) {
    uint16_t blockCount = 1;
    uint16_t blockLen = dataCount * sizeof(T);
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    DataCopyParams dataCopyParams {blockCount, blockLen, srcStride, dstStride};
    uint8_t leftPadding = 0;
    uint8_t rightPadding = NUM_PER_BLOCK * CeilDiv(dataCount, NUM_PER_BLOCK) - dataCount;
    uint64_t paddingValue = 0;
    DataCopyPadParams padParams {true, leftPadding, rightPadding, paddingValue};

    LocalTensor<T> rLocal = inQueR_.AllocTensor<T>();
    int64_t offset = rRcBlockOffset_ + loopIdx * tilingData_.rRcNumPerLoop;
    DataCopyPad(rLocal, rOutGm_[offset], dataCopyParams, padParams);
    inQueR_.EnQue(rLocal);
    rRcLocalTensor_ = inQueR_.DeQue<T>();
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::CopyInNotAlignedUc(int loopIdx, int64_t dataCount) {
    uint16_t blockCount = 1;
    uint16_t blockLen = dataCount * sizeof(T);
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    DataCopyParams dataCopyParams {blockCount, blockLen, srcStride, dstStride};
    uint8_t leftPadding = 0;
    uint8_t rightPadding = NUM_PER_BLOCK * CeilDiv(dataCount, NUM_PER_BLOCK) - dataCount;
    uint64_t paddingValue = 0;
    DataCopyPadParams padParams {true, leftPadding, rightPadding, paddingValue};

    LocalTensor<T> cLocal = inQueC_.AllocTensor<T>();
    int64_t offset = loopIdx * tilingData_.cRcNumPerLoop;
    DataCopyPad(cLocal, cOutGm_[offset], dataCopyParams, padParams);
    inQueC_.EnQue(cLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::CopyInUr(int loopIdx, int64_t dataCount) {
    LocalTensor<T> rLocal = inQueR_.AllocTensor<T>();
    int64_t offset = rRcBlockOffset_ + loopIdx * tilingData_.rRcNumPerLoop;
    // aligned
    DataCopy(rLocal, rOutGm_[offset], dataCount);
    inQueR_.EnQue(rLocal);
    rRcLocalTensor_ = inQueR_.DeQue<T>();
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::CopyInUc(int loopIdx, int64_t dataCount) {
    LocalTensor<T> cLocal = inQueC_.AllocTensor<T>();
    int64_t offset = loopIdx * tilingData_.cRcNumPerLoop;
    // aligned
    DataCopy(cLocal, cOutGm_[offset], dataCount);
    inQueC_.EnQue(cLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::BroadcastR(LocalTensor<T> &dst, LocalTensor<T> &src, int64_t numR, int64_t numC) {
    for (int i = 0; i < numC; ++i) {
        DataCopy(dst[i * numR], src, numR);
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::TransposeR(LocalTensor<T> &dst, LocalTensor<T> &src, int64_t numR, int64_t numC) {
    ConfusionTransposeTiling tiling;
    GetConfusionTransposeTiling(numR, numC, src.GetSize(), sizeof(T), tiling);
    ConfusionTranspose(dst, src, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::MulRC(LocalTensor<T> &dst,
                                                LocalTensor<T> &r, LocalTensor<T> &c,
                                                int64_t numR, int64_t numC) {
    // r:(numR, numC)  c:(numC,)
    int64_t rcStride = (numC * sizeof(T)) / BLOCK_SIZE;
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
__aicore__ inline void ApplyCamePart2<T>::CalcRcCycleMode(LocalTensor<T> &dst, LocalTensor<T> &src,
                                                          LocalTensor<T> &srcScalar,
                                                          int64_t curRNumInLoop, int64_t curCNumInLoop) {
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
__aicore__ inline void ApplyCamePart2<T>::ComputeU(int64_t curRNumInLoop, int64_t curCNumInLoop) {
    int64_t alignedCNum = NUM_PER_BLOCK * CeilDiv(curCNumInLoop, NUM_PER_BLOCK);
    int64_t oriCNum = curCNumInLoop;
    curCNumInLoop = alignedCNum;
    // r*c
    LocalTensor<T> &rLocal = rRcLocalTensor_;
    LocalTensor<T> cLocal = inQueC_.DeQue<T>();
    LocalTensor<T> gradLocal = inQueGrad_.DeQue<T>();
    LocalTensor<T> uOutLocal = outQueU_.AllocTensor<T>();
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
        pipe_barrier(PIPE_V);
        // ub2: r --> broadcast and transpose
        TransposeR(ub2, ub1, curRNumInLoop, curCNumInLoop);
        pipe_barrier(PIPE_V);
        // ub3: r*c
        MulRC(ub3, ub2, cLocal, curRNumInLoop, curCNumInLoop);
    }

    // ub1: rc*coefficient
    pipe_barrier(PIPE_V);
    Muls(ub1, ub3, rcCoefficient_, dataCount);
    // ub2: Rsqrt(ub1)
    pipe_barrier(PIPE_V);
    Sqrt(ub2, ub1, dataCount);
    pipe_barrier(PIPE_V);
    Div(uOutLocal, gradLocal, ub2, dataCount);

    // calc sum_square_u
    pipe_barrier(PIPE_V);
    Mul(ub3, uOutLocal, uOutLocal, dataCount);
    pipe_barrier(PIPE_V);
    ReduceSumU(ub3, dataCount, curRNumInLoop, oriCNum);

    outQueU_.EnQue<T>(uOutLocal);
    inQueC_.FreeTensor(cLocal);
    inQueGrad_.FreeTensor(gradLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::ReduceSumU(LocalTensor<T> &src, int64_t dataCount,
                                                     int64_t curRNumInLoop, int64_t curCNumInLoop) {
    LocalTensor<T> sumSquareUOutLocal = outQueSumSquareU_.AllocTensor<T>();
    LocalTensor<T> workLocal = reduceSumWorkQueue_.AllocTensor<T>();
    int64_t alignedCNum = NUM_PER_BLOCK * CeilDiv(curCNumInLoop, NUM_PER_BLOCK);
    T sum = 0;
    for (int i = 0; i < curRNumInLoop; ++i) {
        ReduceSum(sumSquareUOutLocal, src[i * alignedCNum], workLocal, curCNumInLoop);
        sum += sumSquareUOutLocal.GetValue(0);
    }
    sumSquareUOutLocal.SetValue(0, sum);
    outQueSumSquareU_.EnQue<T>(sumSquareUOutLocal);
    reduceSumWorkQueue_.FreeTensor(workLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart2<T>::GetConfusionTransposeTiling(int64_t numR, int64_t numC, const uint32_t stackBufferSize,
                                                                      const uint32_t typeSize, ConfusionTransposeTiling &tiling)
{
    (void)stackBufferSize;
    uint32_t blockSize = ONE_BLK_SIZE /typeSize;
    uint32_t height = numC;
    uint32_t width = numR;
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