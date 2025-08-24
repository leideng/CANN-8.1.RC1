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
 * \file apply_came_part1_fp32.h
 * \brief
 */
#ifndef APPLY_CAME_PART1_FP32
#define APPLY_CAME_PART1_FP32

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace ApplyCamePart1 {

using namespace AscendC;

template <typename T>
class ApplyCamePart1FP32 {
public:
    __aicore__ inline ApplyCamePart1FP32(){};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR eps, GM_ADDR sum_grad_r, GM_ADDR sum_grad_c,
                                GM_ADDR sum_grad_rc, GM_ADDR workspace,
                                const ApplyCamePart1TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SyncAllCore(GM_ADDR sum_grad_r, GM_ADDR sum_grad_c, GM_ADDR sum_grad_rc);
    __aicore__ inline void ParseTilingData(const ApplyCamePart1TilingData* tilingData);
    __aicore__ inline void ClearAcculateMatrix();
    __aicore__ inline void UpdateRepeatTimes();
    __aicore__ inline void CopyIn(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void Compute(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void CopyOut(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void CopyInNormal(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes, LocalTensor<float> gradLocal);
    __aicore__ inline void CopyInLast(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes, LocalTensor<float> gradLocal);
    __aicore__ inline void ComputeR(int64_t curRepeatTimes, LocalTensor<float> accuComTmpUb);
    __aicore__ inline void ComputeC(int64_t curRepeatTimes, LocalTensor<float> gradSqrtTmpUb);
    __aicore__ inline void ComputeRC(int64_t curRepeatTimes, LocalTensor<float> accuComTmpUb, LocalTensor<float> mComTmpUb);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> gradQueue_;
    TQue<QuePosition::VECIN, 1> epsQueue_;

    TQue<QuePosition::VECOUT, 1> sumGradRQueue_;
    TQue<QuePosition::VECOUT, 1> sumGradCQueue_;
    TQue<QuePosition::VECOUT, 1> sumGradRCQueue_;

    TBuf<QuePosition::VECCALC> accuComTmpBuf_;
    TBuf<QuePosition::VECCALC> gradCastTmpBuf_;
    TBuf<QuePosition::VECCALC> gradSqrtTmpBuf_;
    TBuf<QuePosition::VECCALC> mComTmpBuf_;

    GlobalTensor<float> gmGrad_;
    GlobalTensor<float> gmEps_;
    GlobalTensor<float> gmSumGradR_;
    GlobalTensor<float> gmSumGradC_;
    GlobalTensor<float> gmSumGradRC_;

    GlobalTensor<float> workspaceSumGradR_;
    GlobalTensor<float> workspaceSumGradC_;
    GlobalTensor<float> workspaceSumGradRC_;

    // multi-core sync
    GlobalTensor<int32_t> syncGlobal_;
    TQue<QuePosition::VECIN, 1> syncWorkQueue_;
    GM_ADDR workspaceAddr_;

    // tiling params
    int64_t N{0};
    int64_t M{0};

    int64_t nLoopNormCore_{0};
    int64_t nLoopTailCore_{0};

    int64_t nNormalCoreNum_{0};
    int64_t nTailCoreNum_{0};

    int64_t mNormalCoreNum_{0};
    int64_t mTailCoreNum_{0};

    int64_t totalCoreNum_{0};
    int64_t usedCoreNum_{0};

    int64_t nCoreNum_{0};
    int64_t mCoreNum_{0};
    int64_t mLoopNumCore_{0};

    int64_t ubInputSize_{0};
    int64_t ubOutputSize_{0};

    int64_t inputIdx_{0};
    int64_t inputNum_{0};

    int64_t workspace_{0};
    bool IsContainsTailN{false};
    bool IsContainsTailM{false};

    const int64_t ONCE_HANDLE_NUM64{64};
    const int64_t ONCE_ONE_SIZE8{8};
    const int64_t ONCE_ALGN_NUM{32 / sizeof(float)};
    const int64_t MAX_BOUND_VAL{65535};

    constexpr static uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
};

template <typename T>
__aicore__ inline void ApplyCamePart1FP32<T>::ParseTilingData(const ApplyCamePart1TilingData* tilingData)
{
    // 总维度[N, M]
    N = tilingData->N;
    M = tilingData->M;

    // 单核矩阵维度 [nNormalCoreNum_, nTailCoreNum_]
    nNormalCoreNum_ = tilingData->nNormalCoreNum;
    nTailCoreNum_ = tilingData->nTailCoreNum;

    // 单核矩阵维度 [mNormalCoreNum_, mTailCoreNum_]
    mNormalCoreNum_ = tilingData->mNormalCoreNum;
    mTailCoreNum_ = tilingData->mTailCoreNum;

    // 循环次数
    nLoopNormCore_ = tilingData->nLoopNormCore;
    nLoopTailCore_ = tilingData->nLoopTailCore;
    mLoopNumCore_ = tilingData->mLoopNumCore;

    // 使用核数 && 总核数 [totalCoreNum_, usedCoreNum_]
    totalCoreNum_ = tilingData->totalCoreNum;
    usedCoreNum_ = tilingData->usedCoreNum;

    // 行列方向的核 [nCoreNum_, mCoreNum_]
    nCoreNum_ = tilingData->nCoreNum;
    mCoreNum_ = tilingData->mCoreNum;

    // 分配的UB
    ubInputSize_ = tilingData->ubInputSize;
    ubOutputSize_ = tilingData->ubOutputSize;

    inputNum_ = tilingData->inputNum;
    workspace_ = tilingData->userWorkspaceSize;
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP32<T>::Init(GM_ADDR grad, GM_ADDR eps, GM_ADDR sum_grad_r, GM_ADDR sum_grad_c, GM_ADDR sum_grad_rc, GM_ADDR workspace, const ApplyCamePart1TilingData* tilingData)
{
    // 初始化tiling
    ParseTilingData(tilingData);

    // workspace地址
    workspaceAddr_ = workspace;

    // 清零gmOutput
    SyncAllCore(sum_grad_r, sum_grad_c, sum_grad_rc);

    // gmInput分核 && 输入偏移初始化
    inputIdx_ = GetBlockIdx() / mCoreNum_ * ONCE_HANDLE_NUM64 * nLoopNormCore_ * M + GetBlockIdx() % mCoreNum_ * mNormalCoreNum_;
    gmGrad_.SetGlobalBuffer((__gm__ float *)grad + inputIdx_);
    gmEps_.SetGlobalBuffer((__gm__ float *)eps, 1);

    // buffer申请初始化
    pipe.InitBuffer(gradQueue_, 1, ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64 * sizeof(float));
    pipe.InitBuffer(epsQueue_, 1, ONCE_ONE_SIZE8 * sizeof(float));

    pipe.InitBuffer(sumGradRQueue_, 1, ONCE_HANDLE_NUM64 * sizeof(float));
    pipe.InitBuffer(sumGradCQueue_, 1, ONCE_HANDLE_NUM64 * sizeof(float));
    pipe.InitBuffer(sumGradRCQueue_, 1, ONCE_ONE_SIZE8 * sizeof(float));

    // 更新repeatTimes
    UpdateRepeatTimes();

    // 缓存buf空间清零
    ClearAcculateMatrix();
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP32<T>::SyncAllCore(GM_ADDR sum_grad_r, GM_ADDR sum_grad_c, GM_ADDR sum_grad_rc)
{
    // set workspace as 0, each core handle workspace 32 bytes
    constexpr int32_t EACH_CORE_HANDLE_NUM = 32 / sizeof(int32_t);
    gmSumGradR_.SetGlobalBuffer((__gm__ float *)sum_grad_r + GetBlockIdx() / mCoreNum_ * ONCE_HANDLE_NUM64 * nLoopNormCore_);
    gmSumGradC_.SetGlobalBuffer((__gm__ float *)sum_grad_c);
    gmSumGradRC_.SetGlobalBuffer((__gm__ float *)sum_grad_rc);

    int32_t workspaceOffsets = 0;
    workspaceSumGradRC_.SetGlobalBuffer((__gm__ float*)workspaceAddr_ + workspaceOffsets);
    int32_t rcOffsets = (((usedCoreNum_ - 1) *  nLoopNormCore_ + nLoopTailCore_) * mLoopNumCore_ + 128 -1) / 128 * 128;
    workspaceOffsets = workspaceOffsets + rcOffsets;
    workspaceSumGradC_.SetGlobalBuffer((__gm__ float*)workspaceAddr_ + workspaceOffsets);

    // set workspace for sync
    pipe.InitBuffer(syncWorkQueue_, 1, totalCoreNum_ * 8 * sizeof(int32_t));

    constexpr float initValue = 0.0;
    GlobalTensor<float> gmSumGradRToClear;
    GlobalTensor<float> gmSumGradCToClear;
    GlobalTensor<float> gmSumGradRCToClear;
    gmSumGradRToClear.SetGlobalBuffer((__gm__ float *)sum_grad_r);
    gmSumGradCToClear.SetGlobalBuffer((__gm__ float *)sum_grad_c);
    gmSumGradRCToClear.SetGlobalBuffer((__gm__ float *)sum_grad_rc);
    if ((GetBlockIdx() + 1) < GetBlockNum()) {
        InitOutput<float>(gmSumGradRToClear[N / GetBlockNum() * GetBlockIdx()], N / GetBlockNum(), initValue);
    }
    if ((GetBlockIdx() + 1) == GetBlockNum()) {
        InitOutput<float>(gmSumGradRToClear[N / GetBlockNum() * GetBlockIdx()], N - N / GetBlockNum() * (GetBlockNum() - 1), initValue);
    }
    InitOutput<float>(gmSumGradCToClear, M, initValue);
    InitOutput<float>(gmSumGradRCToClear, 1, initValue);

    SyncAll();
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP32<T>::UpdateRepeatTimes()
{
    IsContainsTailN = false;
    if (mNormalCoreNum_ % ONCE_HANDLE_NUM64) {
        IsContainsTailN = true;
    }

    IsContainsTailM = false;
    if (nTailCoreNum_ % ONCE_HANDLE_NUM64) {
        IsContainsTailM = true;
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP32<T>::CopyInLast(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes, LocalTensor<float> gradLocal)
{
    Duplicate<float>(gradLocal, (float)0.0, curRepeatTimes * ONCE_HANDLE_NUM64);
    int64_t srcStrideLast = (mNormalCoreNum_ - mNormalCoreNum_ % ONCE_HANDLE_NUM64) * sizeof(float);
    int64_t dstStrideLast = (ONCE_HANDLE_NUM64 - mNormalCoreNum_ % ONCE_HANDLE_NUM64) * sizeof(float) / 32;
    int64_t blockLenLast = mNormalCoreNum_ % ONCE_HANDLE_NUM64 * sizeof(float);
    int64_t algnNumLast = (mNormalCoreNum_ + ONCE_ALGN_NUM - 1) / ONCE_ALGN_NUM * ONCE_ALGN_NUM - mNormalCoreNum_;
    if (srcStrideLast > MAX_BOUND_VAL) {
        for (int64_t i = 0; i < curRepeatTimes; i++) {
            DataCopyParams copyParamsLast {
                (uint16_t)1,
                (uint16_t)(blockLenLast),
                (uint16_t)0,
                (uint16_t)0
            };
            DataCopyPadParams padParamsLast {
                true,
                (uint8_t)0,
                (uint8_t)(algnNumLast),
                (uint8_t)0
            };
            DataCopyPad(gradLocal[i * ONCE_HANDLE_NUM64], gmGrad_[(mNormalCoreNum_ - mNormalCoreNum_ % ONCE_HANDLE_NUM64) + i * mNormalCoreNum_], copyParamsLast, padParamsLast);
        }
    } else {
        DataCopyParams copyParamsLast {
            (uint16_t)(curRepeatTimes),
            (uint16_t)(blockLenLast),
            (uint16_t)(srcStrideLast),
            (uint16_t)(dstStrideLast)
        };
        DataCopyPadParams padParamsLast {
            true,
            (uint8_t)0,
            (uint8_t)(algnNumLast),
            (uint8_t)0
        };
        DataCopyPad(gradLocal, gmGrad_[nLoopIdx * ONCE_HANDLE_NUM64 * M + mLoopIdx * ONCE_HANDLE_NUM64], copyParamsLast, padParamsLast);
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP32<T>::CopyInNormal(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes, LocalTensor<float> gradLocal)
{
    int64_t srcStrideNormal = (mNormalCoreNum_ - ONCE_HANDLE_NUM64) * sizeof(float);
    int64_t blockLenNormal = ONCE_HANDLE_NUM64 * sizeof(float);
    if (srcStrideNormal > MAX_BOUND_VAL) {
        for (int64_t i = 0; i < curRepeatTimes; i++) {
                DataCopyParams copyParamsNormal {
                (uint16_t)1,
                (uint16_t)(blockLenNormal),
                (uint16_t)0,
                (uint16_t)0
            };
            DataCopyPadParams padParamsNormal {false, 0, 0, 0};
            DataCopyPad(gradLocal[i * ONCE_HANDLE_NUM64], gmGrad_[mLoopIdx * ONCE_HANDLE_NUM64 + i * mNormalCoreNum_], copyParamsNormal, padParamsNormal);
        }
    } else {
        DataCopyParams copyParamsNormal {
            (uint16_t)(curRepeatTimes),
            (uint16_t)(blockLenNormal),
            (uint16_t)(srcStrideNormal),
            (uint16_t)0
        };
        DataCopyPadParams padParamsNormal {false, 0, 0, 0};
        DataCopyPad(gradLocal, gmGrad_[nLoopIdx * ONCE_HANDLE_NUM64 * M + mLoopIdx * ONCE_HANDLE_NUM64], copyParamsNormal, padParamsNormal);
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP32<T>::CopyIn(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes)
{
    LocalTensor<float> gradLocal = gradQueue_.AllocTensor<float>();
    LocalTensor<float> epsLocal = epsQueue_.AllocTensor<float>();

    if (mLoopIdx == (mLoopNumCore_ - 1) && IsContainsTailN) {
        CopyInLast(nLoopIdx, mLoopIdx, curRepeatTimes, gradLocal);
    } else {
        CopyInNormal(nLoopIdx, mLoopIdx, curRepeatTimes, gradLocal);
    }
    DataCopyPad(epsLocal, gmEps_, {1, sizeof(float), 0, 0, 0}, {false, 0, 0, 0});
    gradQueue_.EnQue(gradLocal);
    epsQueue_.EnQue(epsLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP32<T>::Compute(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes)
{
    LocalTensor<float> gradLocal = gradQueue_.DeQue<float>();
    LocalTensor<float> epsLocal = epsQueue_.DeQue<float>();

    LocalTensor<float> gradCastTmpUb = gradCastTmpBuf_.Get<float>();
    LocalTensor<float> gradSqrtTmpUb = gradSqrtTmpBuf_.Get<float>();
    LocalTensor<float> accuComTmpUb = accuComTmpBuf_.Get<float>();
    LocalTensor<float> mComTmpUb = mComTmpBuf_.Get<float>();

    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    float eps = epsLocal.GetValue(0);

    int64_t calCount = curRepeatTimes * ONCE_HANDLE_NUM64;
    if (sizeof(T) != 4) {
        Cast(gradCastTmpUb, gradLocal, RoundMode::CAST_NONE, calCount);
    } else {
        gradCastTmpUb = gradLocal;
    }
    pipe_barrier(PIPE_V);
    Duplicate(gradSqrtTmpUb, (float)0.0, ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64);
    pipe_barrier(PIPE_V);
    Mul(gradSqrtTmpUb, gradCastTmpUb, gradCastTmpUb, calCount);

    pipe_barrier(PIPE_V);
    if (mLoopIdx == (mLoopNumCore_ - 1) && IsContainsTailN) {
        Adds(gradSqrtTmpUb, gradSqrtTmpUb, eps,
             (uint16_t)(mNormalCoreNum_ - (mLoopNumCore_ - 1) * ONCE_HANDLE_NUM64),
             curRepeatTimes, { 1, 1, 8, 8});
    } else {
        Adds(gradSqrtTmpUb, gradSqrtTmpUb, eps, calCount);
    }

    pipe_barrier(PIPE_V);
    Add(accuComTmpUb, gradSqrtTmpUb, accuComTmpUb, curRepeatTimes * ONCE_HANDLE_NUM64);

    LocalTensor<float> sumGradRCLocal = sumGradRCQueue_.AllocTensor<float>();
    pipe_barrier(PIPE_V);
    sumGradRCQueue_.EnQue<float>(sumGradRCLocal);

    if (mLoopIdx == (mLoopNumCore_ - 1)) {
        ComputeR(curRepeatTimes, accuComTmpUb);

        pipe_barrier(PIPE_V);
        Duplicate(accuComTmpUb, (float)0, curRepeatTimes * ONCE_HANDLE_NUM64);
        Duplicate(mComTmpUb, (float)0, ONCE_HANDLE_NUM64);
    }
    ComputeC(curRepeatTimes, gradSqrtTmpUb);
    LocalTensor<float> sumGradCLocal = sumGradCQueue_.DeQue<float>();
    pipe_barrier(PIPE_V);
    ReduceSum(sumGradRCLocal, sumGradCLocal, mComTmpUb, 64);
    sumGradCQueue_.EnQue<float>(sumGradCLocal);

    gradQueue_.FreeTensor(gradLocal);
    epsQueue_.FreeTensor(epsLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP32<T>::ComputeR(int64_t curRepeatTimes, LocalTensor<float> accuComTmpUb)
{
    LocalTensor<float> sumGradRLocal = sumGradRQueue_.AllocTensor<float>();
    pipe_barrier(PIPE_V);
    uint64_t maskOfRed_R = ONCE_HANDLE_NUM64;
    uint64_t repeatTimesOfRed_R = curRepeatTimes;
    uint8_t dstRepStrideOfRed_R = 1;
    uint8_t srcBlkStrideOfRed_R = 1;
    uint8_t srcRepStrideOfRed_R = 8;
    WholeReduceSum<float>(sumGradRLocal, accuComTmpUb, maskOfRed_R, repeatTimesOfRed_R, dstRepStrideOfRed_R, srcBlkStrideOfRed_R, srcRepStrideOfRed_R);
    sumGradRQueue_.EnQue<float>(sumGradRLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP32<T>::ComputeC(int64_t curRepeatTimes, LocalTensor<float> gradSqrtTmpUb)
{
    LocalTensor<float> sumGradCLocal = sumGradCQueue_.AllocTensor<float>();
    if (((GetBlockIdx() / mCoreNum_ + 1) == nCoreNum_) && ((curRepeatTimes & curRepeatTimes - 1) != 0)) {
        pipe_barrier(PIPE_V);
        Duplicate(gradSqrtTmpUb[ONCE_HANDLE_NUM64 * curRepeatTimes],
                  (float)0.0,
                  (ONCE_HANDLE_NUM64 - curRepeatTimes) * ONCE_HANDLE_NUM64);
    }
    for (int32_t i = 1; i < ONCE_HANDLE_NUM64; i *= 2) {
        pipe_barrier(PIPE_V);
        Add(
            gradSqrtTmpUb[0],
            gradSqrtTmpUb[ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64 / 2 / i],
            gradSqrtTmpUb[0],
            ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64 / 2 / i
        );
    }
    pipe_barrier(PIPE_V);
    Adds(sumGradCLocal, gradSqrtTmpUb, (float)0, ONCE_HANDLE_NUM64);
    sumGradCQueue_.EnQue<float>(sumGradCLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP32<T>::ComputeRC(int64_t curRepeatTimes, LocalTensor<float> accuComTmpUb, LocalTensor<float> mComTmpUb)
{
    LocalTensor<float> sumGradRCLocal = sumGradRCQueue_.AllocTensor<float>();
    pipe_barrier(PIPE_V);
    uint64_t maskOfAdd_RC = ONCE_HANDLE_NUM64;
    uint64_t repeatTimesOfAdd_RC = curRepeatTimes;
    uint8_t dstBlkStrideOfAdd_RC = 1;
    uint8_t src0BlkStrideOfAdd_RC = 1;
    uint8_t src1BlkStrideOfAdd_RC = 1;
    uint8_t dstRepStrideOfAdd_RC = 0;
    uint8_t src0RepStrideOfAdd_RC = 8;
    uint8_t src1RepStrideOfAdd_RC = 0;
    BinaryRepeatParams repeatParams_RC{dstBlkStrideOfAdd_RC, src0BlkStrideOfAdd_RC, src1BlkStrideOfAdd_RC, dstRepStrideOfAdd_RC, src0RepStrideOfAdd_RC, src1RepStrideOfAdd_RC};
    Add(mComTmpUb, accuComTmpUb, mComTmpUb, maskOfAdd_RC, repeatTimesOfAdd_RC, repeatParams_RC);

    pipe_barrier(PIPE_V);
    uint64_t maskOfRed_RC = ONCE_HANDLE_NUM64;
    uint64_t repeatTimesOfRed_RC = 1;
    uint8_t dstRepStrideOfRed_RC = 1;
    uint8_t srcBlkStrideOfRed_RC = 1;
    uint8_t srcRepStrideOfRed_RC = 1;
    WholeReduceSum<float>(sumGradRCLocal, mComTmpUb, maskOfRed_RC, repeatTimesOfRed_RC, dstRepStrideOfRed_RC, srcBlkStrideOfRed_RC, srcRepStrideOfRed_RC);

    sumGradRCQueue_.EnQue<float>(sumGradRCLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP32<T>::CopyOut(int64_t nLoopIdx, int64_t mLoopIdx, int64_t curRepeatTimes)
{
    if (mLoopIdx == (mLoopNumCore_ - 1)) {
        LocalTensor<float> sumGradRLocal = sumGradRQueue_.DeQue<float>();
        DataCopyParams copyParams_R {
            1, (uint16_t)(curRepeatTimes * sizeof(float)), 0, 0
        };
        SetAtomicAdd<float>();
        DataCopyPad(gmSumGradR_[nLoopIdx * ONCE_HANDLE_NUM64], sumGradRLocal, copyParams_R);
        SetAtomicNone();
        sumGradRQueue_.FreeTensor(sumGradRLocal);
    }

    LocalTensor<float> sumGradRCLocal = sumGradRCQueue_.DeQue<float>();
    int64_t offset = (GetBlockIdx() * mLoopNumCore_ * nLoopNormCore_ + nLoopIdx * mLoopNumCore_ + mLoopIdx) * 1;

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    DataCopyPad(workspaceSumGradRC_[offset], sumGradRCLocal, {1, (uint16_t)(1 * sizeof(float)), 0, 0});
    sumGradRCQueue_.FreeTensor(sumGradRCLocal);

    offset = (GetBlockIdx() * mLoopNumCore_ * nLoopNormCore_  + nLoopIdx * mLoopNumCore_ + mLoopIdx) * ONCE_HANDLE_NUM64;
    LocalTensor<float> sumGradCLocal = sumGradCQueue_.DeQue<float>();
    DataCopy(workspaceSumGradC_[offset], sumGradCLocal, ONCE_HANDLE_NUM64);
    sumGradCQueue_.FreeTensor(sumGradCLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP32<T>::Process()
{
    if (GetBlockIdx() >= usedCoreNum_) {
        return;
    }

    if ((GetBlockIdx() + 1) != nCoreNum_) {
        for (int64_t n = 0; n < nLoopNormCore_; n++) {
            for (int64_t m = 0; m < mLoopNumCore_; m++) {
                pipe_barrier(PIPE_ALL);
                CopyIn(n, m, ONCE_HANDLE_NUM64);
                Compute(n, m, ONCE_HANDLE_NUM64);
                CopyOut(n, m, ONCE_HANDLE_NUM64);
            }
        }
    } else {
        for (int64_t n = 0; n < nLoopTailCore_ - 1; n++) {
            for (int64_t m = 0; m < mLoopNumCore_; m++) {
                pipe_barrier(PIPE_ALL);
                CopyIn(n, m, ONCE_HANDLE_NUM64);
                Compute(n, m, ONCE_HANDLE_NUM64);
                CopyOut(n, m, ONCE_HANDLE_NUM64);
            }
        }

        int64_t nTailCoreLastLoop = nTailCoreNum_ - (nLoopTailCore_ - 1) * ONCE_HANDLE_NUM64;
        for (int64_t m = 0; m < mLoopNumCore_; m++) {
            pipe_barrier(PIPE_ALL);
            CopyIn(nLoopTailCore_ - 1, m, nTailCoreLastLoop);
            Compute(nLoopTailCore_ - 1, m, nTailCoreLastLoop);
            CopyOut(nLoopTailCore_ - 1, m, nTailCoreLastLoop);
        }
    }

    SyncAll();
}

template <typename T>
__aicore__ inline void ApplyCamePart1FP32<T>::ClearAcculateMatrix()
{
    constexpr float scalarValue = 0;

    pipe.InitBuffer(gradCastTmpBuf_, ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64 * sizeof(float));
    LocalTensor<float> gradCastTmpUb = gradCastTmpBuf_.Get<float>(ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64);
    Duplicate(gradCastTmpUb, scalarValue, ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64);

    pipe.InitBuffer(gradSqrtTmpBuf_, ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64 * sizeof(float));
    LocalTensor<float> gradSqrtTmpUb = gradSqrtTmpBuf_.Get<float>(ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64);
    Duplicate(gradSqrtTmpUb, scalarValue, ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64);

    pipe.InitBuffer(accuComTmpBuf_, ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64 * sizeof(float));
    LocalTensor<float> accuComTmpUb = accuComTmpBuf_.Get<float>(ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64);
    Duplicate(accuComTmpUb, scalarValue, ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64);

    pipe.InitBuffer(mComTmpBuf_, 128 * ONCE_HANDLE_NUM64 * sizeof(float));
    LocalTensor<float> mComTmpUb = mComTmpBuf_.Get<float>(1 * ONCE_HANDLE_NUM64);
    Duplicate(mComTmpUb, scalarValue, 1 * ONCE_HANDLE_NUM64);
}

}  // namespace ApplyCamePart1
#endif  // APPLY_CAME_PART1_FP32
