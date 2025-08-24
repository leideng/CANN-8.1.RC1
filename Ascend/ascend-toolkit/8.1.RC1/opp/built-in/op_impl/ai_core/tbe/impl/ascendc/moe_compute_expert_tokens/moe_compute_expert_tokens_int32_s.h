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
 * \file moe_compute_expert_tokens_int32_s.h
 * \brief
 */
#ifndef MOE_COMPUTE_EXPERT_TOKENS_INT32_S
#define MOE_COMPUTE_EXPERT_TOKENS_INT32_S

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MoeCompute {

using namespace AscendC;

template <typename T>
class MoeComputeExpertTokensInt32S {
public:
    __aicore__ inline MoeComputeExpertTokensInt32S(){};
    __aicore__ inline void Init(GM_ADDR sortExperts,
                                GM_ADDR out,
                                GM_ADDR workspace,
                                const MoeComputeExpertTokensTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SyncAllCore();
    __aicore__ inline void ProcessBefore();
    __aicore__ inline void ProcessAfter();
    __aicore__ inline void CopyInBefore();
    __aicore__ inline void ComputeBefore();
    __aicore__ inline void CopyOutBefore();

    __aicore__ inline void CopyInAfter(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void ComputeAfter(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void CopyOutAfter(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline int64_t Int32AlignmentProcess(int64_t param);
    __aicore__ inline int64_t PadProcessInt32(int64_t param);
    __aicore__ inline void ParseTilingData(const MoeComputeExpertTokensTilingData *tilingData);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> tmpOutQueue_;

    TQue<QuePosition::VECIN, 1> workspaceQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;

    // syncall before
    GlobalTensor<T> gmInput_;
    GlobalTensor<T> gmWorkspace_;
    GlobalTensor<T> gmOutput_;

    TBuf<QuePosition::VECCALC> inputCastTmpBuf_;
    TBuf<QuePosition::VECCALC> outputCastTmpBuf_;
    GM_ADDR workspace_;

    // multi-core sync
    GlobalTensor<int32_t> syncGlobal_;
    TQue<QuePosition::VECIN, 1> syncWorkQueue_;

    // syncall before base param
    int64_t sortedExpertNum_{0};
    int64_t handleNumPerCoreBefore_{0};
    int64_t handleNumTailCoreBefore_{0};
    int64_t handleNum_{0};
    int64_t usedCoreNumBefore_{0};

    // syncall after base param
    // normal core
    int64_t normalCoreHandleNum_{0};
    int64_t normalCoreLoopNum_{0};
    int64_t normalCoreHandleNumPerLoop_{0};
    int64_t normalCoreHandleNumTailLoop_{0};

    // tail core
    int64_t tailCoreHandleNum_{0};
    int64_t tailCoreLoopNum_{0};
    int64_t tailCoreHandleNumPerLoop_{0};
    int64_t tailCoreHandleNumTailLoop_{0};

    int64_t curCoreHandleNumPerLoop_{0};
    int64_t curCoreHandleNumTailLoop_{0};
    int64_t curCoreHandleNum_{0};
    int64_t loopCount_{0};
    int64_t usedCoreNumAfter_{0};

    // input number
    int64_t totalCoreNum_{0};
    int64_t numOfExpert_{0};
    int64_t inputIndex_{0};
    int64_t outputIndex_{0};

    bool isPaddingBefore_{false};
    int64_t rightPaddingBefore_{0};

    bool isPaddingAfter_{false};
    int64_t rightPaddingAfter_{0};

    const int64_t INT32_BYTES{4};
    const int64_t ONCE_ALGN_NUM_INT32{8};
};

template <typename T>
__aicore__ inline int64_t MoeComputeExpertTokensInt32S<T>::PadProcessInt32(int64_t param)
{
    return  ONCE_ALGN_NUM_INT32 - param % ONCE_ALGN_NUM_INT32;
}

template <typename T>
__aicore__ inline int64_t MoeComputeExpertTokensInt32S<T>::Int32AlignmentProcess(int64_t param)
{
    return (param + ONCE_ALGN_NUM_INT32 - 1) / ONCE_ALGN_NUM_INT32 * ONCE_ALGN_NUM_INT32;
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32S<T>::ParseTilingData(
    const MoeComputeExpertTokensTilingData *tilingData)
{
    // 使用核数
    usedCoreNumBefore_ = tilingData->usedCoreNumBefore;
    usedCoreNumAfter_ = tilingData->usedCoreNumAfter;

    // 输入专家个数
    sortedExpertNum_ = tilingData->sortedExpertNum;
    numOfExpert_ = tilingData->numOfExpert;

    // SyncAll前，尾核 & 非尾核 
    handleNumPerCoreBefore_ = tilingData->normalCoreHandleNumBefore;
    handleNumTailCoreBefore_ = tilingData->tailCoreHandleNumBefore;

    // SyncAll后，非尾核
    normalCoreHandleNum_ = tilingData->normalCoreHandleNumAfter;
    normalCoreLoopNum_ = tilingData->normalCoreLoopNumAfter;
    normalCoreHandleNumPerLoop_ = tilingData->normalCoreHandleNumPerLoopAfter;
    normalCoreHandleNumTailLoop_ = tilingData->normalCoreHandleNumTailLoopAfter;

    // SyncAll后，尾核
    tailCoreHandleNum_ = tilingData->tailCoreHandleNumAfter;
    tailCoreLoopNum_ = tilingData->tailCoreLoopNumAfter;
    tailCoreHandleNumPerLoop_ = tilingData->tailCoreHandleNumPerLoopAfter;
    tailCoreHandleNumTailLoop_ = tilingData->tailCoreHandleNumTailLoopAfter;

    // 使用核数信息
    totalCoreNum_ = tilingData->totalCoreNum;
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32S<T>::SyncAllCore()
{
    // set workspace as 0, each core handle workspace 32bytes
    constexpr int32_t EACH_CORE_HANDLE_NUM = 32 / sizeof(int32_t);
    syncGlobal_.SetGlobalBuffer((__gm__ int32_t*)workspace_+ (numOfExpert_ * usedCoreNumBefore_));
    InitOutput<int32_t>(syncGlobal_[EACH_CORE_HANDLE_NUM * GetBlockIdx()], EACH_CORE_HANDLE_NUM, (int32_t)0);

    // set workspace for sync
    pipe_.InitBuffer(syncWorkQueue_, 1, totalCoreNum_ * 8 * sizeof(int32_t));

    // multi-core sync
    SyncAll();
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32S<T>::Init(GM_ADDR sortExperts,
                                                            GM_ADDR out,
                                                            GM_ADDR workspace,
                                                            const MoeComputeExpertTokensTilingData* tilingData)
{
    // init tiling data
    ParseTilingData(tilingData);
    workspace_ = workspace;

    // syncall before
    handleNum_ = GetBlockIdx() != usedCoreNumBefore_ - 1 ? handleNumPerCoreBefore_ : handleNumTailCoreBefore_;
    gmInput_.SetGlobalBuffer((__gm__ T*)sortExperts + GetBlockIdx() * handleNumPerCoreBefore_);
    gmWorkspace_.SetGlobalBuffer((__gm__ T*)workspace);

    // gmWorkspace_清零
    int64_t n = numOfExpert_ * usedCoreNumAfter_;
    int32_t initValue = 0;
    if ((GetBlockIdx() + 1) < GetBlockNum()) {
        InitOutput<int32_t>(gmWorkspace_[n / GetBlockNum() * GetBlockIdx()], n / GetBlockNum(), initValue);
    }
    if ((GetBlockIdx() + 1) == GetBlockNum()) {
        InitOutput<int32_t>(gmWorkspace_[n / GetBlockNum() * GetBlockIdx()], n - n / GetBlockNum() * (GetBlockNum() - 1), initValue);
    }
    SyncAll();

    pipe_.InitBuffer(inputQueue_, 1, Int32AlignmentProcess(handleNum_) * sizeof(T));
    pipe_.InitBuffer(tmpOutQueue_, 1, Int32AlignmentProcess(8 * (numOfExpert_ + 1)) * sizeof(T));

    // syncall before
    if (GetBlockIdx() + 1 == usedCoreNumAfter_) {
        curCoreHandleNumPerLoop_ = tailCoreHandleNumPerLoop_;
        curCoreHandleNumTailLoop_ = tailCoreHandleNumTailLoop_;
        curCoreHandleNum_ = tailCoreHandleNum_;
        loopCount_ = tailCoreLoopNum_;
    } else {
        curCoreHandleNumPerLoop_ = normalCoreHandleNumPerLoop_;
        curCoreHandleNumTailLoop_ = normalCoreHandleNumTailLoop_;
        curCoreHandleNum_ = normalCoreHandleNum_;
        loopCount_ = normalCoreLoopNum_;
    }

    // output 初始化
    outputIndex_ = GetBlockIdx() * normalCoreHandleNum_;
    gmOutput_.SetGlobalBuffer((__gm__ T *)out + outputIndex_, curCoreHandleNum_);

    // 申请buffer空间
    pipe_.InitBuffer(workspaceQueue_, 1, curCoreHandleNumPerLoop_ * Int32AlignmentProcess(usedCoreNumBefore_) * sizeof(int32_t));
    pipe_.InitBuffer(outputQueue_, 1, Int32AlignmentProcess(curCoreHandleNum_) * sizeof(int32_t));

    pipe_.InitBuffer(inputCastTmpBuf_, curCoreHandleNumPerLoop_ * Int32AlignmentProcess(usedCoreNumBefore_) * sizeof(float));
    pipe_.InitBuffer(outputCastTmpBuf_, Int32AlignmentProcess(curCoreHandleNum_) * sizeof(float));
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32S<T>::CopyInBefore()
{
    LocalTensor<T> ubInput = inputQueue_.AllocTensor<T>();
    if (handleNum_ * sizeof(T) % 32) {
        isPaddingBefore_ = true;
        rightPaddingBefore_ = PadProcessInt32(handleNum_);
    }
    DataCopyParams copyParams {
        (uint16_t)1,
        (uint16_t)(handleNum_ * sizeof(T)),
        (uint16_t)0,
        (uint16_t)0
    };
    DataCopyPadParams padParams {
        isPaddingBefore_,
        (uint8_t)0,
        (uint8_t)rightPaddingBefore_,
        (uint8_t)0
    };
    DataCopyPad(ubInput, gmInput_, copyParams, padParams);
    inputQueue_.EnQue(ubInput);
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32S<T>::ComputeBefore()
{
    LocalTensor<T> input = inputQueue_.DeQue<T>();
    LocalTensor<T> output = tmpOutQueue_.AllocTensor<T>();

    // output清零
    Duplicate(output, 0, (7 * (numOfExpert_ - 1) + numOfExpert_));
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    int32_t startIdx = 0;
    int32_t endIdx = startIdx + handleNum_ - 1;

    int32_t startTarget = input.GetValue(startIdx);
    int32_t endTarget = input.GetValue(endIdx);

    int32_t lastIdx = 0; // 最后一个可以找到的专家索引
    int32_t lastVal = 0; // 最后一个可以找到的专家号

    // main core
    if (GetBlockIdx() != usedCoreNumBefore_ - 1) {
        for (int32_t target = startTarget, idx = 0; target <= endTarget; target++, idx++) {
            int32_t low = startIdx;
            int32_t high = endIdx - startIdx;
            int32_t targetLocation = 0;
            int32_t mid = 0;
            while (low <= high) {
                mid = (low + high) / 2;
                if (input.GetValue(mid) > target) {
                    high = mid - 1;
                } else {
                    low = mid + 1;
                    targetLocation = mid;
                }
            }
            // 可以找到
            int32_t startOffset = handleNumPerCoreBefore_ * GetBlockIdx();
            if (input.GetValue(targetLocation) == target) {
                Duplicate(output[target * 8], startOffset + targetLocation + 1, 1);
                lastIdx = target;
                lastVal = startOffset + targetLocation + 1;
            } else {
                // target找不到，该位置数置为0
                Duplicate(output[target * 8], lastVal, 1);
            }
        }
        if ((lastIdx + 1) * 8 != output.GetSize()) {
            Duplicate(output[(lastIdx + 1) * 8], lastVal, (numOfExpert_ - (lastIdx + 1)) * 8);
        }
    } else {
        // tail core
        for (int32_t target = startTarget, idx = 0; target <= endTarget; target++, idx++) {
            int32_t low = startIdx;
            int32_t high = endIdx - startIdx;
            int32_t targetLocation = 0;
            int32_t mid = 0;
            while (low <= high) {
                mid = (low + high) / 2;
                if (input.GetValue(mid) > target) {
                    high = mid - 1;
                } else {
                    low = mid + 1;
                    targetLocation = mid;
                }
            }
            // 可以找到
            int32_t startOffset = handleNumPerCoreBefore_ * GetBlockIdx();
            if (input.GetValue(targetLocation) == target) {
                Duplicate(output[target * 8], startOffset + targetLocation + 1, 1);
                lastIdx = target;
                lastVal = startOffset + targetLocation + 1;
            } else {
                // target找不到，该位置数置为0
                Duplicate(output[target * 8], lastVal, 1);
            }
        }
        if ((lastIdx + 1) * 8 != output.GetSize()) {
            Duplicate(output[(lastIdx + 1) * 8], lastVal, (numOfExpert_ - (lastIdx + 1)) * 8);
        }
    }
    tmpOutQueue_.EnQue<T>(output);
    inputQueue_.FreeTensor(input);
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32S<T>::CopyOutBefore()
{
    LocalTensor<T> output = tmpOutQueue_.DeQue<T>();
    uint16_t blockCount = numOfExpert_;
    uint16_t blockLen = sizeof(T);
    uint16_t srcStride = 0;
    uint16_t dstStride = (usedCoreNumBefore_ - 1) * sizeof(T);
    DataCopyParams dataCopyParams {blockCount, blockLen, srcStride, dstStride};
    DataCopyPad(gmWorkspace_[GetBlockIdx()], output, dataCopyParams);
    tmpOutQueue_.FreeTensor(output);
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32S<T>::ProcessBefore()
{
    if (GetBlockIdx() >= usedCoreNumBefore_) {
        return;
    }

    CopyInBefore();
    ComputeBefore();
    CopyOutBefore();
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32S<T>::CopyInAfter(int64_t nLoopIdx, int64_t numOfLoop)
{
    LocalTensor<T> inputLocal = workspaceQueue_.AllocTensor<T>();
    if (usedCoreNumBefore_ * sizeof(T) % 32) {
        isPaddingAfter_ = true;
        rightPaddingAfter_ = PadProcessInt32(usedCoreNumBefore_);
    }

    DataCopyParams copyParams {
        (uint16_t)(numOfLoop),
        (uint16_t)(usedCoreNumBefore_ * sizeof(T)),
        (uint16_t)0,
        (uint16_t)0
    };
    DataCopyPadParams padParams {
        isPaddingAfter_,
        (uint8_t)0,
        (uint8_t)rightPaddingAfter_,
        (uint8_t)0
    };
    DataCopyPad(inputLocal, gmWorkspace_[nLoopIdx * curCoreHandleNumPerLoop_ * usedCoreNumBefore_], copyParams, padParams);
    workspaceQueue_.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32S<T>::ComputeAfter(int64_t nLoopIdx, int64_t numOfLoop)
{
    LocalTensor<T> inputLocal = workspaceQueue_.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();

    LocalTensor<float> inputCastTmpUb = inputCastTmpBuf_.Get<float>();
    LocalTensor<float> outputCastTmpUb = outputCastTmpBuf_.Get<float>();

    Cast(inputCastTmpUb, inputLocal, RoundMode::CAST_NONE, numOfLoop * Int32AlignmentProcess(usedCoreNumBefore_));
    uint64_t mask = usedCoreNumBefore_;
    int32_t repeatTimes = numOfLoop;
    int32_t dstRepStride = 1;
    int32_t srcBlkStride = 1;
    int32_t srcRepStride = Int32AlignmentProcess(usedCoreNumBefore_) * sizeof(float) / 32;
    pipe_barrier(PIPE_V);
    WholeReduceMax<float>(outputCastTmpUb, inputCastTmpUb, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride,
                          ReduceOrder::ORDER_ONLY_VALUE);
    pipe_barrier(PIPE_V);
    Cast(outputLocal, outputCastTmpUb, RoundMode::CAST_ROUND, Int32AlignmentProcess(numOfLoop));
    outputQueue_.EnQue(outputLocal);
    workspaceQueue_.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32S<T>::CopyOutAfter(int64_t nLoopIdx, int64_t numOfLoop)
{
    LocalTensor<T> outLocal = outputQueue_.DeQue<T>();
    DataCopyParams copyParamsOut{
        (uint16_t)1,
        (uint16_t)(numOfLoop * sizeof(T)),
        (uint16_t)0,
        (uint16_t)0
    };
    DataCopyPad(gmOutput_[nLoopIdx * curCoreHandleNumPerLoop_], outLocal, copyParamsOut);
    outputQueue_.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32S<T>::ProcessAfter()
{
    if (GetBlockIdx() >= usedCoreNumAfter_) {
        return;
    }

    int64_t inputIdx = GetBlockIdx() * normalCoreHandleNum_ * usedCoreNumBefore_;
    gmWorkspace_.SetGlobalBuffer((__gm__ T *)workspace_ + inputIdx, curCoreHandleNum_ * Int32AlignmentProcess(usedCoreNumBefore_));

    auto numOfLoop = curCoreHandleNumPerLoop_;
    for (int64_t n = 0; n < loopCount_; n++) {
        if (n == (loopCount_ - 1)) {
            numOfLoop = curCoreHandleNumTailLoop_;
        }
        CopyInAfter(n, numOfLoop);
        ComputeAfter(n, numOfLoop);
        CopyOutAfter(n, numOfLoop);
    }
}

template <typename T>
__aicore__ inline void MoeComputeExpertTokensInt32S<T>::Process()
{
    ProcessBefore();
    SyncAll();
    ProcessAfter();
}

}  // namespace Moe
#endif  // MOE_COMPUTE_EXPERT_TOKENS_INT32_S