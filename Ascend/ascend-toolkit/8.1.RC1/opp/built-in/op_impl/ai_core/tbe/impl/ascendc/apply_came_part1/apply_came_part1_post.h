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
 * \file apply_came_part1_post.h
 * \brief
 */
#ifndef APPLY_CAME_PART1_POST
#define APPLY_CAME_PART1_POST

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace ApplyCamePart1 {

using namespace AscendC;

template <typename T>
class ApplyCamePart1Post {
public:
    __aicore__ inline ApplyCamePart1Post(){};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR eps, GM_ADDR sum_grad_r, GM_ADDR sum_grad_c,
                                GM_ADDR sum_grad_rc, GM_ADDR workspace,
                                const ApplyCamePart1TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const ApplyCamePart1TilingData* tilingData);
    __aicore__ inline void Pre_Core_Compute(uint64_t gmOffsets, uint64_t cal_m, uint64_t base_m);
    __aicore__ inline void ReduceAdd(LocalTensor<float> accuUb, int64_t n, int64_t m);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> inputBuf_;
    TBuf<QuePosition::VECCALC> tmpBuf_;

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
    const int64_t ONCE_HANDLE_NUM512{512};
    const int64_t ONCE_ONE_SIZE8{8};
    const int64_t ONCE_ALGN_NUM{32 / sizeof(float)};
    const int64_t MAX_BOUND_VAL{65535};

    constexpr static uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
};

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ParseTilingData(const ApplyCamePart1TilingData* tilingData)
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
__aicore__ inline void ApplyCamePart1Post<T>::Init(GM_ADDR grad, GM_ADDR eps, GM_ADDR sum_grad_r, GM_ADDR sum_grad_c, GM_ADDR sum_grad_rc, GM_ADDR workspace, const ApplyCamePart1TilingData* tilingData)
{
    // 初始化tiling
    ParseTilingData(tilingData);

    // gmInput分核 && 输入偏移初始化
    gmSumGradC_.SetGlobalBuffer((__gm__ float *)sum_grad_c);
    gmSumGradRC_.SetGlobalBuffer((__gm__ float *)sum_grad_rc);

    // workspace地址
    int32_t syncOffsets = SYNC_GLOBAL_WORKSPACE_SIZE / sizeof(int32_t);
    int32_t workspaceOffsets = 0;
    workspaceSumGradRC_.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets);
    int32_t rcOffsets = (((usedCoreNum_ - 1) *  nLoopNormCore_ + nLoopTailCore_) * mLoopNumCore_ + 128 -1) / 128 * 128;
    workspaceOffsets = workspaceOffsets + rcOffsets;
    workspaceSumGradC_.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets);

    // buffer申请初始化
    pipe.InitBuffer(inputBuf_, ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64 * sizeof(float));
    pipe.InitBuffer(tmpBuf_, ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64 * sizeof(float));
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::Process()
{
    pipe_barrier(PIPE_ALL);
    if (GetBlockIdx() == 0) {
        uint64_t core_loop = (mLoopNumCore_ + ONCE_HANDLE_NUM512 -1) / ONCE_HANDLE_NUM512;
        uint64_t pre_core_m = (mLoopNumCore_+ core_loop -1) / core_loop * ONCE_HANDLE_NUM64;
        uint64_t last_core_m = (mLoopNumCore_ - pre_core_m * (core_loop -1) / ONCE_HANDLE_NUM64) * ONCE_HANDLE_NUM64;
        uint64_t gmOffsets = 0;
        uint64_t base_m = 0;

        for (int64_t core_loop_idx = 0; core_loop_idx < core_loop - 1; core_loop_idx++) {
            gmOffsets = core_loop_idx * pre_core_m;
            Pre_Core_Compute(gmOffsets, pre_core_m, pre_core_m);
        }
        gmOffsets = (core_loop -1) * pre_core_m;
        base_m = M - pre_core_m * (core_loop -1);
        Pre_Core_Compute(gmOffsets, last_core_m, base_m);
    }
    pipe_barrier(PIPE_ALL);
    if (GetBlockIdx() == 0) {
        LocalTensor<float> mComTmpUb = tmpBuf_.Get<float>(ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64);
        LocalTensor<float> inputLocal = inputBuf_.Get<float>(ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64);
        uint64_t ele_num = ((usedCoreNum_ - 1) *  nLoopNormCore_ + nLoopTailCore_) * mLoopNumCore_;
        uint64_t loop_time = (ele_num + 128 * ONCE_HANDLE_NUM64 - 1) / (128 * ONCE_HANDLE_NUM64);
        uint64_t pre_ele_num = (ele_num + loop_time - 1) / loop_time;
        uint64_t last_ele_num = ele_num - pre_ele_num * (loop_time - 1);

        for (int64_t i = 0; i < loop_time - 1; i++) {
            DataCopyPad(inputLocal, workspaceSumGradRC_[i * pre_ele_num], {1, (uint16_t)(pre_ele_num * 4), 0, 0}, {false, 0, 0, 0});
            pipe_barrier(PIPE_ALL);
            ReduceSum(inputLocal, inputLocal, mComTmpUb, pre_ele_num);
            pipe_barrier(PIPE_ALL);
            SetAtomicAdd<float>();
            DataCopyPad(gmSumGradRC_, inputLocal, {1, (uint16_t)(1 * 4), 0, 0});
            SetAtomicNone();
            pipe_barrier(PIPE_ALL);
        }
        pipe_barrier(PIPE_ALL);
        DataCopyPad(inputLocal, workspaceSumGradRC_[(loop_time - 1) * pre_ele_num], {1, (uint16_t)(last_ele_num * 4), 0, 0}, {false, 0, 0, 0});
        pipe_barrier(PIPE_ALL);
        ReduceSum(inputLocal, inputLocal, mComTmpUb, last_ele_num);
        pipe_barrier(PIPE_ALL);
        SetAtomicAdd<float>();
        DataCopyPad(gmSumGradRC_, inputLocal, {1, (uint16_t)(1 * 4), 0, 0});
        SetAtomicNone();
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::Pre_Core_Compute(uint64_t gmOffsets, uint64_t cal_m, uint64_t base_m)
{
    uint64_t total_n = (usedCoreNum_ - 1) * nLoopNormCore_ + nLoopTailCore_;
    uint64_t pre_loop_n = 1;
    while (pre_loop_n < total_n) {
        pre_loop_n = pre_loop_n << 1;
        if (pre_loop_n * cal_m > ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64) {
            pre_loop_n = pre_loop_n >> 1;
            break;
        }
        if (pre_loop_n >= total_n)  {
            break;
        }
    }
    uint64_t loop_time = (total_n + pre_loop_n -1) / pre_loop_n;
    uint64_t last_loop_n = total_n - (loop_time -1) * pre_loop_n;
    LocalTensor<float> inputLocal = inputBuf_.Get<float>(ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64);
    pipe_barrier(PIPE_ALL);

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = base_m * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    for (int64_t i = 0; i < loop_time - 1; i++) {
        DataCopy(inputLocal, workspaceSumGradC_[gmOffsets + i * pre_loop_n * mLoopNumCore_ * ONCE_HANDLE_NUM64], pre_loop_n * cal_m);
        pipe_barrier(PIPE_ALL);
        ReduceAdd(inputLocal, pre_loop_n, cal_m);
        SetAtomicAdd<float>();
        DataCopyPad(gmSumGradC_[gmOffsets], inputLocal, copyParams);
        SetAtomicNone();
        pipe_barrier(PIPE_ALL);
    }

    constexpr float scalarValue = 0;
    Duplicate(inputLocal, scalarValue, pre_loop_n * cal_m);
    pipe_barrier(PIPE_ALL);
    DataCopy(inputLocal, workspaceSumGradC_[gmOffsets + (loop_time - 1) * pre_loop_n * mLoopNumCore_ * ONCE_HANDLE_NUM64], last_loop_n * cal_m);
    pipe_barrier(PIPE_ALL);
    ReduceAdd(inputLocal, pre_loop_n, cal_m);
    SetAtomicAdd<float>();
    DataCopyPad(gmSumGradC_[gmOffsets], inputLocal, copyParams);
    SetAtomicNone();
}

template <typename T>
__aicore__ inline void ApplyCamePart1Post<T>::ReduceAdd(LocalTensor<float> accuUb, int64_t n, int64_t m)
{
    for (int32_t j = 1; j < n; j *= 2) {
        Add(
            accuUb[0],
            accuUb[n * m / 2 / j],
            accuUb[0],
            n * m / 2 / j
        );
        pipe_barrier(PIPE_ALL);
    }
}

}  // namespace ApplyCamePart1
#endif  // APPLY_CAME_PART1_POST
