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
 * \file apply_came_part2_post.h
 * \brief
 */
#ifndef APPLY_CAME_PART2_POST
#define APPLY_CAME_PART2_POST

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apply_came_part2_common.h"

using namespace AscendC;

template <typename T>
class ApplyCamePart2Post {
public:
    __aicore__ inline ApplyCamePart2Post(){};
    __aicore__ inline void Init(GM_ADDR sum_square_u, GM_ADDR workspace,
                                const ApplyCamePart2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const ApplyCamePart2TilingData* tilingData);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> inputBuf_;
    TBuf<QuePosition::VECCALC> tmpBuf_;

    GlobalTensor<float> gmSumSquareU_;

    GlobalTensor<float> workspaceInnerSumSquareU_;

    // multi-core sync
    GlobalTensor<int32_t> syncGlobal_;
    GM_ADDR workspaceAddr_;

    // tiling params
    int64_t cLoop_{0};
    int64_t rLoop_{0};
    int64_t totalCoreNum_{0};
    int64_t usedCoreNum_{0};

    int64_t workspace_{0};

    const int64_t ONCE_HANDLE_NUM64{64};
    const int64_t ONCE_HANDLE_NUM512{512};
    const int64_t ONCE_ONE_SIZE8{8};
    const int64_t ONCE_ALGN_NUM{32 / sizeof(float)};
    const int64_t MAX_BOUND_VAL{65535};
};

template <typename T>
__aicore__ inline void ApplyCamePart2Post<T>::ParseTilingData(const ApplyCamePart2TilingData* tilingData)
{
    // 使用核数 && 总核数 [totalCoreNum_, usedCoreNum_]
    totalCoreNum_ = tilingData->totalCoreNum;
    cLoop_ = tilingData->cRcLoopCount;
    rLoop_ = tilingData->rRcLoopCount;
    usedCoreNum_ = tilingData->rRcCoreNumToUse;

    workspace_ = tilingData->workspaceSize;
}

template <typename T>
__aicore__ inline void ApplyCamePart2Post<T>::Init(GM_ADDR sum_square_u, GM_ADDR workspace, const ApplyCamePart2TilingData* tilingData)
{
    // 初始化tiling
    ParseTilingData(tilingData);

    // gmInput分核 && 输入偏移初始化
    gmSumSquareU_.SetGlobalBuffer((__gm__ float *)sum_square_u);

    // workspace地址
    workspaceInnerSumSquareU_.SetGlobalBuffer((__gm__ float*)workspace + WORKSPACE_ALIGNED_SIZE / FLOAT_SIZE);

    // buffer申请初始化
    pipe.InitBuffer(inputBuf_, ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64 * sizeof(float));
    pipe.InitBuffer(tmpBuf_, ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64 * sizeof(float));

    if (GetBlockIdx() == 0) {
        InitOutput<float>(gmSumSquareU_, 1, (float)0);
    }
    SyncAll();
}

template <typename T>
__aicore__ inline void ApplyCamePart2Post<T>::Process()
{
    if (GetBlockIdx() != 0) {
        return;
    }
    LocalTensor<float> mComTmpUb = tmpBuf_.Get<float>(ONCE_HANDLE_NUM64 * ONCE_HANDLE_NUM64);
    LocalTensor<float> inputLocal = inputBuf_.Get<float>(ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64);
    uint64_t ele_num = (cLoop_ + 1) * (rLoop_ + 1) * usedCoreNum_;
    uint64_t loop_time = (ele_num + 128 * ONCE_HANDLE_NUM64 - 1) / (128 * ONCE_HANDLE_NUM64);
    uint64_t pre_ele_num = (ele_num + loop_time - 1) / loop_time;
    uint64_t last_ele_num = ele_num - pre_ele_num * (loop_time - 1);

    if (loop_time == 1) {
        DataCopyPad(inputLocal, workspaceInnerSumSquareU_, {1, (uint16_t)(last_ele_num * FLOAT_SIZE), 0, 0}, {false, 0, 0, 0});

        event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventMte2V);
        ReduceSum(inputLocal, inputLocal, mComTmpUb, last_ele_num);

        event_t eventV2Mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventV2Mte3);
        WaitFlag<HardEvent::V_MTE3>(eventV2Mte3);
        DataCopyPad(gmSumSquareU_, inputLocal, {1, (uint16_t)(FLOAT_SIZE), 0, 0});
    } else {
        for (int64_t i = 0; i < loop_time - 1; i++) {
            DataCopyPad(inputLocal, workspaceInnerSumSquareU_[i * pre_ele_num], {1, (uint16_t)(pre_ele_num * FLOAT_SIZE), 0, 0}, {false, 0, 0, 0});

            event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMte2V);
            WaitFlag<HardEvent::MTE2_V>(eventMte2V);
            ReduceSum(inputLocal, inputLocal, mComTmpUb, pre_ele_num);

            event_t eventV2Mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventV2Mte3);
            WaitFlag<HardEvent::V_MTE3>(eventV2Mte3);
            SetAtomicAdd<float>();
            DataCopyPad(gmSumSquareU_, inputLocal, {1, (uint16_t)(FLOAT_SIZE), 0, 0});
            SetAtomicNone();

            event_t eventMte3toMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
        }

        DataCopyPad(inputLocal, workspaceInnerSumSquareU_[(loop_time - 1) * pre_ele_num], {1, (uint16_t)(last_ele_num * FLOAT_SIZE), 0, 0}, {false, 0, 0, 0});

        event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventMte2V);
        ReduceSum(inputLocal, inputLocal, mComTmpUb, last_ele_num);

        event_t eventV2Mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventV2Mte3);
        WaitFlag<HardEvent::V_MTE3>(eventV2Mte3);
        SetAtomicAdd<float>();
        DataCopyPad(gmSumSquareU_, inputLocal, {1, (uint16_t)(FLOAT_SIZE), 0, 0});
        SetAtomicNone();
    }
}

#endif  // APPLY_CAME_PART2_POST
