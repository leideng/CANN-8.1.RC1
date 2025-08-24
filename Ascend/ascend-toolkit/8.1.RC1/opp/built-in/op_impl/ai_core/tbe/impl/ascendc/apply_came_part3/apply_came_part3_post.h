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
 * \file apply_came_part3_post.h
 * \brief
 */
#ifndef ASCENDC_APPLY_CAME_PART3_POST_H_
#define ASCENDC_APPLY_CAME_PART3_POST_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apply_came_part3_common.h"

using namespace AscendC;

template <typename T>
class ApplyCamePart3Post {
public:
    __aicore__ inline ApplyCamePart3Post(){};
    __aicore__ inline void Init(CamePart3InOut camePart3InOut, GM_ADDR workspace,
                                const ApplyCamePart3TilingData* tiling_data);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const ApplyCamePart3TilingData* tiling_data);
    __aicore__ inline void Pre_Core_Compute(uint64_t gmOffsets, uint64_t cal_m);
    __aicore__ inline void ReduceAdd(LocalTensor<float> accuUb, int64_t n, int64_t m);
    __aicore__ inline void CalcSumC(LocalTensor<float> inputLocal, int64_t gmOffsets, int64_t idx,
                                    int64_t preN, int64_t calcN, int64_t calcM);
    __aicore__ inline void CalcSumURC();
    __aicore__ inline int64_t DivCeil(int64_t a, int64_t b);
    __aicore__ inline int64_t Ceil(int64_t a, int64_t b);
    __aicore__ inline void L0ReduceSum(LocalTensor<float> dst, LocalTensor<float> src,
                                       LocalTensor<float> worklocal, int64_t size);

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> inputBuf;

    GlobalTensor<float> gmSumGradR_;
    GlobalTensor<float> gmSumGradC_;
    GlobalTensor<float> gmSumGradRC_;

    GlobalTensor<float> workspaceSumGradRC_;
    GlobalTensor<float> workspaceSumGradC_;

    // multi-core sync
    GlobalTensor<int32_t> syncGlobal_;
    GM_ADDR workspaceAddr_;

    // tiling params
    int64_t usedCoreNum_{0};
    int64_t curN{0};
    int64_t curM{0};
    int64_t rNumCalc_{0};
    int64_t cNumCalc_{0};
    int64_t baseN{0};
    int64_t baseM{0};
    int64_t rCoreNum_{0};
    int64_t cCoreNum_{0};

    int64_t offset{0};
    bool isGlobalShape{false};
    bool useFirstMoment{false};

    // part1 params
    int64_t baseRCSize{0};
    int64_t baseCSize{0};
    int64_t mLoopNumCore_{0};

    const int64_t ONCE_HANDLE_NUM64{64};
    const int64_t ONCE_HANDLE_NUM512{512};
    const int64_t ONCE_ONE_SIZE8{8};
    const int64_t ONCE_ALGN_NUM{32 / sizeof(float)};
    int64_t MAX_BUF_SIZE{16384};
    int64_t MAX_BLOCK_LEN{65535};
    int64_t MAX_BLOCK_LEN_SIZE_FP32{16376};
    int64_t MAX_DATA_COPY_BLOCK_COUNT{4096};
    int64_t ONE_BLOCK_SIZE_FP32{8};

    constexpr static uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
};

template <typename T>
__aicore__ inline int64_t ApplyCamePart3Post<T>::Ceil(int64_t a, int64_t b) {
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

template <typename T>
__aicore__ inline int64_t ApplyCamePart3Post<T>::DivCeil(int64_t a, int64_t b) {
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::ParseTilingData(const ApplyCamePart3TilingData* tiling_data)
{
    // 总维度[curN, curM]
    curN = tiling_data->curN;
    curM = tiling_data->curM;
    rNumCalc_ = tiling_data->rNumCalc;
    cNumCalc_ = tiling_data->cNumCalc;
    baseN = tiling_data->baseN;
    baseM = tiling_data->baseM;
    rCoreNum_ = tiling_data->rCoreNum;
    cCoreNum_ = tiling_data->cCoreNum;

    // 使用核数 [usedCoreNum_]
    usedCoreNum_ = tiling_data->usedCoreNum;

    // 行列方向的核 [rCoreNum_, cCoreNum_]
    rCoreNum_ = tiling_data->rCoreNum;
    cCoreNum_ = tiling_data->cCoreNum;
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::Init(CamePart3InOut camePart3InOut, GM_ADDR workspace,
                                                const ApplyCamePart3TilingData* tiling_data)
{
    // 初始化tiling
    ParseTilingData(tiling_data);

    // gm输出
    gmSumGradR_.SetGlobalBuffer((__gm__ float *)(camePart3InOut.sumUR));
    gmSumGradC_.SetGlobalBuffer((__gm__ float *)(camePart3InOut.sumUC));
    gmSumGradRC_.SetGlobalBuffer((__gm__ float *)(camePart3InOut.sumURC));

    // workspace vars
    int64_t cTailNumCalc = curM - cNumCalc_ * (cCoreNum_ - 1);
    int64_t cBlockNum = DivCeil(cNumCalc_, baseM) * (cCoreNum_ - 1) + DivCeil(cTailNumCalc, baseM);
    int64_t rTailNumCalc = curN - rNumCalc_ * (rCoreNum_ - 1);
    int64_t rBlockNum = DivCeil(rNumCalc_, baseN) * (rCoreNum_ - 1) + DivCeil(rTailNumCalc, baseN);
    baseRCSize = cBlockNum * rBlockNum;
    baseCSize = rBlockNum;

    int64_t workspaceRCSize = DivCeil(cNumCalc_, baseM) * rCoreNum_ * DivCeil(rNumCalc_, baseN) * cCoreNum_;

    // workspace地址
    workspaceSumGradRC_.SetGlobalBuffer((__gm__ float*)workspace + DET_WORKSPACE_SIZE);
    workspaceSumGradC_.SetGlobalBuffer((__gm__ float*)workspace + workspaceRCSize + DET_WORKSPACE_SIZE);

    pipe.InitBuffer(inputBuf, ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64 * sizeof(float));
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::L0ReduceSum(LocalTensor<float> dst, LocalTensor<float> src,
                                                          LocalTensor<float> worklocal, int64_t size)
{
    if (size <= ONCE_HANDLE_NUM64) {
        ReduceSum(dst, src, worklocal, size);
    } else if (size % ONCE_HANDLE_NUM64) {
        int64_t repeat = size / ONCE_HANDLE_NUM64;
        int64_t tail = size % ONCE_HANDLE_NUM64;

        ReduceSum(dst, src, worklocal, ONCE_HANDLE_NUM64, repeat, 8);
        pipe_barrier(PIPE_V);
        ReduceSum(dst[1], src[repeat * ONCE_HANDLE_NUM64], worklocal, tail, 1, 8);
        pipe_barrier(PIPE_V);
        ReduceSum(dst, dst, worklocal, 2, 1, 8);
    } else {
        int64_t repeat = size / ONCE_HANDLE_NUM64;
        ReduceSum(dst, src, worklocal, ONCE_HANDLE_NUM64, repeat, 8);
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::Process()
{
    if (GetBlockIdx() != 0) {
        return;
    }

    CalcSumURC();


    mLoopNumCore_ = DivCeil(curM, ONCE_HANDLE_NUM64);
    uint64_t core_loop = DivCeil(mLoopNumCore_ , ONCE_HANDLE_NUM512 -1);
    uint64_t pre_core_m = mLoopNumCore_ / core_loop * ONCE_HANDLE_NUM64;
    uint64_t last_core_m = (mLoopNumCore_ - pre_core_m * (core_loop -1)) * ONCE_HANDLE_NUM64;
    uint64_t gmOffsets = 0;
    uint64_t base_m = 0;

    for (int64_t core_loop_idx = 0; core_loop_idx < core_loop - 1; core_loop_idx++) {
        gmOffsets = core_loop_idx * pre_core_m;
        Pre_Core_Compute(gmOffsets, pre_core_m);
    }
    gmOffsets = (core_loop - 1) * pre_core_m;
    base_m = curM - pre_core_m * (core_loop -1);
    Pre_Core_Compute(gmOffsets, base_m);
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::Pre_Core_Compute(uint64_t gmOffsets, uint64_t cal_m)
{
    uint64_t pre_loop_n = 1;
    while (pre_loop_n < baseCSize && pre_loop_n < MAX_DATA_COPY_BLOCK_COUNT) {
        pre_loop_n = pre_loop_n << 1;
        if (pre_loop_n * Ceil(cal_m, ONE_BLOCK_SIZE_FP32) > ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64 ||
            pre_loop_n >= MAX_DATA_COPY_BLOCK_COUNT) {
            pre_loop_n = pre_loop_n >> 1;
            break;
        }
        if (pre_loop_n >= baseCSize) {
            break;
        }
    }
    uint64_t loop_time = DivCeil(baseCSize, pre_loop_n);
    uint64_t last_loop_n = baseCSize - (loop_time -1) * pre_loop_n;
    LocalTensor<float> inputLocal = inputBuf.Get<float>(ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64);

    event_t eventMte3toS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventMte3toS);
    WaitFlag<HardEvent::MTE3_S>(eventMte3toS);

    constexpr float scalarValue = 0;
    Duplicate(inputLocal, scalarValue, ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64);

    event_t eventS2Mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventS2Mte2);
    WaitFlag<HardEvent::S_MTE2>(eventS2Mte2);

    for (int64_t i = 0; i < loop_time - 1; i++) {
        CalcSumC(inputLocal, gmOffsets, i, pre_loop_n, pre_loop_n, cal_m);

        event_t eventMte3toMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
    }

    if (loop_time > 1) {
        SetFlag<HardEvent::MTE3_S>(eventMte3toS);
        WaitFlag<HardEvent::MTE3_S>(eventMte3toS);

        Duplicate(inputLocal, scalarValue, pre_loop_n * Ceil(cal_m, FP32_ONE_BLOCK_COUNT));

        SetFlag<HardEvent::S_MTE2>(eventS2Mte2);
        WaitFlag<HardEvent::S_MTE2>(eventS2Mte2);
    }

    CalcSumC(inputLocal, gmOffsets, loop_time - 1, pre_loop_n, last_loop_n, cal_m);

    event_t eventMte3toMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::CalcSumC(LocalTensor<float> inputLocal, int64_t gmOffsets, int64_t idx,
                                                       int64_t preN, int64_t calcN, int64_t calcM)
{
    /*
    workspace -> UB -> reduceAdd -> GM
    */
    int64_t calcSize = calcM * sizeof(float);
    if (calcSize > MAX_BLOCK_LEN) {
        int64_t loop = calcM / MAX_BLOCK_LEN_SIZE_FP32;
        int64_t tail = calcM - loop * MAX_BLOCK_LEN_SIZE_FP32;

        for (int32_t i = 0; i < loop; i ++) {
            DataCopy(inputLocal[i * MAX_BLOCK_LEN_SIZE_FP32],
                     workspaceSumGradC_[gmOffsets + idx * preN * calcM + i * MAX_BLOCK_LEN_SIZE_FP32],
                     MAX_BLOCK_LEN_SIZE_FP32);
        }
        DataCopyPad(inputLocal[loop * MAX_BLOCK_LEN_SIZE_FP32],
                    workspaceSumGradC_[gmOffsets + idx * preN * calcM + loop * MAX_BLOCK_LEN_SIZE_FP32],
                    {(uint16_t)1, (uint16_t)(tail * sizeof(float)), 0, 0},
                    {false, 0, 0, 0});

        event_t eventMte2toMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventMte2toMte3);
        WaitFlag<HardEvent::MTE2_MTE3>(eventMte2toMte3);

        SetAtomicAdd<float>();
        for (int32_t i = 0; i < loop; i ++) {
            DataCopy(gmSumGradC_[i * MAX_BLOCK_LEN_SIZE_FP32],
                     inputLocal[i * MAX_BLOCK_LEN_SIZE_FP32],
                     MAX_BLOCK_LEN_SIZE_FP32);
        }
        DataCopyPad(gmSumGradC_[gmOffsets + loop * MAX_BLOCK_LEN_SIZE_FP32],
                    inputLocal[loop * MAX_BLOCK_LEN_SIZE_FP32],
                    {1, (uint16_t)(tail * sizeof(float)), 0, 0});
        SetAtomicNone();
    } else {
        uint8_t rightPadding = Ceil(calcM, ONE_BLOCK_SIZE_FP32) - calcM;
        DataCopyPad(inputLocal,
                    workspaceSumGradC_[gmOffsets + idx * preN * calcM],
                    {(uint16_t)calcN, (uint16_t)calcSize, 0, 0},
                    {true, 0, rightPadding, 0});

        event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventMte2V);

        ReduceAdd(inputLocal, preN, Ceil(calcM, ONE_BLOCK_SIZE_FP32));

        event_t eventV2Mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventV2Mte3);
        WaitFlag<HardEvent::V_MTE3>(eventV2Mte3);

        SetAtomicAdd<float>();
        DataCopyPad(gmSumGradC_[gmOffsets],
                    inputLocal,
                    {1, (uint16_t)calcSize, 0, 0});
        SetAtomicNone();
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::CalcSumURC()
{
    LocalTensor<float> inputLocal = inputBuf.Get<float>(MAX_BUF_SIZE * 2);
    LocalTensor<float> workLocal = inputLocal[MAX_BUF_SIZE];

    uint64_t loop_time = DivCeil(baseRCSize, MAX_BUF_SIZE);
    uint64_t pre_ele_num = DivCeil(baseRCSize, loop_time);
    uint64_t last_ele_num = baseRCSize - pre_ele_num * (loop_time - 1);

    // if rc workspace > maxub
    for (int64_t i = 0; i < loop_time - 1; i++) {
        DataCopyPad(inputLocal,
                    workspaceSumGradRC_[i * pre_ele_num],
                    {1, (uint16_t)(pre_ele_num * 4), 0, 0},
                    {false, 0, 0, 0});

        event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventMte2V);

        L0ReduceSum(inputLocal, inputLocal, workLocal, pre_ele_num);

        event_t eventV2Mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventV2Mte3);
        WaitFlag<HardEvent::V_MTE3>(eventV2Mte3);

        SetAtomicAdd<float>();
        DataCopyPad(gmSumGradRC_, inputLocal, {1, (uint16_t)(1 * 4), 0, 0});
        SetAtomicNone();

        event_t eventMte3toMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
    }

    uint8_t rightPadding = Ceil(last_ele_num, ONE_BLOCK_SIZE_FP32) - last_ele_num;
    DataCopyPad(inputLocal,
                workspaceSumGradRC_[(loop_time - 1) * pre_ele_num],
                {1, (uint16_t)(last_ele_num * 4), 0, 0},
                {true, 0, rightPadding, 0});

    event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMte2V);
    WaitFlag<HardEvent::MTE2_V>(eventMte2V);

    L0ReduceSum(inputLocal, inputLocal, workLocal, last_ele_num);

    event_t eventV2Mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventV2Mte3);
    WaitFlag<HardEvent::V_MTE3>(eventV2Mte3);

    SetAtomicAdd<float>();
    DataCopyPad(gmSumGradRC_, inputLocal, {1, (uint16_t)(1 * 4), 0, 0});
    SetAtomicNone();

    event_t eventMte3toMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::ReduceAdd(LocalTensor<float> accuUb, int64_t n, int64_t m)
{
    for (int32_t j = 1; j < n; j *= 2) {
        Add(
            accuUb[0],
            accuUb[n * m / 2 / j],
            accuUb[0],
            n * m / 2 / j
        );
        pipe_barrier(PIPE_V);
    }
}

#endif // _ASCENDC_APPLY_CAME_PART3_POST_H_