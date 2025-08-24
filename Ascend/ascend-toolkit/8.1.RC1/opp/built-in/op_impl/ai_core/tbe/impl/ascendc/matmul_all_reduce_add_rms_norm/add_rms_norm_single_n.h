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
 * \file add_rms_norm_single_n.h
 * \brief
 */
#ifndef MC2_ADD_RMS_NORM_SINGLE_N_H
#define MC2_ADD_RMS_NORM_SINGLE_N_H
#include "rms_norm_base.h"

using namespace AscendC;

template <typename T>
class KernelAddRmsNormSingleN {
public:
    __aicore__ inline KernelAddRmsNormSingleN() {}
    __aicore__ inline void Init(GM_ADDR gammaGM, AddRMSNormTilingData &tiling, TPipe *pipe, uint32_t blockDim) {
        ASSERT(blockDim != 0 && "Block dim can not be zero!");
        this->numCol_ = tiling.num_col;
        this->ubFactor_ = tiling.ub_factor;
        this->epsilon_ = tiling.epsilon;
        this->avgFactor_ = (numCol_ != 0) ? (float)1.0 / numCol_ : 0;
        // get start index for current core, core parallel
        gamma_.SetGlobalBuffer((__gm__ T *)gammaGM, numCol_);
        pipe->InitBuffer(unitBuf_, 195584);  // (192 - 1) * 1024 byte
    }

    __aicore__ inline void Process() {
        if constexpr (IsSameType<T, half>::value) {
            ProcessFp16();
        } else {
            ProcessBf16();
        }
    }

    __aicore__ inline void ComputeProcess(GM_ADDR normOutGM, GM_ADDR residualGM, GM_ADDR yGM,
                                          AddRMSNormTilingData &tilingData, uint32_t addRmsNormCount, uint32_t rcvCnt)
    {
        uint64_t cOffset = CalcShapeOffset(sizeof(T), tilingData.num_row, tilingData.num_col); // 偏移*size
        for ( ; addRmsNormCount <= rcvCnt; ++addRmsNormCount) {
            normOut_.SetGlobalBuffer((__gm__ T *)normOutGM + GetBlockIdx() * numCol_, numCol_);
            residual_.SetGlobalBuffer((__gm__ T *)residualGM + GetBlockIdx() * numCol_, numCol_);
            y_.SetGlobalBuffer((__gm__ T *)yGM + GetBlockIdx() * numCol_, numCol_);
            Process();
            normOutGM += cOffset;
            residualGM += cOffset;
            yGM += cOffset;
        }
    }

private:
    __aicore__ inline void ProcessFp16() {
        LocalTensor<float> ubLocal = unitBuf_.Get<float>();
        LocalTensor<T> xLocal = ubLocal.template ReinterpretCast<T>();
        LocalTensor<T> x1Local = xLocal[0];
        LocalTensor<T> x2Local = xLocal[ubFactor_];
        LocalTensor<float> xFp32Local = ubLocal[ubFactor_];
        LocalTensor<float> sqxLocal = ubLocal[ubFactor_ * 2];
        LocalTensor<float> tmpLocal = ubLocal[ubFactor_ * 3];

        DataCopyCustom<T>(x1Local, normOut_, numCol_);
        event_t eventMTE2V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        set_flag(PIPE_MTE2, PIPE_V, eventMTE2V1);
        DataCopyCustom<T>(x2Local, residual_, numCol_);
        event_t eventMTE2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        set_flag(PIPE_MTE2, PIPE_V, eventMTE2V2);
        wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V1);
        wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V2);
        Add(x1Local, x1Local, x2Local, numCol_);
        pipe_barrier(PIPE_V);

        // copy gamma
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        set_flag(PIPE_V, PIPE_MTE2, eventVMTE2);
        wait_flag(PIPE_V, PIPE_MTE2, eventVMTE2);

        DataCopyCustom<T>(x2Local, gamma_, numCol_);  // gammaLocal use x2Local
        set_flag(PIPE_MTE2, PIPE_V, eventMTE2V2);

        // copy x out
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        set_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
        wait_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
        DataCopyCustom<T>(y_, x1Local, numCol_);
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        set_flag(PIPE_MTE3, PIPE_V, eventMTE3V);

        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol_);
        pipe_barrier(PIPE_V);
        Mul(sqxLocal, xFp32Local, xFp32Local, numCol_);
        pipe_barrier(PIPE_V);
        Muls(sqxLocal, sqxLocal, avgFactor_, numCol_);
        pipe_barrier(PIPE_V);
        ReduceSumCustom(sqxLocal, sqxLocal, tmpLocal, numCol_);
        pipe_barrier(PIPE_V);
        Adds(sqxLocal, sqxLocal, epsilon_, 1);
        pipe_barrier(PIPE_V);
        Sqrt(sqxLocal, sqxLocal, 1);
        Duplicate(tmpLocal, (float)1.0, 1);
        pipe_barrier(PIPE_V);
        Div(sqxLocal, tmpLocal, sqxLocal, 1);
        pipe_barrier(PIPE_V);

        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        set_flag(PIPE_V, PIPE_S, eventVS);
        wait_flag(PIPE_V, PIPE_S, eventVS);
        float rstdValue = sqxLocal.GetValue(0);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        set_flag(PIPE_S, PIPE_V, eventSV);
        wait_flag(PIPE_S, PIPE_V, eventSV);

        Muls(xFp32Local, xFp32Local, rstdValue, numCol_);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, eventMTE3V);
        Cast(x1Local, xFp32Local, RoundMode::CAST_NONE, numCol_);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V2);
        Mul(x1Local, x1Local, x2Local, numCol_);
        set_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
        wait_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
        DataCopyCustom<T>(normOut_, x1Local, numCol_);
    }

    __aicore__ inline void ProcessBf16(){
        LocalTensor<float> ubLocal = unitBuf_.Get<float>();
        LocalTensor<T> xLocal = ubLocal.template ReinterpretCast<T>();
        LocalTensor<T> x1Local = xLocal[0];
        LocalTensor<T> x2Local = xLocal[ubFactor_];
        LocalTensor<float> xFp32Local = ubLocal[ubFactor_];
        LocalTensor<float> sqxLocal = ubLocal[ubFactor_ * 2];
        LocalTensor<float> tmpLocal = ubLocal[ubFactor_ * 3];

        DataCopyCustom<T>(x1Local, normOut_, numCol_);
        event_t eventMTE2V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        set_flag(PIPE_MTE2, PIPE_V, eventMTE2V1);
        DataCopyCustom<T>(x2Local, residual_, numCol_);
        event_t eventMTE2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        set_flag(PIPE_MTE2, PIPE_V, eventMTE2V2);
        wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V1);
        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol_);
        wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V2);
        Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol_);
        pipe_barrier(PIPE_V);
        Add(xFp32Local, xFp32Local, sqxLocal, numCol_);
        pipe_barrier(PIPE_V);
        Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol_);
        pipe_barrier(PIPE_V);
        // copy gamma
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        set_flag(PIPE_V, PIPE_MTE2, eventVMTE2);
        wait_flag(PIPE_V, PIPE_MTE2, eventVMTE2);

        DataCopyCustom<T>(x2Local, gamma_, numCol_);  // gammaLocal use x2Local
        set_flag(PIPE_MTE2, PIPE_V, eventMTE2V2);

        // copy x out
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        set_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
        wait_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
        DataCopyCustom<T>(y_, x1Local, numCol_);
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        set_flag(PIPE_MTE3, PIPE_V, eventMTE3V);

        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol_);
        pipe_barrier(PIPE_V);
        Mul(sqxLocal, xFp32Local, xFp32Local, numCol_);
        pipe_barrier(PIPE_V);
        Muls(sqxLocal, sqxLocal, avgFactor_, numCol_);
        pipe_barrier(PIPE_V);
        ReduceSumCustom(sqxLocal, sqxLocal, tmpLocal, numCol_);
        pipe_barrier(PIPE_V);
        Adds(sqxLocal, sqxLocal, epsilon_, 1);
        pipe_barrier(PIPE_V);
        Sqrt(sqxLocal, sqxLocal, 1);
        Duplicate(tmpLocal, (float)1.0, 1);
        pipe_barrier(PIPE_V);
        Div(sqxLocal, tmpLocal, sqxLocal, 1);
        pipe_barrier(PIPE_V);

        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        set_flag(PIPE_V, PIPE_S, eventVS);
        wait_flag(PIPE_V, PIPE_S, eventVS);
        float rstdValue = sqxLocal.GetValue(0);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        set_flag(PIPE_S, PIPE_V, eventSV);
        wait_flag(PIPE_S, PIPE_V, eventSV);
        Muls(xFp32Local, xFp32Local, rstdValue, numCol_);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, eventMTE3V);

        Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol_);
        pipe_barrier(PIPE_V);
        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol_);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V2);

        Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol_);
        pipe_barrier(PIPE_V);
        Mul(xFp32Local, xFp32Local, sqxLocal, numCol_);
        pipe_barrier(PIPE_V);
        Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol_);
        set_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
        wait_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
        DataCopyCustom<T>(normOut_, x1Local, numCol_);
    }

private:
    TBuf<TPosition::VECCALC> unitBuf_;
    GlobalTensor<T> normOut_;
    GlobalTensor<T> residual_;
    GlobalTensor<T> gamma_;
    GlobalTensor<T> y_;

    uint32_t numCol_;
    uint32_t ubFactor_;
    float epsilon_;
    float avgFactor_;
};
#endif  // MC2_ADD_RMS_NORM_SINGLE_N_H