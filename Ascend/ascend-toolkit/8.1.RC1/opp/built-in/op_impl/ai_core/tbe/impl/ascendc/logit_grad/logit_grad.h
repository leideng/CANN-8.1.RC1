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
 * \file logit_grad.h
 * \brief logit_grad head file
 */
#ifndef LOGIT_GRAD_H
#define LOGIT_GRAD_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace LogitGrad {

using namespace AscendC;

constexpr int32_t MAX_UB_SIZE = 192 * 1024;
constexpr int32_t PP_ELEMENT_NUM = 8 * 1024;
constexpr int32_t ONE_REPEAT_ELE_NUM_FP32 = 64;
constexpr int32_t ALIGN = 16;

template <typename T>
class LogitGradND {
public:
    TPipe pipe;
    __aicore__ inline LogitGradND(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dy, GM_ADDR dx, GM_ADDR workspace,
                                const LogitGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInAndCast(int64_t inputOffset, int64_t dataCount);
    __aicore__ inline void ComputeStepOne(int64_t dataCount);
    __aicore__ inline void ComputeStepTwo(int64_t dataCount);
    __aicore__ inline void CastAndCopyOut(int64_t outputOffset, int64_t dataCount);

private:

    TBuf<QuePosition::VECCALC> ubTBuf;
    LocalTensor<uint8_t> tmpTensor;

    LocalTensor<uint8_t> selMaskOne;
    LocalTensor<uint8_t> selMaskTwo;

    LocalTensor<T> x1Tmp;
    LocalTensor<T> x2Tmp;

    LocalTensor<T> x1Tensor;
    LocalTensor<T> x2Tensor;

    LocalTensor<float> x1TensorFp32;
    LocalTensor<float> x2TensorFp32;
    LocalTensor<float> tmpTensorFp32;

    GlobalTensor<T> inputGm;
    GlobalTensor<T> dyGm;
    GlobalTensor<T> outputGm;

    int32_t elementNum;
    uint32_t needCoreNumber;
    int32_t blockIdx;
    float eps;

    // 准备compare参数
    float epslion;
    float selectValue;

    event_t eventId = EVENT_ID0;
    int32_t pingPongFlag = 0;
};

template <typename T>
__aicore__ inline void LogitGradND<T>::Init(GM_ADDR x, GM_ADDR dy, GM_ADDR dx, GM_ADDR workspace,
                                              const LogitGradTilingData* tilingData) {
    inputGm.SetGlobalBuffer((__gm__ T*)x);
    dyGm.SetGlobalBuffer((__gm__ T*)dy);
    outputGm.SetGlobalBuffer((__gm__ T*)dx);

    eps = tilingData->eps;

    if (eps >= 0) {
        epslion = eps;
        selectValue = 0.0;
    } else {
        epslion = 0.0;
        selectValue = sqrt(static_cast<float>(-1.0));
    }
    elementNum = tilingData->elementNum;
    needCoreNumber = tilingData->needCoreNum;

    blockIdx = GetBlockIdx();
    pipe.InitBuffer(ubTBuf, MAX_UB_SIZE);
    tmpTensor = ubTBuf.Get<uint8_t>();
}

template <typename T>
__aicore__ inline void LogitGradND<T>::Process() {
    if (blockIdx >= needCoreNumber) {
        return;
    }
    
    int32_t totalTimes = elementNum / PP_ELEMENT_NUM;
    int32_t remain = elementNum % PP_ELEMENT_NUM;
    if (remain > 0) {
        totalTimes++;
    }
    int32_t loopNum = totalTimes / needCoreNumber;
    int32_t loopRemain = totalTimes % needCoreNumber;
    
    if (loopRemain > 0 && blockIdx < loopRemain) {
        loopNum++;
    }
    int32_t eachCoreStartOffset = loopNum * blockIdx * PP_ELEMENT_NUM;
    if (loopRemain > 0) {
        if (blockIdx >= loopRemain) {
            eachCoreStartOffset += elementNum % (PP_ELEMENT_NUM * needCoreNumber);
        }
    }
    
    int32_t calNum = PP_ELEMENT_NUM;

    int32_t lastCoreNum = loopRemain == 0 ? needCoreNumber - 1 : loopRemain - 1;
    pingPongFlag = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t i = 0; i < loopNum; i++) {
        int32_t localOffset = i * PP_ELEMENT_NUM;

        // 最后一轮的最后一个核处理余数
        if (remain > 0 && i == loopNum -1 && blockIdx == lastCoreNum) {
            calNum = remain;
        }
        eventId = pingPongFlag ? EVENT_ID1 : EVENT_ID0;
        CopyInAndCast(eachCoreStartOffset + localOffset, calNum);
        

        ComputeStepOne(calNum);

        ComputeStepTwo(calNum);

        CastAndCopyOut(eachCoreStartOffset + localOffset, calNum);

        pingPongFlag = 1 - pingPongFlag;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename T>
__aicore__ inline void LogitGradND<T>::CopyInAndCast(int64_t inputOffset, int64_t dataCount) {

    x1Tensor = pingPongFlag ?
            tmpTensor[MAX_UB_SIZE / 2].ReinterpretCast<T>() :
            tmpTensor[0].ReinterpretCast<T>();
    x2Tensor = pingPongFlag ?
            tmpTensor[PP_ELEMENT_NUM * sizeof(float) + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
            tmpTensor[PP_ELEMENT_NUM * sizeof(float)].ReinterpretCast<T>();
    WaitFlag<HardEvent::MTE3_MTE2>(eventId);

    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        int32_t elementByte = PP_ELEMENT_NUM * sizeof(T);
        x1Tmp = pingPongFlag ?
                tmpTensor[elementByte + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                tmpTensor[elementByte].ReinterpretCast<T>();
        x2Tmp = pingPongFlag ?
                tmpTensor[elementByte + PP_ELEMENT_NUM * sizeof(float) + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                tmpTensor[elementByte + PP_ELEMENT_NUM * sizeof(float)].ReinterpretCast<T>();
        DataCopyPad(x1Tmp, inputGm[inputOffset], dataCopyParams, padParams);
        DataCopyPad(x2Tmp, dyGm[inputOffset], dataCopyParams, padParams);

    } else {
        DataCopyPad(x1Tensor, inputGm[inputOffset], dataCopyParams, padParams);
        DataCopyPad(x2Tensor, dyGm[inputOffset], dataCopyParams, padParams);
    }

    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);

    x1TensorFp32 = x1Tensor.template ReinterpretCast<float>();
    x2TensorFp32 = x2Tensor.template ReinterpretCast<float>();
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        Cast(x1TensorFp32, x1Tmp, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(x2TensorFp32, x2Tmp, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void LogitGradND<T>::ComputeStepOne(int64_t dataCount) {
    int32_t elementByte = PP_ELEMENT_NUM * sizeof(float);
    selMaskOne = pingPongFlag ?
              tmpTensor[elementByte * 2 + MAX_UB_SIZE / 2] :
              tmpTensor[elementByte * 2];
    selMaskTwo = pingPongFlag ?
              tmpTensor[elementByte * 2 + elementByte / 2 + MAX_UB_SIZE / 2] :
              tmpTensor[elementByte * 2 + elementByte / 2];
    tmpTensorFp32 = selMaskOne.template ReinterpretCast<float>();

    Muls(tmpTensorFp32, x1TensorFp32, float(-1.0), dataCount);
    PipeBarrier<PIPE_V>();
    Adds(tmpTensorFp32, tmpTensorFp32, float(1.0), dataCount);
    PipeBarrier<PIPE_V>();
    Mul(tmpTensorFp32, x1TensorFp32, tmpTensorFp32, dataCount);
    PipeBarrier<PIPE_V>();
    Div(x2TensorFp32, x2TensorFp32, tmpTensorFp32, dataCount);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void LogitGradND<T>::ComputeStepTwo(int64_t dataCount) {
    float lo = epslion;
    float hi = static_cast<float>(1.0) - epslion;

    auto tmpDataCount = (dataCount + ONE_REPEAT_ELE_NUM_FP32 - 1) / ONE_REPEAT_ELE_NUM_FP32 * ONE_REPEAT_ELE_NUM_FP32;
    CompareScalar(selMaskOne, x1TensorFp32, (float)lo, CMPMODE::GE, tmpDataCount);
    PipeBarrier<PIPE_V>();

    CompareScalar(selMaskTwo, x1TensorFp32, (float)hi, CMPMODE::LE, tmpDataCount);
    PipeBarrier<PIPE_V>();

    auto tmpMaskOne = selMaskOne.template ReinterpretCast<uint16_t>();
    auto tmpMaskTwo = selMaskTwo.template ReinterpretCast<uint16_t>();
    And(tmpMaskOne, tmpMaskOne, tmpMaskTwo, tmpDataCount / ALIGN);
    PipeBarrier<PIPE_V>();

    Select(x1TensorFp32, selMaskOne, x2TensorFp32, (float)selectValue, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void LogitGradND<T>::CastAndCopyOut(int64_t outputOffset, int64_t dataCount) {
    if (std::is_same_v<T, half>) {
        Cast(x1Tensor, x1TensorFp32, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    } else if (std::is_same_v<T, bfloat16_t>) {
        Cast(x1Tensor, x1TensorFp32, RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGm[outputOffset], x1Tensor, dataCopyParams);
    SetFlag<HardEvent::MTE3_MTE2>(eventId);
}

}
#endif