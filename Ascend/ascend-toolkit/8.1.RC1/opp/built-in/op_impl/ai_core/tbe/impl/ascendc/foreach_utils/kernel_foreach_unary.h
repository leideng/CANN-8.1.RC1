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
 * \file kernel_foreach_unary.h
 * \brief
 */

#ifndef KERNEL_FOREACH_UNARY_H
#define KERNEL_FOREACH_UNARY_H

#include "kernel_foreach_base.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint8_t INPUT_PARAMETER_COUNT = 2;
constexpr bool NEED_COPY_OUT = true;

template <typename T, typename Predicate, int32_t bufferNum=BUFFER_NUM, uint8_t paramsCount=INPUT_PARAMETER_COUNT, bool needCopyOut=NEED_COPY_OUT>
class KernelForeachUnary : public KernelForeachBase<T> {
protected:
    using Base = KernelForeachBase<T>;

    explicit __aicore__ inline KernelForeachUnary(Predicate &p): Base(), pred(p) {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                const ForeachCommonTilingData* tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void SingleTensorProcess(int64_t dataCount, LocalTensor<float> &float32Tensor);
private:
    __aicore__ inline void CopyIn(uint32_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, LocalTensor<float> &float32Tensor, bool isRemainder);
    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void BeforeProcess();
    __aicore__ inline void AfterProcess();
    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart);

protected:
    TQue<QuePosition::VECIN, bufferNum> dataQueue;
    TQue<QuePosition::VECOUT, bufferNum> outQueue;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;

    GM_ADDR inTensorsPtr = nullptr;
    GM_ADDR outTensorsPtr = nullptr;

    TQue<QuePosition::VECIN, 1> float32Queue;

private:
    Predicate &pred;
};

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachUnary<T, Predicate, bufferNum, paramsCount, needCopyOut>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
    const ForeachCommonTilingData* tilingData) {
    Base::Init(tilingData);

    inTensorsPtr = x;
    outTensorsPtr = y;

    #if __CCE_AICORE__ == 220
        if (std::is_same_v<T, bfloat16_t>) {
            Base::pipe.InitBuffer(dataQueue, bufferNum, Base::totalTensorUbSize);
            if (needCopyOut) {
                Base::pipe.InitBuffer(outQueue, bufferNum, Base::totalTensorUbSize);
            }
            Base::pipe.InitBuffer(float32Queue, 1, Base::inputsTensorUbSize * paramsCount);
            LocalTensor<float> float32Tensor = float32Queue.AllocTensor<float>();
            float32Queue.EnQue(float32Tensor);
        } else {
            Base::pipe.InitBuffer(dataQueue, bufferNum, Base::inputsTensorUbSize);
            if (needCopyOut) {
                Base::pipe.InitBuffer(outQueue, bufferNum, Base::inputsTensorUbSize);
            }
        }
    #else 
        Base::pipe.InitBuffer(dataQueue, bufferNum, Base::inputsTensorUbSize);
        if (needCopyOut) {
            Base::pipe.InitBuffer(outQueue, bufferNum, Base::inputsTensorUbSize);
        }
    #endif
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachUnary<T, Predicate, bufferNum, paramsCount, needCopyOut>::Process() {
    /*将中间量预留出来*/
    LocalTensor<float> float32Tensor;
    #if __CCE_AICORE__ == 220
        if (std::is_same_v<T, bfloat16_t>) {
            float32Tensor = float32Queue.DeQue<float>(); 
        }
    #endif

    BeforeProcess();

    for (uint16_t i = Base::tensorStart; i <= Base::tensorEnd; i++) {
        int64_t cursorStart = 0;
        int64_t cursorEnd = Base::tensorDataCountList[i] - 1;
        int64_t dataCount = 0;
        if (i == Base::tensorStart) {
            cursorStart = Base::tensorStartOffset;
        }
        if (i == Base::tensorEnd) {
            cursorEnd = Base::tensorEndOffset;
        }

        dataCount = cursorEnd - cursorStart + 1;

        inTensorsGM.SetGlobalBuffer(Base::GetTensorAddr(i, inTensorsPtr) + cursorStart);
        outTensorsGM.SetGlobalBuffer(Base::GetTensorAddr(i, outTensorsPtr) + cursorStart);
        ProcessPlusInLoop(i, cursorStart);
        SingleTensorProcess(dataCount, float32Tensor);
    }

    AfterProcess();

    #if __CCE_AICORE__ == 220
        if (std::is_same_v<T, bfloat16_t>) {
            float32Queue.FreeTensor(float32Tensor);
        }
    #endif
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachUnary<T, Predicate, bufferNum, paramsCount, needCopyOut>::SingleTensorProcess(
    int64_t dataCount, LocalTensor<float> &float32Tensor) {
    // Batch handling and calculation.
    uint32_t copyTimes = dataCount / Base::maxDataCount;
    uint32_t copyTimesRemainder = dataCount % Base::maxDataCount;
    uint32_t tempDataCount = Base::maxDataCount;

    if (copyTimesRemainder > 0) {
        copyTimes++;
    }
 
    for (uint32_t i = 0; i < copyTimes; i++) {
        bool isRemainder = false;
        if (i == copyTimes - 1 && copyTimesRemainder > 0) {
            isRemainder = true;
            tempDataCount = copyTimesRemainder;
        }
        CopyIn(i, tempDataCount, isRemainder);
        CopyInPlus(i, tempDataCount, isRemainder);
        Compute(i, tempDataCount, float32Tensor, isRemainder);
        if (needCopyOut) {
            CopyOut(i, tempDataCount, isRemainder);
        }
    }
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachUnary<T, Predicate, bufferNum, paramsCount, needCopyOut>::CopyIn(uint32_t index, int64_t dataCount, bool isRemainder) {
    LocalTensor<T> dataLocal = dataQueue.template AllocTensor<T>();
    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; 
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(dataLocal, inTensorsGM[index * Base::maxDataCount], copyParams, padParams);
    } else {
        DataCopy(dataLocal, inTensorsGM[index * Base::maxDataCount], dataCount);
    }
    dataQueue.EnQue(dataLocal);
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline bool KernelForeachUnary<T, Predicate, bufferNum, paramsCount, needCopyOut>::CopyOut(uint32_t index, int64_t dataCount, bool isRemainder) {
    static_assert(std::is_member_function_pointer_v<decltype(&Predicate::CopyOut)>);
    if (!pred.CopyOut(index, dataCount, isRemainder)) {
        LocalTensor<T> outLocal = outQueue.template DeQue<T>();
	    if (isRemainder) {
	        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0}; 
	        DataCopyPad(outTensorsGM[index * Base::maxDataCount], outLocal, copyParams);
	    } else {
	        DataCopy(outTensorsGM[index * Base::maxDataCount], outLocal, dataCount);
	    }

        outQueue.FreeTensor(outLocal);
    }
    return true;
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachUnary<T, Predicate, bufferNum, paramsCount, needCopyOut>::Compute(
    uint32_t index, int64_t dataCount, LocalTensor<float> &float32Tensor, bool isRemainder) {
    static_assert(std::is_member_function_pointer_v<decltype(&Predicate::Compute)>);
    pred.Compute(index, dataCount, float32Tensor, isRemainder);
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachUnary<T, Predicate, bufferNum, paramsCount, needCopyOut>::CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder) {
    if (std::is_member_function_pointer_v<decltype(&Predicate::CopyInPlus)>) {
        pred.CopyInPlus(index, dataCount, isRemainder);
    }
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachUnary<T, Predicate, bufferNum, paramsCount, needCopyOut>::BeforeProcess(){
    if (std::is_member_function_pointer_v<decltype(&Predicate::BeforeProcess)>) {
        pred.BeforeProcess();
    }
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachUnary<T, Predicate, bufferNum, paramsCount, needCopyOut>::AfterProcess(){
    if (std::is_member_function_pointer_v<decltype(&Predicate::AfterProcess)>) {
        pred.AfterProcess();
    }
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachUnary<T, Predicate, bufferNum, paramsCount, needCopyOut>::ProcessPlusInLoop(uint32_t index, uint64_t cursorStart){
    if (std::is_member_function_pointer_v<decltype(&Predicate::ProcessPlusInLoop)>) {
        pred.ProcessPlusInLoop(index, cursorStart);
    }
}

}  // namespace OpKernel
}  // namespace Common

#endif  // KERNEL_FOREACH_UNARY_H