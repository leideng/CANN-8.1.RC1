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
 * \file avg_pool3_d_grad_no_cast.h
 * \brief
 */
#ifndef AVG_POOL3D_GRAD_NO_CAST_H
#define AVG_POOL3D_GRAD_NO_CAST_H

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "avg_pool3_d_grad_base.h"

namespace AvgPool3DGrad {
using namespace AscendC;

template <typename T>
class KernelAvgPool3DGradNoCast : public KernelAvgPool3DGradBase<T> {
public:
    __aicore__ inline KernelAvgPool3DGradNoCast() {}
    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR output, const AvgPool3dGradTilingParam &tilingData, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ClearOutput();
    __aicore__ inline void ProcessPerLine(int64_t index, int64_t ubIndex);
    __aicore__ inline void Compute(int64_t ubIndex);
    __aicore__ inline void CopyOut(int64_t n, int64_t d, int64_t h, int64_t w, int64_t c, int64_t ubIndex);
};

template <typename T>
__aicore__ inline void KernelAvgPool3DGradNoCast<T>::Init(GM_ADDR grads, GM_ADDR output, 
                                                          const AvgPool3dGradTilingParam &tilingData, 
                                                          GM_ADDR workspace) 
{
    // Get all required params
    this->thisCoreIdx = GetBlockIdx();
    this->coreNum = GetBlockNum();
    this->ParseTilingData(tilingData);
    this->GetEventIds();

    // Init
    this->xGm.SetGlobalBuffer((__gm__ T*)grads);
    this->yGm.SetGlobalBuffer((__gm__ T*)output);
    this->pipe.InitBuffer(this->xInBuf, this->cLine * this->cAlign * sizeof(T));
    this->pipe.InitBuffer(this->yOutBuf, this->cLine * this->cAlign * sizeof(T));

    ClearOutput();

    this->outputGradUb = this->xInBuf.template Get<T>();
    this->inputGradUb = this->yOutBuf.template Get<T>();
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradNoCast<T>::ClearOutput() {
    auto yGmCount = this->outN * this->inD * this->inH * this->inW * this->cTotal;
    auto formerNum = yGmCount % this->coreNum;
    auto tailNum = this->coreNum - formerNum;
    auto formerCount = (yGmCount + this->coreNum - 1) / this->coreNum;
    auto tailCount = yGmCount / this->coreNum;
    if (this->thisCoreIdx < formerNum) {
        auto offset = this->thisCoreIdx * formerCount;
        InitOutput(this->yGm[offset], formerCount, static_cast<float>(0));
    } else {
        auto offset = formerNum * formerCount + (this->thisCoreIdx - formerNum) * tailCount;
        InitOutput(this->yGm[offset], tailCount, static_cast<float>(0));
    }
    SyncAll();
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradNoCast<T>::Process() {
    auto processStart = this->normalCoreCNum * this->thisCoreIdx;
    auto processEnd = processStart + this->thisCoreNum;

    if (this->kernelsOverlap) {
        SetAtomicAdd<T>();
    }

    SetFlag<HardEvent::V_MTE2>(this->eventIdV2Mte2);
    SetFlag<HardEvent::MTE3_V>(this->eventIdMte3ToV);
    for (auto index = processStart; index < processEnd; index += this->cLine) {
        auto indexStart = index;
        auto indexEnd = this->min(index + this->cLine, processEnd);
        WaitFlag<HardEvent::V_MTE2>(this->eventIdV2Mte2);
        WaitFlag<HardEvent::MTE3_V>(this->eventIdMte3ToV);
        for (auto i = index; i < indexEnd; ++i) {
            auto ubIndex = i - index;
            this->ProcessPerLine(i, ubIndex);
        }
        SetFlag<HardEvent::V_MTE2>(this->eventIdV2Mte2);
        SetFlag<HardEvent::MTE3_V>(this->eventIdMte3ToV);
    }
    WaitFlag<HardEvent::V_MTE2>(this->eventIdV2Mte2);
    WaitFlag<HardEvent::MTE3_V>(this->eventIdMte3ToV);

    if (this->kernelsOverlap) {
        SetAtomicNone();
    }
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradNoCast<T>::ProcessPerLine(int64_t index, int64_t ubIndex) {
    this->CalcIndex(index);

    this->CopyIn(this->indexN, this->indexD, this->indexH, this->indexW, this->indexC, ubIndex);
    SetFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV);

    WaitFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV);
    Compute(ubIndex);
    SetFlag<HardEvent::V_MTE3>(this->eventIdV2Mte3);
    
    WaitFlag<HardEvent::V_MTE3>(this->eventIdV2Mte3);
    for (auto iD = this->dStart; iD < this->dEnd; ++iD) {
        for (auto iH = this->hStart; iH < this->hEnd; ++iH) {
            for (auto iW = this->wStart; iW < this->wEnd; ++iW) {
                CopyOut(this->indexN, iD, iH, iW, this->indexC, ubIndex);
            }
        }
    }

    if (this->isDetermine) {
        pipe_barrier(PIPE_MTE3);
    }
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradNoCast<T>::Compute(int64_t ubIndex) {
    auto offset = ubIndex * this->cAlign;
    Muls(this->inputGradUb[offset], this->outputGradUb[offset], this->mulsFactor, this->thisIterCount);
}

template <typename T>
__aicore__ inline void KernelAvgPool3DGradNoCast<T>::CopyOut(int64_t n, int64_t d, int64_t h, int64_t w, int64_t c, int64_t ubIndex) {
    auto srcUbOffset = ubIndex * this->cAlign;
    auto dstGmOffset = n * (this->inD * this->inH * this->inW * this->cTotal) + 
                    d * (this->inH * this->inW * this->cTotal) + 
                    h * (this->inW * this->cTotal) + 
                    w * this->cTotal + 
                    c * this->cCount;
    DataCopyExtParams copyExtParams = {1, static_cast<uint32_t>(this->thisIterCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(this->yGm[dstGmOffset], this->inputGradUb[srcUbOffset], copyExtParams);
}

}  // namespace AvgPool3DGrad
#endif  // AVG_POOL3D_GRAD_NO_CAST_H