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
 * \file scaled_masked_softmax_grad_v2_norm_headdim.h
 * \brief
 */
#ifndef SCALED_MASKED_SOFTMAX_GRAD_V2_NORM_HEADDIM_H
#define SCALED_MASKED_SOFTMAX_GRAD_V2_NORM_HEADDIM_H

#include "scaled_masked_softmax_grad_v2_base.h"

namespace ScaledMaskedSoftmaxGradV2 {
using namespace AscendC;

template <typename T>
class ScaledMaskedSoftmaxGradV2NormHeadDim : public ScaledMaskedSoftmaxGradV2Base<T> {
public:
    __aicore__ inline ScaledMaskedSoftmaxGradV2NormHeadDim() {}
    __aicore__ inline void Init(const GM_ADDR yGrad, const GM_ADDR y, const GM_ADDR mask, const GM_ADDR xGrad,
                                const ScaledMaskedSoftmaxGradV2TilingData& tilingData);
    __aicore__ inline void Process();

protected:
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueYGrad;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueMask;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueXGrad;

    __aicore__ inline void CopyIn(const uint64_t& offset);
    __aicore__ inline void CopyInMask(const uint64_t& currentLoop);
    __aicore__ inline void CopyOut(const uint64_t& offset);
    __aicore__ inline void ComputeSoftmaxGrad();
    __aicore__ inline void DoSoftmaxGrad(LocalTensor<float>& xGradLocal, LocalTensor<float>& yGradLocal,
                                        LocalTensor<float>& yLocal);
    __aicore__ inline void DoScaleAndMask(LocalTensor<float>& tmpOutLocal);

private:
    SoftMaxTiling softmaxTiling_;
};

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2NormHeadDim<T>::Init(const GM_ADDR yGrad, const GM_ADDR y,
    const GM_ADDR mask, const GM_ADDR xGrad, const ScaledMaskedSoftmaxGradV2TilingData& tiling)
{
    ScaledMaskedSoftmaxGradV2Base<T>::Init(yGrad, y, mask, xGrad, tiling);
    softmaxTiling_ = tiling.softmaxGradTilingData;
    this->pipe.InitBuffer(inQueueYGrad, BUFFER_NUM, this->maxLineBytes);
    this->pipe.InitBuffer(inQueueY, BUFFER_NUM, this->maxLineBytes);
    this->pipe.InitBuffer(inQueueMask, BUFFER_NUM, this->maxLineBytesMask);
    this->pipe.InitBuffer(outQueueXGrad, BUFFER_NUM, this->maxLineBytes);
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2NormHeadDim<T>::Process()
{
    if (this->currentCoreIdx >= this->usedCoreNum_) {
        return;
    }
    for (uint64_t loop = 0; loop < this->loopTimes; ++loop) {
        if (loop == this->loopTimes - 1) {
            this->lineNum = this->minLine;
            this->moveNum = this->minLine * this->headDim_;
            this->calcNum = this->minLine * this->paddedHeadDim_;
        }
        uint64_t offset = loop * this->maxLinePerLoop_ * this->headDim_;
        CopyIn(offset);
        CopyInMask(loop);
        ComputeSoftmaxGrad();
        CopyOut(offset);
    }
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2NormHeadDim<T>::CopyIn(const uint64_t& offset)
{
    LocalTensor<T> yGradLocal = inQueueYGrad.AllocTensor<T>();
    LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
    ScaledMaskedSoftmaxGradV2Base<T>::CopyIn(yGradLocal, yLocal, offset);
    inQueueYGrad.EnQue(yGradLocal);
    inQueueY.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2NormHeadDim<T>::CopyInMask(const uint64_t& currentLoop)
{
    LocalTensor<bool> maskLocal = inQueueMask.AllocTensor<bool>();
    ScaledMaskedSoftmaxGradV2Base<T>::CopyInMask(maskLocal, currentLoop);
    inQueueMask.EnQue(maskLocal);
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2NormHeadDim<T>::CopyOut(const uint64_t& offset)
{
    LocalTensor<T> outLocal = outQueueXGrad.DeQue<T>();
    ScaledMaskedSoftmaxGradV2Base<T>::CopyOut(outLocal, offset);
    outQueueXGrad.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2NormHeadDim<T>::ComputeSoftmaxGrad()
{
    LocalTensor<T> yGradLocal = inQueueYGrad.DeQue<T>();
    LocalTensor<T> yLocal = inQueueY.DeQue<T>();
    LocalTensor<T> xGradLocal = outQueueXGrad.AllocTensor<T>();
    if constexpr (IsSameType<T, float>::value) {
        DoSoftmaxGrad(xGradLocal, yGradLocal, yLocal);
        inQueueYGrad.FreeTensor(yGradLocal);
        inQueueY.FreeTensor(yLocal);
        DoScaleAndMask(xGradLocal);
    } else {
        LocalTensor<float> tmpBufYGrad = this->yGradTmpBuffer.template Get<float>();
        LocalTensor<float> tmpBufY = this->yTmpBuffer.template Get<float>();
        Cast(tmpBufYGrad, yGradLocal, RoundMode::CAST_NONE, this->calcNum);
        Cast(tmpBufY, yLocal, RoundMode::CAST_NONE, this->calcNum);
        inQueueYGrad.FreeTensor(yGradLocal);
        inQueueY.FreeTensor(yLocal);
        DoSoftmaxGrad(tmpBufYGrad, tmpBufYGrad, tmpBufY);
        DoScaleAndMask(tmpBufYGrad);
        Cast(xGradLocal, tmpBufYGrad, RoundMode::CAST_RINT, this->calcNum);
    }
    outQueueXGrad.EnQue(xGradLocal);
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2NormHeadDim<T>::DoSoftmaxGrad(LocalTensor<float>& xGradLocal,
    LocalTensor<float>& yGradLocal, LocalTensor<float>& yLocal)
{
    LocalTensor<uint8_t> softmaxGradTmpBuf = this->sharedBuffer.template Get<uint8_t>();
    SoftMaxShapeInfo srcShape = {static_cast<uint32_t>(this->lineNum), static_cast<uint32_t>(this->paddedHeadDim_),
        static_cast<uint32_t>(this->lineNum), static_cast<uint32_t>(this->paddedHeadDim_)};
    if (this->dupNum != 0) {
        for (uint64_t i = 0; i < this->lineNum; ++i) {
            Duplicate(yGradLocal[i * this->paddedHeadDim_ + this->alignedHeadDim], MASK_VALUE, this->dupNum);
            Duplicate(yLocal[i * this->paddedHeadDim_ + this->alignedHeadDim], MASK_VALUE, this->dupNum);
        }
        pipe_barrier(PIPE_V);
    }
    SoftmaxGrad<float, false, false>(xGradLocal, yGradLocal, yLocal, softmaxGradTmpBuf, softmaxTiling_, false, srcShape);
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2NormHeadDim<T>::DoScaleAndMask(LocalTensor<float>& tmpOutLocal)
{
    if (this->scale_ != DEFAULT_SCALE) {
        Muls(tmpOutLocal, tmpOutLocal, this->scale_, this->calcNum);
        pipe_barrier(PIPE_V);
    }

    LocalTensor<bool> maskLocal = inQueueMask.DeQue<bool>();
    LocalTensor<uint8_t> maskTmpBuf = this->sharedBuffer.template Get<uint8_t>();
    SelectWithBytesMaskShapeInfo shapeInfo;
    shapeInfo.firstAxis = this->lineNum;
    shapeInfo.srcLastAxis = this->paddedHeadDim_;
    shapeInfo.maskLastAxis = this->paddedHeadDim_;
    tmpOutLocal.SetSize(this->calcNum);
    maskLocal.SetSize(this->calcNum);
    SelectWithBytesMask(tmpOutLocal, tmpOutLocal, MASK_VALUE, maskLocal, maskTmpBuf, shapeInfo);
    inQueueMask.FreeTensor(maskLocal);
}
} // namespace ScaledMaskedSoftmaxGradV2

#endif