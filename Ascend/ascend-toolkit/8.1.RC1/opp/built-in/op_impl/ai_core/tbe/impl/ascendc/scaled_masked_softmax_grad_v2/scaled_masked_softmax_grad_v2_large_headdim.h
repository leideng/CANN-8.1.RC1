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
 * \file scaled_masked_softmax_grad_v2_large_headdim.h
 * \brief
 */
#ifndef SCALED_MASKED_SOFTMAX_GRAD_V2_LARGE_HEADDIM_H
#define SCALED_MASKED_SOFTMAX_GRAD_V2_LARGE_HEADDIM_H

#include "scaled_masked_softmax_grad_v2_base.h"

namespace ScaledMaskedSoftmaxGradV2 {
using namespace AscendC;

template <typename T>
class ScaledMaskedSoftmaxGradV2LargeHeadDim : public ScaledMaskedSoftmaxGradV2Base<T> {
public:
    __aicore__ inline ScaledMaskedSoftmaxGradV2LargeHeadDim() {}
    __aicore__ inline void Init(const GM_ADDR yGrad, const GM_ADDR y, const GM_ADDR mask, const GM_ADDR xGrad,
                                const ScaledMaskedSoftmaxGradV2TilingData& tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessBit16();
    __aicore__ inline void ProcessBit32();

protected:
    TBuf<TPosition::VECCALC> inBufferYGrad;
    TBuf<TPosition::VECCALC> inBufferY;

    __aicore__ inline void ComputeSoftmaxGradBit16(LocalTensor<T>& yGradLocal, LocalTensor<T>& yLocal,
        LocalTensor<float>& tmpBufYGrad, LocalTensor<float>& tmpBufY, const uint64_t& currentLoop);
    __aicore__ inline void DoSoftmaxGrad(LocalTensor<float>& yGradLocal, LocalTensor<float>& yLocal,
                                        const uint64_t& currentLoop);
    __aicore__ inline void ComputeReduceSum(LocalTensor<float>& dstLocal, LocalTensor<float>& srcLocal);
    __aicore__ inline void DoScaleAndMask(LocalTensor<float>& tmpOutLocal, LocalTensor<bool>& maskLocal);

private:
    int32_t eventIdMTE32MTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    int32_t eventIdMTE32V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    int32_t eventIdMTE22VA = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    int32_t eventIdMTE22VB = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    int32_t eventIdV2MTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    int32_t eventIdV2MTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    int32_t eventIdV2MTE2Bit16 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    UnaryRepeatParams repeatParams;
};

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2LargeHeadDim<T>::Init(const GM_ADDR yGrad, const GM_ADDR y,
    const GM_ADDR mask, const GM_ADDR xGrad, const ScaledMaskedSoftmaxGradV2TilingData& tiling)
{
    ScaledMaskedSoftmaxGradV2Base<T>::Init(yGrad, y, mask, xGrad, tiling);
    this->pipe.InitBuffer(inBufferYGrad, this->maxLineBytes);
    this->pipe.InitBuffer(inBufferY, this->maxLineBytes);

    repeatParams.dstBlkStride = BLK_STRIDE;
    repeatParams.srcBlkStride = BLK_STRIDE;
    repeatParams.dstRepStride = REP_STRIDE;
    repeatParams.srcRepStride = REP_STRIDE;
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2LargeHeadDim<T>::Process()
{
    if (this->currentCoreIdx >= this->usedCoreNum_) {
        return;
    }
    if constexpr (IsSameType<T, float>::value) {
        ProcessBit32();
    } else {
        ProcessBit16();
    }
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2LargeHeadDim<T>::ProcessBit16()
{
    for (uint64_t loop = 0; loop < this->loopTimes; ++loop) {
        if (loop == this->loopTimes - 1) {
            this->lineNum = this->minLine;
            this->moveNum = this->minLine * this->headDim_;
            this->calcNum = this->minLine * this->paddedHeadDim_;
        }
        uint64_t offset = loop * this->maxLinePerLoop_ * this->headDim_;
        LocalTensor<T> yGradLocal = inBufferYGrad.Get<T>();
        LocalTensor<T> yLocal = inBufferY.Get<T>();
        LocalTensor<float> tmpBufYGrad = this->yGradTmpBuffer.template Get<float>();
        LocalTensor<float> tmpBufY = this->yTmpBuffer.template Get<float>();
        ScaledMaskedSoftmaxGradV2Base<T>::CopyIn(yGradLocal, yLocal, offset);
        SetFlag<HardEvent::MTE2_V>(eventIdMTE22VA);
        WaitFlag<HardEvent::MTE2_V>(eventIdMTE22VA);
        if (loop != 0) {
            WaitFlag<HardEvent::MTE3_V>(eventIdMTE32V);
        }
        ComputeSoftmaxGradBit16(yGradLocal, yLocal, tmpBufYGrad, tmpBufY, loop);
        LocalTensor<T> out = tmpBufY.template ReinterpretCast<T>();
        pipe_barrier(PIPE_V);
        Cast(out, tmpBufYGrad, RoundMode::CAST_RINT, this->calcNum);
        SetFlag<HardEvent::V_MTE3>(eventIdV2MTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdV2MTE3);
        ScaledMaskedSoftmaxGradV2Base<T>::CopyOut(out, offset);
        if (loop != this->loopTimes - 1) {
            SetFlag<HardEvent::MTE3_V>(eventIdMTE32V);
        }
    }
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2LargeHeadDim<T>::ProcessBit32()
{
    for (uint64_t loop = 0; loop < this->loopTimes; ++loop) {
        if (loop == this->loopTimes - 1) {
            this->lineNum = this->minLine;
            this->moveNum = this->minLine * this->headDim_;
            this->calcNum = this->minLine * this->paddedHeadDim_;
        }
        uint64_t offset = loop * this->maxLinePerLoop_ * this->headDim_;
        LocalTensor<T> yGradLocal = inBufferYGrad.Get<T>();
        LocalTensor<T> yLocal = inBufferY.Get<T>();
        if (loop != 0) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE32MTE2);
        }
        ScaledMaskedSoftmaxGradV2Base<T>::CopyIn(yGradLocal, yLocal, offset);
        SetFlag<HardEvent::MTE2_V>(eventIdMTE22VA);
        WaitFlag<HardEvent::MTE2_V>(eventIdMTE22VA);
        DoSoftmaxGrad(yGradLocal, yLocal, loop);
        SetFlag<HardEvent::V_MTE3>(eventIdV2MTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdV2MTE3);
        ScaledMaskedSoftmaxGradV2Base<T>::CopyOut(yGradLocal, offset);
        if (loop != this->loopTimes - 1) {
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE32MTE2);
        }
    }
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2LargeHeadDim<T>::ComputeSoftmaxGradBit16(
    LocalTensor<T>& yGradLocal, LocalTensor<T>& yLocal, LocalTensor<float>& tmpBufYGrad,
    LocalTensor<float>& tmpBufY,const uint64_t& currentLoop)
{
    Cast(tmpBufYGrad, yGradLocal, RoundMode::CAST_NONE, this->calcNum);
    Cast(tmpBufY, yLocal, RoundMode::CAST_NONE, this->calcNum);
    pipe_barrier(PIPE_V);
    DoSoftmaxGrad(tmpBufYGrad, tmpBufY, currentLoop);
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2LargeHeadDim<T>::DoSoftmaxGrad(LocalTensor<float>& yGradLocal,
    LocalTensor<float>& yLocal, const uint64_t& currentLoop)
{
    LocalTensor<float> sumTmpBuf = this->sharedBuffer.template Get<float>();
    Mul(yGradLocal, yGradLocal, yLocal, this->calcNum);
    pipe_barrier(PIPE_V);
    if (this->dupNum != 0) {
        for (uint64_t i = 0; i < this->lineNum; ++i) {
            Duplicate(yGradLocal[i * this->paddedHeadDim_ + this->alignedHeadDim], MASK_VALUE, this->dupNum);
        }
        pipe_barrier(PIPE_V);
    }
    ComputeReduceSum(sumTmpBuf, yGradLocal);
    pipe_barrier(PIPE_V);
    for (uint64_t i = 0; i < this->lineNum; ++i) {
        Muls(yLocal[i * this->paddedHeadDim_], yLocal[i * this->paddedHeadDim_], sumTmpBuf.GetValue(i), MASK_LEN_B32,
            this->paddedHeadDim_ / MASK_LEN_B32, repeatParams);
    }
    pipe_barrier(PIPE_V);
    Sub(yGradLocal, yGradLocal, yLocal, this->calcNum);
    SetFlag<HardEvent::V_MTE2>(eventIdV2MTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIdV2MTE2);
    LocalTensor<bool> maskLocal = yLocal.template ReinterpretCast<bool>();
    ScaledMaskedSoftmaxGradV2Base<T>::CopyInMask(maskLocal, currentLoop);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE22VB);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE22VB);
    DoScaleAndMask(yGradLocal, maskLocal);
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2LargeHeadDim<T>::ComputeReduceSum(LocalTensor<float>& dstLocal,
    LocalTensor<float>& srcLocal)
{
    uint64_t repeatTimes = this->paddedHeadDim_ / MASK_LEN_B32;
    for (uint64_t i = 0; i < this->lineNum; ++i) {
        WholeReduceSum(dstLocal[i * MASK_LEN_B32], srcLocal[i * this->paddedHeadDim_], MASK_LEN_B32, repeatTimes, BLK_STRIDE, BLK_STRIDE, REP_STRIDE);
        pipe_barrier(PIPE_V);
        WholeReduceSum(dstLocal[i], dstLocal[i * MASK_LEN_B32], repeatTimes, 1, BLK_STRIDE, BLK_STRIDE, REP_STRIDE);
    }
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2LargeHeadDim<T>::DoScaleAndMask(LocalTensor<float>& tmpOutLocal,
    LocalTensor<bool>& maskLocal)
{
    if (this->scale_ != DEFAULT_SCALE) {
        Muls(tmpOutLocal, tmpOutLocal, this->scale_, this->calcNum);
        pipe_barrier(PIPE_V);
    }

    LocalTensor<uint8_t> maskTmpBuf = this->sharedBuffer.template Get<uint8_t>();
    SelectWithBytesMaskShapeInfo shapeInfo;
    shapeInfo.firstAxis = this->lineNum;
    shapeInfo.srcLastAxis = this->paddedHeadDim_;
    shapeInfo.maskLastAxis = this->paddedHeadDim_;
    tmpOutLocal.SetSize(this->calcNum);
    maskLocal.SetSize(this->calcNum);
    SelectWithBytesMask(tmpOutLocal, tmpOutLocal, MASK_VALUE, maskLocal, maskTmpBuf, shapeInfo);
}
} // namespace ScaledMaskedSoftmaxGradV2

#endif