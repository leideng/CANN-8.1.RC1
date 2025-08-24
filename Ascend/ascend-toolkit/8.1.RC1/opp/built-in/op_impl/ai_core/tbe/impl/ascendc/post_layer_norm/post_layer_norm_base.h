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
 * \file post_layer_norm_base.h
 * \brief
 */
#ifndef ASDOPS_POST_LAYER_NORM_BASE
#define ASDOPS_POST_LAYER_NORM_BASE

#include "kernel_operator.h"
#include "utils.h"

using AscendC::HardEvent;

template <typename T>
class PostLayerNormBase {
public:
    __aicore__ inline PostLayerNormBase() {}

    __aicore__ inline void InitBase(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *gamma,
                                    __gm__ uint8_t *beta, PostLayerNormTilingData &tilingData)
    {
        zoomScale_ = tilingData.zoomScale;
        coreNum_ = tilingData.numCore;
        colNum_ = tilingData.numLastDim;
        uint32_t rowNumPerCore = tilingData.nlFirstdimPerCore;
        uint32_t rowNumLastCore = tilingData.lFirstdimPerCore;
        uint32_t rowNumPerTimes = tilingData.firstDimPerTimes;
        aveNum_ = tilingData.aveStr;
        eps_ = tilingData.epsStr;
        sliceNum_ = tilingData.sliceNum;
        sliceSize_ = tilingData.sliceSize;
        tailSliceSize_ = tilingData.tailSliceSize;
        if (AscendC::GetBlockIdx() != coreNum_ - 1) {
            rowNumCurrCore_ = rowNumPerCore;
            rowNumPerStep_ = rowNumPerTimes;
        } else {
            rowNumCurrCore_ = rowNumLastCore;
            rowNumPerStep_ = Min(rowNumPerTimes, rowNumCurrCore_);
        }

        rowNumLastStep_ = (rowNumCurrCore_ % rowNumPerTimes == 0) ? rowNumPerTimes :
                                                                    (rowNumCurrCore_ % rowNumPerTimes);

        gmOffset_ = static_cast<uint64_t>(rowNumPerCore) * colNum_ * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ T *)x + gmOffset_);
        yGm.SetGlobalBuffer((__gm__ T *)y + gmOffset_);
        gammaGm.SetGlobalBuffer((__gm__ T *)gamma);
        betaGm.SetGlobalBuffer((__gm__ T *)beta);

        pipe.InitBuffer(xQue, BUFFER_NUM, rowNumPerStep_ * RoundUp(sliceSize_) * sizeof(T));
        pipe.InitBuffer(yQue, BUFFER_NUM, rowNumPerStep_ * RoundUp(sliceSize_) * sizeof(T));
        pipe.InitBuffer(betaQue, BUFFER_NUM, RoundUp(sliceSize_) * sizeof(T));
        pipe.InitBuffer(gammaQue, BUFFER_NUM, RoundUp(sliceSize_) * sizeof(T));
        pipe.InitBuffer(xBufFp32, RoundUp(sliceSize_) * sizeof(float));
        pipe.InitBuffer(yBufFp32, RoundUp(sliceSize_) * sizeof(float));
        yLocalFp32 = yBufFp32.Get<float>();
        xLocalFp32 = xBufFp32.Get<float>();
        xLocalFp16 = xBufFp32.Get<T>();
    }

    __aicore__ inline void CopyInMultiRow(uint64_t procId, int32_t size)
    {
        uint64_t offset = procId * rowNumPerStep_ * colNum_;
        CopyIn(xGm, xQue, offset, size);
        CopyIn(yGm, yQue, offset, size);
        CopyIn(betaGm, betaQue, 0, colNum_);
        CopyIn(gammaGm, gammaQue, 0, colNum_);
    }

    __aicore__ inline void ComputeResidual(uint64_t offset, uint32_t count)
    {
        CopyInAndCastF32(xLocalFp32, xGm, xQue, offset, count);
        CopyInAndCastF32(yLocalFp32, yGm, yQue, offset, count);
        Muls(yLocalFp32, yLocalFp32, zoomScale_, count);
        AscendC::PipeBarrier<PIPE_V>();
        ComputeResidualAdd(xLocalFp32, xLocalFp32, yLocalFp32, count);
    }

    __aicore__ inline void GetSizeAndOffset(uint64_t rowIdx, uint64_t sliceIdx,
        uint32_t &size, uint64_t &offset, uint64_t &sliceOffset)
    {
        size = (sliceIdx != sliceNum_ - 1) ? sliceSize_ : tailSliceSize_;
        sliceOffset = sliceIdx * sliceSize_;
        offset = rowIdx * colNum_ + sliceOffset;
    }

    __aicore__ inline float ComputeRowMean(uint64_t rowIdx)
    {
        float sum = 0;
        for (uint64_t sliceIdx = 0; sliceIdx < sliceNum_; sliceIdx++) {
            uint32_t size = 0;
            uint64_t offset = 0;
            uint64_t sliceOffset = 0;
            GetSizeAndOffset(rowIdx, sliceIdx, size, offset, sliceOffset);
            ComputeResidual(offset, size);
            sum += ComputeSum(xLocalFp32, yLocalFp32, yLocalFp32, size);
            AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
        }

        return sum * aveNum_;
    }

    __aicore__ inline float ComputeRowStd(uint64_t rowIdx, float mean)
    {
        float squareSum = 0;
        for (uint64_t sliceIdx = 0; sliceIdx < sliceNum_; sliceIdx++) {
            uint32_t size = 0;
            uint64_t offset = 0;
            uint64_t sliceOffset = 0;
            GetSizeAndOffset(rowIdx, sliceIdx, size, offset, sliceOffset);
            ComputeResidual(offset, size);
            Adds(yLocalFp32, xLocalFp32, -mean, size);
            AscendC::PipeBarrier<PIPE_V>();
            squareSum += ComputeSliceSquareSum(yLocalFp32, yLocalFp32, yLocalFp32, size);
            AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
        }

        float var = squareSum * aveNum_ + eps_;
        return sqrt(var);
    }

    __aicore__ inline void ComputeSliceLayernorm(uint64_t offset, uint64_t sliceOffset, uint32_t size,
        float mean, float std)
    {
        ComputeResidual(offset, size);
        Adds(yLocalFp32, xLocalFp32, -mean, size);
        AscendC::PipeBarrier<PIPE_V>();
        if (std == 0) {
            return;
        }
        Muls(yLocalFp32, yLocalFp32, (float)1.0 / std, size);
        AscendC::PipeBarrier<PIPE_V>();
        CopyInAndCastF32(xLocalFp32, gammaGm, gammaQue, sliceOffset, size);
        Mul(yLocalFp32, xLocalFp32, yLocalFp32, size);
        AscendC::PipeBarrier<PIPE_V>();
        CopyInAndCastF32(xLocalFp32, betaGm, betaQue, sliceOffset, size);
        Add(yLocalFp32, xLocalFp32, yLocalFp32, size);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeRes(uint32_t rid, const AscendC::LocalTensor<T> &xLocal,
        const AscendC::LocalTensor<T> &yLocal)
    {
        const AscendC::LocalTensor<T> &xLocalInner = xLocal[rid * colNum_];
        const AscendC::LocalTensor<T> &yLocalInner = yLocal[rid * colNum_];
        CastFrom16To32(xLocalFp32, xLocalInner, colNum_);
        CastFrom16To32(yLocalFp32, yLocalInner, colNum_);
        Muls(yLocalFp32, yLocalFp32, zoomScale_, colNum_);
        AscendC::PipeBarrier<PIPE_V>();
        ComputeResidualAdd(xLocalFp32, xLocalFp32, yLocalFp32, colNum_);
    }

    __aicore__ inline void ComputeRowLayernorm(const AscendC::LocalTensor<T> gamma,
        const AscendC::LocalTensor<T> beta)
    {
        ComputeMean(yLocalFp32, xLocalFp32, aveNum_, colNum_);
        ComputeLayerNorm(yLocalFp32, xLocalFp32, yLocalFp32, eps_, aveNum_, gamma, beta, colNum_);
    }
protected:
    static constexpr int32_t BUFFER_NUM = 1;

protected:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> xQue;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> yQue;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> gammaQue;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> betaQue;

    AscendC::TBuf<AscendC::TPosition::VECCALC> xBufFp32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yBufFp32;
    AscendC::LocalTensor<float> yLocalFp32;
    AscendC::LocalTensor<float> xLocalFp32;
    AscendC::LocalTensor<T> xLocalFp16;
    AscendC::GlobalTensor<T> xGm, yGm, gammaGm, betaGm;

    uint32_t coreNum_{0};
    uint32_t colNum_{0};
    uint32_t rowNumCurrCore_{0};
    uint32_t rowNumPerStep_{0};
    uint32_t rowNumLastStep_{0};
    float aveNum_{0};
    float eps_{0};
    float zoomScale_{0};
    uint32_t sliceNum_{0};
    uint32_t sliceSize_{0};
    uint32_t tailSliceSize_{0};
    uint64_t gmOffset_{0};
};
#endif