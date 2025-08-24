/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_flashv2_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_V220_SOFTMAX_FLASHV2_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_V220_SOFTMAX_FLASHV2_IMPL_H

#include "softmax_common_impl.h"
#include "../common/softmax_flashv2_impl/softmax_flashv2_basic_block_impl.h"
#include "../common/softmax_flashv2_impl/softmax_flashv2_update_impl.h"
#include "../common/softmax_flashv2_impl/softmax_flashv2_no_update_impl.h"
#include "../common/softmax_flashv2_impl/softmax_flashv2_nz_impl.h"
#include "../common/softmax_flashv2_impl/softmax_flashv2_common_impl.h"

namespace AscendC {
    __aicore__ inline void SoftMaxFlashV2M1CastIntrinsicsImpl(const LocalTensor<float>& dstLocal, const LocalTensor<half>& srcLocal,
        const uint32_t calCount)
    {
        UnaryRepeatParams unaryParams;
        unaryParams.srcRepStride = DEFAULT_REPEAT_STRIDE / sizeof(half);
    
        SetVectorMask<float, MaskMode::COUNTER>(0, calCount);
        Cast<float, half, false>(dstLocal, srcLocal, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
    }
    
    __aicore__ inline void SoftMaxFlashV2M1CastIntrinsicsImpl(const LocalTensor<half>& dstLocal, const LocalTensor<float>& srcLocal,
        const uint32_t calCount)
    {
        UnaryRepeatParams unaryParams;
        unaryParams.dstRepStride = DEFAULT_REPEAT_STRIDE / sizeof(half);
    
        SetVectorMask<float, MaskMode::COUNTER>(0, calCount);
        Cast<half, float, false>(dstLocal, srcLocal, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
    }
    
    template <bool isBasicBlock = false>
    __aicore__ inline void SoftmaxFlashV2M1BrcbSubImpl(const LocalTensor<float>& dstLocal, const LocalTensor<float>& src0Local,
        const LocalTensor<float>& src1Local, const LocalTensor<float>& tmpBuffer, const uint32_t srcM, const uint32_t srcK)
    {
        // (m,k) - (m,1)
        uint32_t splitCeilM = DivCeil(srcM, FLOAT_NUM_PER_BLK);
        Brcb(tmpBuffer, src1Local, splitCeilM, { 1, DEFAULT_REPEAT_STRIDE }); // (m,1) -> (m,8)
        PipeBarrier<PIPE_V>();
    
        if constexpr (isBasicBlock) {
            uint32_t splitBlock = srcK / FLOAT_REPEAT_SIZE;
            uint8_t offset = srcK / FLOAT_NUM_PER_BLK;
            SetVectorMask<float, MaskMode::COUNTER>(0, srcM * FLOAT_REPEAT_SIZE);
            for (uint32_t j = 0; j < splitBlock; ++j) {
                Sub<float, false>(dstLocal[FLOAT_REPEAT_SIZE * j], src0Local[FLOAT_REPEAT_SIZE * j], tmpBuffer,
                    MASK_PLACEHOLDER, 1, { 1, 1, 0, offset, offset, 1 }); // (m,k) - (m,8)
            }
            PipeBarrier<PIPE_V>();
        } else if (srcK < SOFTMAX_SUB_DIV_ROW_COLUMN_SIZE) {
            uint8_t blockStride = srcK / FLOAT_NUM_PER_BLK;
            SetVectorMask<float, MaskMode::COUNTER>(0, srcM * FLOAT_NUM_PER_BLK);
            for (uint8_t j = 0; j < blockStride; j++) {
                Sub<float, false>(dstLocal[j * FLOAT_NUM_PER_BLK], src0Local[j * FLOAT_NUM_PER_BLK], tmpBuffer,
                    MASK_PLACEHOLDER, 1, { blockStride, blockStride, DEFAULT_BLK_STRIDE, (uint8_t)srcK, (uint8_t)srcK, DEFAULT_REPEAT_STRIDE});
            }
            PipeBarrier<PIPE_V>();
        } else {
            SetVectorMask<float, MaskMode::COUNTER>(0, srcK);
            for (uint32_t j = 0; j < srcM; j++) {
                Sub<float, false>(dstLocal[j * srcK], src0Local[j * srcK], tmpBuffer[j * FLOAT_NUM_PER_BLK],
                    MASK_PLACEHOLDER, 1, { 1, 1, 0, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, 0});
            }
            PipeBarrier<PIPE_V>();
        }
    }
    
    __aicore__ inline void SoftmaxFlashV2M1NoUpdateBasicBlockProcess(const LocalTensor<float>& dstLocal,
        const LocalTensor<float>& expSumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<float>& srcLocal,
        const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling)
    {
        UnaryRepeatParams unaryParams;
        const LocalTensor<float>& reduceBuffer = workLocal; // [splitM, 64]
        uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
    
        SetMaskNorm();
        ResetMask();
    
        // reduceMax (m,k) -> (m,1)
        BasicBlockReduceMaxImpl(maxTensor, srcLocal, reduceBuffer, splitBlock, tiling.splitM, tiling.splitK);
        PipeBarrier<PIPE_V>();
    
        SetMaskCount();
    
        // src(m,k) - max(m,1)
        SoftmaxFlashV2M1BrcbSubImpl<true>(srcLocal, srcLocal, maxTensor, reduceBuffer, tiling.splitM, tiling.splitK);
    
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.splitSize);
        Exp<float, false>(dstLocal, srcLocal, MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
    
        SetMaskNorm();
        ResetMask();
    
        // reduceMax (m,k) -> (m,1)
        BasicBlockReduceSumImpl(expSumTensor, dstLocal, reduceBuffer, splitBlock, tiling.splitM, tiling.splitK);
        PipeBarrier<PIPE_V>();
    
        SetMaskCount();
    }
    
    __aicore__ inline void SoftmaxFlashV2M1NoUpdateBasicBlock(const LocalTensor<half>& dstLocal,
        const LocalTensor<half>& expSumTensor, const LocalTensor<half>& maxTensor, const LocalTensor<half>& srcLocal,
        const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling)
    {
        const LocalTensor<float>& srcBuffer = workLocal; // [splitM, srcK]
        const LocalTensor<float>& sumBuffer = srcBuffer[tiling.splitSize]; // [splitM, 1]
        const LocalTensor<float>& maxBuffer = sumBuffer[tiling.splitM]; // [splitM, 1]
        const LocalTensor<float>& tmpBuffer = maxBuffer[tiling.splitM]; // [splitM, 64]
    
        SoftMaxFlashV2M1CastIntrinsicsImpl(srcBuffer, srcLocal, tiling.splitSize);
        SoftmaxFlashV2M1NoUpdateBasicBlockProcess(srcBuffer, sumBuffer, maxBuffer, srcBuffer, tmpBuffer, tiling);
        SoftMaxFlashV2M1CastIntrinsicsImpl(dstLocal, srcBuffer, tiling.splitSize);
        SoftMaxFlashV2M1CastIntrinsicsImpl(expSumTensor, sumBuffer, tiling.splitM);
        SoftMaxFlashV2M1CastIntrinsicsImpl(maxTensor, maxBuffer, tiling.splitM);
    }
    
    __aicore__ inline void SoftmaxFlashV2M1NoUpdateBasicBlock(const LocalTensor<float>& dstLocal,
        const LocalTensor<float>& expSumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<float>& srcLocal,
        const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling)
    {
        SoftmaxFlashV2M1NoUpdateBasicBlockProcess(dstLocal, expSumTensor, maxTensor, srcLocal, workLocal, tiling);
    }
    
    __aicore__ inline void SoftmaxFlashV2M1NoUpdateBasicBlock(const LocalTensor<half>& dstLocal,
        const LocalTensor<float>& expSumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcLocal,
        const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling)
    {
        const LocalTensor<float>& srcBuffer = workLocal; // [splitM, srcK]
        const LocalTensor<float>& tmpBuffer = srcBuffer[tiling.splitSize]; // [splitM, 64]
    
        SoftMaxFlashV2M1CastIntrinsicsImpl(srcBuffer, srcLocal, tiling.splitSize);
        SoftmaxFlashV2M1NoUpdateBasicBlockProcess(srcBuffer, expSumTensor, maxTensor, srcBuffer, tmpBuffer, tiling);
        SoftMaxFlashV2M1CastIntrinsicsImpl(dstLocal, srcBuffer, tiling.splitSize);
    }
    
    __aicore__ inline void SoftmaxFlashV2M1BasicBlockImplProcess(const LocalTensor<float>& dstLocal, const LocalTensor<float>& expSumTensor,
        const LocalTensor<float>& maxTensor, const LocalTensor<float>& srcLocal, const LocalTensor<float>& expMaxTensor,
        const LocalTensor<float>& inExpSumTensor, const LocalTensor<float>& inMaxTensor, const LocalTensor<float>& workLocal,
        const SoftMaxTiling& tiling)
    {
        UnaryRepeatParams unaryParams;
        BinaryRepeatParams binaryParams;
        CopyRepeatParams copyParams(1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
    
        const LocalTensor<float>& reduceBuffer = workLocal; // [splitM, 64]
        const LocalTensor<float>& tmpBufferM1 = workLocal[tiling.splitM * FLOAT_REPEAT_SIZE]; // [splitM, 8]
    
        uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
    
        SetMaskNorm();
        ResetMask();
    
        // reduceMax (m,k) -> (m,1)
        BasicBlockReduceMaxImpl(tmpBufferM1, srcLocal, reduceBuffer, splitBlock, tiling.splitM, tiling.splitK);
        PipeBarrier<PIPE_V>();
    
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.splitM);
    
        // backup inMaxTensor
        Copy<float, false>(reduceBuffer, inMaxTensor, MASK_PLACEHOLDER, 1, copyParams);
        PipeBarrier<PIPE_V>();
    
        // max(inmax, reducemax)
        Max<float, false>(maxTensor, inMaxTensor, tmpBufferM1, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();
    
        // expmax = exp(inmax - max)
        Sub<float, false>(tmpBufferM1, reduceBuffer, maxTensor, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();
    
        Exp<float, false>(expMaxTensor, tmpBufferM1, MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
    
        // src[m,k] - max[m,1]
        SoftmaxFlashV2M1BrcbSubImpl<true>(dstLocal, srcLocal, maxTensor, tmpBufferM1, tiling.splitM, tiling.splitK);
    
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.splitSize);
        Exp<float, false>(dstLocal, dstLocal, MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
    
        SetMaskNorm();
        ResetMask();
    
        BasicBlockReduceSumImpl(tmpBufferM1, dstLocal, reduceBuffer, splitBlock, tiling.splitM, tiling.splitK);
        PipeBarrier<PIPE_V>();
    
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.splitM);
    
        // update sum = expmax * insum + sum
        Mul<float, false>(inExpSumTensor, expMaxTensor, inExpSumTensor, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();
    
        Add<float, false>(expSumTensor, inExpSumTensor, tmpBufferM1, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();
    }
    
    __aicore__ inline void SoftmaxFlashV2M1BasicBlockImpl(const LocalTensor<float>& dstLocal, const LocalTensor<float>& expSumTensor,
        const LocalTensor<float>& maxTensor, const LocalTensor<float>& srcLocal, const LocalTensor<float>& expMaxTensor,
        const LocalTensor<float>& inExpSumTensor, const LocalTensor<float>& inMaxTensor, const LocalTensor<float>& workLocal,
        const SoftMaxTiling& tiling)
    {
        SoftmaxFlashV2M1BasicBlockImplProcess(dstLocal, expSumTensor, maxTensor, srcLocal, expMaxTensor,
            inExpSumTensor, inMaxTensor, workLocal, tiling);
    }
    
    __aicore__ inline void SoftmaxFlashV2M1BasicBlockImpl(const LocalTensor<half>& dstLocal, const LocalTensor<half>& expSumTensor,
        const LocalTensor<half>& maxTensor, const LocalTensor<half>& srcLocal, const LocalTensor<half>& expMaxTensor,
        const LocalTensor<half>& inExpSumTensor, const LocalTensor<half>& inMaxTensor, const LocalTensor<float>& workLocal,
        const SoftMaxTiling& tiling)
    {
        const LocalTensor<float>& srcBuffer = workLocal; // [splitM, srcK]
        const LocalTensor<float>& sumBuffer = srcBuffer[tiling.splitSize]; // [splitM, 1]
        const LocalTensor<float>& maxBuffer = sumBuffer[tiling.splitM]; // [splitM, 1]
        const LocalTensor<float>& expMaxBuffer = maxBuffer[tiling.splitM]; // [splitM, 1]
        const LocalTensor<float>& tmpBuffer = expMaxBuffer[tiling.splitM]; // [splitM, 64] + [splitM, 8]
    
        SoftMaxFlashV2M1CastIntrinsicsImpl(srcBuffer, srcLocal, tiling.splitSize);
        SoftMaxFlashV2M1CastIntrinsicsImpl(sumBuffer, inExpSumTensor, tiling.splitM);
        SoftMaxFlashV2M1CastIntrinsicsImpl(maxBuffer, inMaxTensor, tiling.splitM);
    
        SoftmaxFlashV2M1BasicBlockImplProcess(srcBuffer, sumBuffer, maxBuffer, srcBuffer, expMaxBuffer, sumBuffer, maxBuffer, tmpBuffer, tiling);
    
        SoftMaxFlashV2M1CastIntrinsicsImpl(dstLocal, srcBuffer, tiling.splitSize);
        SoftMaxFlashV2M1CastIntrinsicsImpl(expSumTensor, sumBuffer, tiling.splitM);
        SoftMaxFlashV2M1CastIntrinsicsImpl(maxTensor, maxBuffer, tiling.splitM);
        SoftMaxFlashV2M1CastIntrinsicsImpl(expMaxTensor, expMaxBuffer, tiling.splitM);
    }
    
    __aicore__ inline void SoftmaxFlashV2M1BasicBlockImpl(const LocalTensor<half>& dstLocal, const LocalTensor<float>& expSumTensor,
        const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcLocal, const LocalTensor<half>& expMaxTensor,
        const LocalTensor<float>& inExpSumTensor, const LocalTensor<float>& inMaxTensor,
        const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling)
    {
        const LocalTensor<float>& srcBuffer = workLocal; // [splitM, srcK]
        const LocalTensor<float>& expMaxBuffer = srcBuffer[tiling.splitSize]; // [splitM, 1]
        const LocalTensor<float>& tmpBuffer = expMaxBuffer[tiling.splitM]; // [splitM, 64] + [splitM, 8]
    
        SoftMaxFlashV2M1CastIntrinsicsImpl(srcBuffer, srcLocal, tiling.splitSize);
        SoftmaxFlashV2M1BasicBlockImplProcess(srcBuffer, expSumTensor, maxTensor, srcBuffer,
            expMaxBuffer, inExpSumTensor, inMaxTensor, tmpBuffer, tiling);
        SoftMaxFlashV2M1CastIntrinsicsImpl(dstLocal, srcBuffer, tiling.splitSize);
        SoftMaxFlashV2M1CastIntrinsicsImpl(expMaxTensor, expMaxBuffer, tiling.splitM);
    }
    
    __aicore__ inline void SoftmaxFlashV2M1NoUpdateImplProcess(const LocalTensor<float>& dstLocal, const LocalTensor<float>& expSumTensor,
        const LocalTensor<float>& maxTensor, const LocalTensor<float>& srcLocal, const LocalTensor<float>& workLocal,
        const ReduceLastND& reduceParam)
    {
        UnaryRepeatParams unaryParams;
        const LocalTensor<float>& tmpBuffer = workLocal; // [splitM, 64]
    
        SetMaskNorm();
        ResetMask();
    
        NewReduceMaxLastNDImpl<false>(maxTensor, srcLocal, tmpBuffer, reduceParam);
        PipeBarrier<PIPE_V>();
    
        SetMaskCount();
    
        SoftmaxFlashV2M1BrcbSubImpl(dstLocal, srcLocal, maxTensor, tmpBuffer, reduceParam.srcM, reduceParam.srcK);
        PipeBarrier<PIPE_V>();
    
        SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM * reduceParam.srcK);
        Exp<float, false>(dstLocal, dstLocal, MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
    
        SetMaskNorm();
        ResetMask();
    
        NewReduceSumLastNDImpl<false>(expSumTensor, dstLocal, tmpBuffer, reduceParam);
        PipeBarrier<PIPE_V>();
    
        SetMaskCount();
    }
    
    template <typename T1, typename T2>
    __aicore__ inline void SoftmaxFlashV2M1NoUpdateImplPreCast(LocalTensor<float>& dstLocalOut,
        LocalTensor<float>& expSumTensorOut, LocalTensor<float>& maxTensorOut, LocalTensor<float>& srcLocalOut,
        LocalTensor<float>& workLocalOut, const LocalTensor<T1>& dstLocalIn, const LocalTensor<T2>& expSumTensorIn,
        const LocalTensor<T2>& maxTensorIn, const LocalTensor<T1>& srcLocalIn, const LocalTensor<float>& workLocalIn,
        const ReduceLastND& param)
    {
        uint32_t splitM = DivCeil(param.srcM, FLOAT_NUM_PER_BLK) * FLOAT_NUM_PER_BLK;
        uint32_t splitSize = param.srcM * param.srcK;
        if constexpr (SupportType<T1, float>() && SupportType<T2, float>()) {
            dstLocalOut = dstLocalIn;
            expSumTensorOut = expSumTensorIn;
            maxTensorOut = maxTensorIn;
            srcLocalOut = srcLocalIn;
            workLocalOut = workLocalIn;
        } else if constexpr (SupportType<T1, half>() && SupportType<T2, half>()) {
            LocalTensor<float> srcBuffer = workLocalIn; // [splitM, srcK]
            LocalTensor<float> sumBuffer = srcBuffer[splitSize]; // [splitM, 1]
            LocalTensor<float> maxBuffer = sumBuffer[splitM]; // [splitM, 1]
            LocalTensor<float> tmpBuffer = maxBuffer[splitM]; // [splitM, 64] + [splitM, 8]
            SoftMaxFlashV2M1CastIntrinsicsImpl(srcBuffer, srcLocalIn, splitSize);
            dstLocalOut = srcBuffer;
            expSumTensorOut = sumBuffer;
            maxTensorOut = maxBuffer;
            srcLocalOut = srcBuffer;
            workLocalOut = tmpBuffer;
        } else if constexpr (SupportType<T1, half>() && SupportType<T2, float>()) {
            LocalTensor<float> srcBuffer = workLocalIn; // [splitM, srcK]
            LocalTensor<float> tmpBuffer = srcBuffer[splitM]; // [splitM, 64] + [splitM, 8]
            SoftMaxFlashV2M1CastIntrinsicsImpl(srcBuffer, srcLocalIn, splitSize);
            dstLocalOut = srcBuffer;
            expSumTensorOut = expSumTensorIn;
            maxTensorOut = maxTensorIn;
            srcLocalOut = srcBuffer;
            workLocalOut = tmpBuffer;
        }
    }
    
    template <typename T1, typename T2>
    __aicore__ inline void SoftmaxFlashV2M1NoUpdateImplPostCast(const LocalTensor<T1>& dstLocalOut,
        const LocalTensor<T2>& expSumTensorOut, const LocalTensor<T2>& maxTensorOut,
        const LocalTensor<float>& dstLocalIn, const LocalTensor<float>& expSumTensorIn, const LocalTensor<float>& maxTensorIn,
        const ReduceLastND& param)
    {
        uint32_t splitM = param.srcM;
        uint32_t splitSize = param.srcM * param.srcK;
        if constexpr (SupportType<T1, half>() && SupportType<T2, half>()) {
            SoftMaxFlashV2M1CastIntrinsicsImpl(dstLocalOut, dstLocalIn, splitSize);
            SoftMaxFlashV2M1CastIntrinsicsImpl(expSumTensorOut, expSumTensorIn, splitM);
            SoftMaxFlashV2M1CastIntrinsicsImpl(maxTensorOut, maxTensorIn, splitM);
        } else if constexpr (SupportType<T1, half>() && SupportType<T2, float>()) {
            SoftMaxFlashV2M1CastIntrinsicsImpl(dstLocalOut, dstLocalIn, splitSize);
        }
    }
    
    template <typename T1, typename T2>
    __aicore__ inline void SoftmaxFlashV2M1NoUpdateImpl(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& expSumTensor,
        const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& srcLocal, const LocalTensor<float>& workLocal,
        const ReduceLastND& reduceParam, const SoftMaxTiling& tiling)
    {
        LocalTensor<float> dstLocalOut;
        LocalTensor<float> expSumTensorOut;
        LocalTensor<float> maxTensorOut;
        LocalTensor<float> srcLocalOut;
        LocalTensor<float> tmpBuffer;
    
        SoftmaxFlashV2M1NoUpdateImplPreCast<T1, T2>(dstLocalOut, expSumTensorOut, maxTensorOut, srcLocalOut, tmpBuffer,
            dstLocal, expSumTensor, maxTensor, srcLocal, workLocal, reduceParam);
        SoftmaxFlashV2M1NoUpdateImplProcess(dstLocalOut, expSumTensorOut, maxTensorOut, srcLocalOut, tmpBuffer, reduceParam);
        SoftmaxFlashV2M1NoUpdateImplPostCast<T1, T2>(dstLocal, expSumTensor, maxTensor, dstLocalOut, expSumTensorOut, maxTensorOut, reduceParam);
    }
    
    __aicore__ inline void SoftmaxFlashV2M1UpdateImplProcess(const LocalTensor<float>& dstLocal, const LocalTensor<float>& expSumTensor,
        const LocalTensor<float>& maxTensor, const LocalTensor<float>& srcLocal, const LocalTensor<float>& expMaxTensor,
        const LocalTensor<float>& inExpSumTensor, const LocalTensor<float>& inMaxTensor,
        const LocalTensor<float>& workLocal, const ReduceLastND& reduceParam, const SoftMaxTiling& tiling)
    {
        UnaryRepeatParams unaryParams;
        BinaryRepeatParams binaryParams;
        CopyRepeatParams copyParams(1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        const LocalTensor<float>& tmpBuffer0 = workLocal; // [splitM, 8]
        const LocalTensor<float>& reduceBuffer = workLocal[tiling.reduceSize]; // [splitM, 64]
    
        SetMaskNorm();
        ResetMask();
    
        NewReduceMaxLastNDImpl<false>(tmpBuffer0, srcLocal, reduceBuffer, reduceParam);
        PipeBarrier<PIPE_V>();
    
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM);
    
        Max<float, false>(tmpBuffer0, inMaxTensor, tmpBuffer0, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();
    
        Sub<float, false>(reduceBuffer, inMaxTensor, tmpBuffer0, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();
    
        Exp<float, false>(expMaxTensor, reduceBuffer, MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
    
        SoftmaxFlashV2M1BrcbSubImpl(dstLocal, srcLocal, tmpBuffer0, reduceBuffer, reduceParam.srcM, reduceParam.srcK);
        PipeBarrier<PIPE_V>();
    
        SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM * reduceParam.srcK);
        Exp<float, false>(dstLocal, dstLocal, MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
    
        SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM);
        Copy<float, false>(maxTensor, tmpBuffer0, MASK_PLACEHOLDER, 1, copyParams);
        PipeBarrier<PIPE_V>();
    
        SetMaskNorm();
        ResetMask();
    
        NewReduceSumLastNDImpl<false>(tmpBuffer0, dstLocal, reduceBuffer, reduceParam);
    
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcM);
    
        Mul<float, false>(expSumTensor, expMaxTensor, inExpSumTensor, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();
    
        Add<float, false>(expSumTensor, expSumTensor, tmpBuffer0, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();
    }
    
    template <typename T1, typename T2>
    __aicore__ inline void SoftmaxFlashV2M1UpdateImplPreCast(LocalTensor<float>& dstLocalOut, LocalTensor<float>& expSumTensorOut,
        LocalTensor<float>& maxTensorOut, LocalTensor<float>& srcLocalOut, LocalTensor<float>& expMaxTensorOut, LocalTensor<float>& inExpSumTensorOut,
        LocalTensor<float>& inMaxTensorOut, LocalTensor<float>& workLocalOut, const LocalTensor<T1>& dstLocalIn, const LocalTensor<T2>& expSumTensorIn,
        const LocalTensor<T2>& maxTensorIn, const LocalTensor<T1>& srcLocalIn, const LocalTensor<T1>& expMaxTensorIn,
        const LocalTensor<T2>& inExpSumTensorIn, const LocalTensor<T2>& inMaxTensorIn, const LocalTensor<float>& workLocalIn, const ReduceLastND& param)
    {
        uint32_t splitM = DivCeil(param.srcM, FLOAT_NUM_PER_BLK) * FLOAT_NUM_PER_BLK;
        uint32_t splitSize = param.srcM * param.srcK;
        if constexpr (SupportType<T1, float>() && SupportType<T2, float>()) {
            dstLocalOut = dstLocalIn;
            expSumTensorOut = expSumTensorIn;
            maxTensorOut = maxTensorIn;
            srcLocalOut = srcLocalIn;
            expMaxTensorOut = expMaxTensorIn;
            inExpSumTensorOut = inExpSumTensorIn;
            inMaxTensorOut = inMaxTensorIn;
            workLocalOut = workLocalIn;
        } else if constexpr (SupportType<T1, half>() && SupportType<T2, half>()) {
            const LocalTensor<float>& srcBuffer = workLocalIn; // [splitM, srcK]
            const LocalTensor<float>& sumBuffer = srcBuffer[splitSize]; // [splitM, 1]
            const LocalTensor<float>& maxBuffer = sumBuffer[splitM]; // [splitM, 1]
            const LocalTensor<float>& expMaxBuffer = maxBuffer[splitM]; // [splitM, 1]
            const LocalTensor<float>& tmpBuffer = expMaxBuffer[splitM]; // [splitM, 64] + [splitM, 8]
            SoftMaxFlashV2M1CastIntrinsicsImpl(srcBuffer, srcLocalIn, splitSize);
            SoftMaxFlashV2M1CastIntrinsicsImpl(sumBuffer, inExpSumTensorIn, param.srcM);
            SoftMaxFlashV2M1CastIntrinsicsImpl(maxBuffer, inMaxTensorIn, param.srcM);
            dstLocalOut = srcBuffer;
            expSumTensorOut = sumBuffer;
            maxTensorOut = maxBuffer;
            srcLocalOut = srcBuffer;
            expMaxTensorOut = expMaxBuffer;
            inExpSumTensorOut = sumBuffer;
            inMaxTensorOut = maxBuffer;
            workLocalOut = tmpBuffer;
        } else if constexpr (SupportType<T1, half>() && SupportType<T2, float>()) {
            const LocalTensor<float>& srcBuffer = workLocalIn; // [splitM, srcK]
            const LocalTensor<float>& expMaxBuffer = srcBuffer[splitSize]; // [splitM, 1]
            const LocalTensor<float>& tmpBuffer = expMaxBuffer[splitM]; // [splitM, 64] + [splitM, 8]
            SoftMaxFlashV2M1CastIntrinsicsImpl(srcBuffer, srcLocalIn, splitSize);
            dstLocalOut = srcBuffer;
            expSumTensorOut = expSumTensorIn;
            maxTensorOut = maxTensorIn;
            srcLocalOut = srcBuffer;
            expMaxTensorOut = expMaxBuffer;
            inExpSumTensorOut = inExpSumTensorIn;
            inMaxTensorOut = inMaxTensorIn;
            workLocalOut = tmpBuffer;
        }
    }
    
    template <typename T1, typename T2>
    __aicore__ inline void SoftmaxFlashV2M1UpdateImplPostCast(const LocalTensor<T1>& dstLocalOut, const LocalTensor<T2>& expSumTensorOut,
        const LocalTensor<T2>& maxTensorOut, const LocalTensor<T1>& expMaxTensorOut,
        const LocalTensor<float>& dstLocalIn, const LocalTensor<float>& expSumTensorIn, const LocalTensor<float>& maxTensorIn,
        const LocalTensor<float>& expMaxTensorIn, const ReduceLastND& param)
    {
        uint32_t splitM = param.srcM;
        uint32_t splitSize = param.srcM * param.srcK;
        if constexpr (SupportType<T1, half>() && SupportType<T2, half>()) {
            SoftMaxFlashV2M1CastIntrinsicsImpl(dstLocalOut, dstLocalIn, splitSize);
            SoftMaxFlashV2M1CastIntrinsicsImpl(expSumTensorOut, expSumTensorIn, splitM);
            SoftMaxFlashV2M1CastIntrinsicsImpl(maxTensorOut, maxTensorIn, splitM);
            SoftMaxFlashV2M1CastIntrinsicsImpl(expMaxTensorOut, expMaxTensorIn, splitM);
        } else if constexpr (SupportType<T1, half>() && SupportType<T2, float>()) {
            SoftMaxFlashV2M1CastIntrinsicsImpl(dstLocalOut, dstLocalIn, splitSize);
            SoftMaxFlashV2M1CastIntrinsicsImpl(expMaxTensorOut, expMaxTensorIn, splitM);
        }
    }
    
    template <typename T1, typename T2>
    __aicore__ inline void SoftmaxFlashV2M1UpdateImpl(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& expSumTensor,
        const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& srcLocal, const LocalTensor<T1>& expMaxTensor,
        const LocalTensor<T2>& inExpSumTensor, const LocalTensor<T2>& inMaxTensor,
        const LocalTensor<float>& workLocal, const ReduceLastND& reduceParam, const SoftMaxTiling& tiling)
    {
        LocalTensor<float> dstLocalOut;
        LocalTensor<float> expSumTensorOut;
        LocalTensor<float> maxTensorOut;
        LocalTensor<float> srcLocalOut;
        LocalTensor<float> expMaxTensorOut;
        LocalTensor<float> inExpSumTensorOut;
        LocalTensor<float> inMaxTensorOut;
        LocalTensor<float> tmpBuffer;
    
        SoftmaxFlashV2M1UpdateImplPreCast<T1, T2>(dstLocalOut, expSumTensorOut, maxTensorOut, srcLocalOut,
            expMaxTensorOut, inExpSumTensorOut, inMaxTensorOut, tmpBuffer,
            dstLocal, expSumTensor, maxTensor, srcLocal, expMaxTensor, inExpSumTensor, inMaxTensor, workLocal, reduceParam);
    
        SoftmaxFlashV2M1UpdateImplProcess(dstLocalOut, expSumTensorOut, maxTensorOut, srcLocalOut, 
            expMaxTensorOut, inExpSumTensorOut, inMaxTensorOut, tmpBuffer, reduceParam, tiling);
    
        SoftmaxFlashV2M1UpdateImplPostCast<T1, T2>(dstLocal, expSumTensor, maxTensor, expMaxTensor,
            dstLocalOut, expSumTensorOut, maxTensorOut, expMaxTensorOut, reduceParam);
    }
    
    template <typename T1, typename T2, bool isUpdate = false, bool isBasicBlock = false>
    __aicore__ inline void SoftmaxFlashV2M1ImplProcess(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& expSumTensor,
        const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<T1>& expMaxTensor,
        const LocalTensor<T2>& inExpSumTensor, const LocalTensor<T2>& inMaxTensor, const LocalTensor<float>& workLocal,
        const ReduceLastND& reduceParam, const SoftMaxTiling& tiling)
    {
        if constexpr (isBasicBlock && !isUpdate) {
            SoftmaxFlashV2M1NoUpdateBasicBlock(dstTensor, expSumTensor, maxTensor, srcTensor, workLocal, tiling);
        } else if constexpr (isBasicBlock && isUpdate) {
            SoftmaxFlashV2M1BasicBlockImpl(dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor, workLocal, tiling);
        } else if constexpr (!isBasicBlock && !isUpdate) {
            SoftmaxFlashV2M1NoUpdateImpl<T1, T2>(dstTensor, expSumTensor, maxTensor, srcTensor, workLocal, reduceParam, tiling);
        } else if constexpr (!isBasicBlock && isUpdate) {
            SoftmaxFlashV2M1UpdateImpl<T1, T2>(dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor,
                workLocal, reduceParam, tiling);
        }
    }
    
    template <typename T1, typename T2, bool isUpdate = false, bool isBasicBlock = false>
    __aicore__ inline void SoftmaxFlashV2M1PostProcess(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& expSumTensor,
        const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<T1>& expMaxTensor,
        const LocalTensor<T2>& inExpSumTensor, const LocalTensor<T2>& inMaxTensor, const LocalTensor<float>& workLocal,
        const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
    {
        ReduceLastND reduceParam = { tiling.splitM, originalSrcShape.k, tiling.splitM, tiling.splitK, tiling.reduceM, tiling.reduceK};
        ReduceLastND tailParam = { tiling.tailM, originalSrcShape.k, tiling.tailM, tiling.splitK, tiling.tailM, tiling.reduceK};
    
        uint32_t offset1 = 0;
        uint32_t offset2 = 0;
        PipeBarrier<PIPE_V>();
        SetMaskCount();
        for (uint32_t i = 0; i < tiling.rangeM; i++) {
            SoftmaxFlashV2M1ImplProcess<T1, T2, isUpdate, isBasicBlock>(dstTensor[offset1], expSumTensor[offset2], maxTensor[offset2], srcTensor[offset1],
                expMaxTensor[offset2], inExpSumTensor[offset2], inMaxTensor[offset2], workLocal, reduceParam, tiling);
            offset1 += tiling.splitSize;
            offset2 += tiling.reduceM;
        }
    
        if constexpr (!isBasicBlock) {
            if (tiling.tailM != 0) {
                offset1 = tiling.rangeM * tiling.splitSize;
                offset2 = tiling.rangeM * tiling.reduceM;
                SoftmaxFlashV2M1ImplProcess<T1, T2, isUpdate, isBasicBlock>(dstTensor[offset1], expSumTensor[offset2], maxTensor[offset2], srcTensor[offset1],
                    expMaxTensor[offset2], inExpSumTensor[offset2], inMaxTensor[offset2], workLocal, tailParam, tiling);
            }
        }
        SetMaskNorm();
        ResetMask();
    }
}
#endif // IMPL_ACTIVATION_SOFTMAX_V220_SOFTMAX_FLASHV2_IMPL_H