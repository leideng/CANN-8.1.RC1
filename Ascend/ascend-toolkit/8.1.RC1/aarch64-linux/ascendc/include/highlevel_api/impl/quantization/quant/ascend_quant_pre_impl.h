/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ascend_quant_pre_impl.h
 * \brief
 */

#ifndef IMPL_QUANTIZATION_QUANT_ASCEND_QUANT_PRE_IMPL_H
#define IMPL_QUANTIZATION_QUANT_ASCEND_QUANT_PRE_IMPL_H
#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/quantization/ascend_quant_utils.h"

namespace AscendC {

template <typename T, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void IsQuantValid(const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    ASCENDC_ASSERT((calCount <= srcTensor.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should not larger than srcTensor size %u", calCount,
            srcTensor.GetSize());
    });
    ASCENDC_ASSERT((config.calcCount <= srcTensor.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR, "config.calcCount is %u, which should not larger than srcTensor size %u",
            config.calcCount, srcTensor.GetSize());
    });
    ASCENDC_ASSERT((config.workLocalSize <= sharedTmpBuffer.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR, "workLocalSize is %u, which should not larger than sharedTmpBuffer size %u",
            config.workLocalSize, sharedTmpBuffer.GetSize());
    });
}

template <typename T, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void IsQuantConfigValid(const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor)
{
    ASCENDC_ASSERT((config.calcCount <= srcTensor.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR, "config.calCount is %u, which should not larger than srcTensor size %u.",
            config.calcCount, srcTensor.GetSize());
    });
    ASCENDC_ASSERT((config.scaleCount <= scaleTensor.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR, "config.scaleCount is %u, which should not larger than scaleTensor size %u.",
            config.scaleCount, scaleTensor.GetSize());
    });
    ASCENDC_ASSERT((config.scaleCount == 0 || config.calcCount % config.scaleCount == 0), {
        KERNEL_LOG(KERNEL_ERROR,
            "config.calcCount is %u, which should be integral multiple of config.scaleCount %u.",
            config.calcCount, config.scaleCount);
    });
    ASCENDC_ASSERT((config.scaleCount % ONE_BLK_SIZE == 0), {
        KERNEL_LOG(KERNEL_ERROR,
            "config.scaleCount is %u, which should be integral multiple of 32.",
            config.scaleCount);
    });
    ASCENDC_ASSERT((config.workLocalSize <= sharedTmpBuffer.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR, "workLocalSize is %u, which should not larger than sharedTmpBuffer size %u",
            config.workLocalSize, sharedTmpBuffer.GetSize());
    });
}

template <typename T, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void IsQuantConfigValid(const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor,
    const LocalTensor<T>& offsetTensor)
{
    IsQuantConfigValid<T, config>(srcTensor, sharedTmpBuffer, scaleTensor);
    ASCENDC_ASSERT((config.offsetCount <= offsetTensor.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR,
            "config.offsetCount is %u, which should not larger than offsetTensor size %u.",
            config.offsetCount, offsetTensor.GetSize());
    });
    ASCENDC_ASSERT((config.scaleCount == config.offsetCount), {
        KERNEL_LOG(KERNEL_ERROR,
            "config.scaleCount is %u, which should be equal to config.offsetCount %u.",
            config.scaleCount, config.offsetCount);
    });
}

// per channel intrinsics
__aicore__ inline void AscendQuantPerChannelIntrinsicsImpl(const LocalTensor<int8_t>& dstTensor,
    const LocalTensor<float>& srcTensor, const LocalTensor<half>& stackTensor, const LocalTensor<half>& scaleTensor,
    const half offset)
{
    BinaryRepeatParams binaryParam;
    UnaryRepeatParams f162s8Param;
    UnaryRepeatParams unaryParams;
    f162s8Param.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    Cast<half, float, false>(stackTensor, srcTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, f162s8Param);
    PipeBarrier<PIPE_V>();
    Mul<half, false>(stackTensor, stackTensor, scaleTensor, MASK_PLACEHOLDER, 1, binaryParam);
    PipeBarrier<PIPE_V>();
    Adds<half, false>(stackTensor, stackTensor, offset, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Cast<int8_t, half, false>(dstTensor, stackTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, f162s8Param);
    PipeBarrier<PIPE_V>();
}
__aicore__ inline void AscendQuantPerChannelIntrinsicsImpl(const LocalTensor<int8_t>& dstTensor,
    const LocalTensor<half>& srcTensor, const LocalTensor<half>& stackTensor, const LocalTensor<half>& scaleTensor,
    const half offset)
{
    BinaryRepeatParams binaryParam;
    UnaryRepeatParams f162s8Param;
    UnaryRepeatParams unaryParams;
    f162s8Param.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    Mul<half, false>(stackTensor, srcTensor, scaleTensor, MASK_PLACEHOLDER, 1, binaryParam);
    PipeBarrier<PIPE_V>();
    Adds<half, false>(stackTensor, stackTensor, static_cast<half>(offset), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Cast<int8_t, half, false>(dstTensor, stackTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, f162s8Param);
    PipeBarrier<PIPE_V>();
}
__aicore__ inline void AscendQuantPerChannelIntrinsicsImpl(const LocalTensor<int8_t>& dstTensor,
    const LocalTensor<float>& srcTensor, const LocalTensor<half>& stackTensor, const LocalTensor<half>& scaleTensor,
    const LocalTensor<half>& offsetTensor)
{
    BinaryRepeatParams binaryParam;
    UnaryRepeatParams f162s8Param;
    UnaryRepeatParams unaryParams;
    f162s8Param.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    Cast<half, float, false>(stackTensor, srcTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, f162s8Param);
    PipeBarrier<PIPE_V>();
    Mul<half, false>(stackTensor, stackTensor, scaleTensor, MASK_PLACEHOLDER, 1, binaryParam);
    PipeBarrier<PIPE_V>();
    Add<half, false>(stackTensor, stackTensor, offsetTensor, MASK_PLACEHOLDER, 1, binaryParam);
    PipeBarrier<PIPE_V>();
    Cast<int8_t, half, false>(dstTensor, stackTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, f162s8Param);
    PipeBarrier<PIPE_V>();
}
__aicore__ inline void AscendQuantPerChannelIntrinsicsImpl(const LocalTensor<int8_t>& dstTensor,
    const LocalTensor<half>& srcTensor, const LocalTensor<half>& stackTensor, const LocalTensor<half>& scaleTensor,
    const LocalTensor<half>& offsetTensor)
{
    BinaryRepeatParams binaryParam;
    UnaryRepeatParams f162s8Param;
    UnaryRepeatParams unaryParams;
    f162s8Param.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    Mul<half, false>(stackTensor, srcTensor, scaleTensor, MASK_PLACEHOLDER, 1, binaryParam);
    PipeBarrier<PIPE_V>();
    Add<half, false>(stackTensor, stackTensor, offsetTensor, MASK_PLACEHOLDER, 1, binaryParam);
    PipeBarrier<PIPE_V>();
    Cast<int8_t, half, false>(dstTensor, stackTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, f162s8Param);
    PipeBarrier<PIPE_V>();
}

// per channel impl
template <typename T, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuantPerChannelImpl(const LocalTensor<int8_t>& dstTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const LocalTensor<half>& scaleTensor, const LocalTensor<half>& offsetTensor, const uint32_t calCount)
{
    LocalTensor<half> tmpBuffer = sharedTmpBuffer.ReinterpretCast<half>();
    if constexpr(config.workLocalSize != 0 && config.scaleCount != 0) {
        constexpr uint32_t splitSize = config.workLocalSize / sizeof(half) / ONE_BLK_SIZE * ONE_BLK_SIZE;
        ASCENDC_ASSERT((splitSize > 0), {
            KERNEL_LOG(KERNEL_ERROR, "splitSize should not be zero.");
        });
        constexpr uint32_t loopCount = config.scaleCount / splitSize;
        constexpr uint32_t calcTail = config.scaleCount % splitSize;
        SetVectorMask<T, MaskMode::COUNTER>(0, splitSize);
        for (uint32_t i = 0; i < loopCount; ++i) {
            AscendQuantPerChannelIntrinsicsImpl(dstTensor[i * splitSize], srcTensor[i * splitSize],
                tmpBuffer, scaleTensor[i * splitSize], offsetTensor[i * splitSize]);
        }
        if constexpr(calcTail > 0) {
            SetVectorMask<T, MaskMode::COUNTER>(0, calcTail);
            AscendQuantPerChannelIntrinsicsImpl(dstTensor[loopCount * splitSize], srcTensor[loopCount * splitSize],
                tmpBuffer, scaleTensor[loopCount * splitSize], offsetTensor[loopCount * splitSize]);
        }
        return;
    }

    uint32_t splitSize = sharedTmpBuffer.GetSize() / sizeof(half) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    ASCENDC_ASSERT((splitSize > 0), {
        KERNEL_LOG(KERNEL_ERROR, "splitSize should not be zero.");
    });
    uint32_t loopCount = calCount / splitSize;
    uint32_t calcTail = calCount % splitSize;
    SetVectorMask<T, MaskMode::COUNTER>(0, splitSize);
    for (uint32_t i = 0; i < loopCount; ++i) {
        AscendQuantPerChannelIntrinsicsImpl(dstTensor[i * splitSize], srcTensor[i * splitSize],
            sharedTmpBuffer.ReinterpretCast<half>(), scaleTensor[i * splitSize],
            offsetTensor[i * splitSize]);
    }
    if (calcTail > 0) {
        SetVectorMask<T, MaskMode::COUNTER>(0, calcTail);
        AscendQuantPerChannelIntrinsicsImpl(dstTensor[loopCount * splitSize], srcTensor[loopCount * splitSize],
            sharedTmpBuffer.ReinterpretCast<half>(), scaleTensor[loopCount * splitSize],
            offsetTensor[loopCount * splitSize]);
    }
}
template <typename T, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuantPerChannelImpl(const LocalTensor<int8_t>& dstTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const LocalTensor<half>& scaleTensor, const half offset, const uint32_t calCount)
{
    LocalTensor<half> tmpBuffer = sharedTmpBuffer.ReinterpretCast<half>();
    if constexpr(config.workLocalSize != 0 && config.scaleCount != 0) {
        constexpr uint32_t splitSize = config.workLocalSize / sizeof(half) / ONE_BLK_SIZE * ONE_BLK_SIZE;
        ASCENDC_ASSERT((splitSize > 0), { KERNEL_LOG(KERNEL_ERROR, "splitSize should not be zero."); });
        constexpr uint32_t calcTail = config.scaleCount % splitSize;
        constexpr uint32_t loopCount = config.scaleCount / splitSize;
        SetVectorMask<T, MaskMode::COUNTER>(0, splitSize);

        for (uint32_t i = 0; i < loopCount; ++i) {
            AscendQuantPerChannelIntrinsicsImpl(dstTensor[i * splitSize], srcTensor[i * splitSize], tmpBuffer,
                scaleTensor[i * splitSize], offset);
        }
        if constexpr(calcTail > 0) {
            SetVectorMask<T, MaskMode::COUNTER>(0, calcTail);
            AscendQuantPerChannelIntrinsicsImpl(dstTensor[loopCount * splitSize],
                srcTensor[loopCount * splitSize], tmpBuffer, scaleTensor[loopCount * splitSize], offset);
        }
        return;
    }

    uint32_t splitSize = sharedTmpBuffer.GetSize() / sizeof(half) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    ASCENDC_ASSERT((splitSize > 0), { KERNEL_LOG(KERNEL_ERROR, "splitSize should not be zero."); });
    SetVectorMask<T, MaskMode::COUNTER>(0, splitSize);

    uint32_t calcTail = calCount % splitSize;
    uint32_t loopCount = calCount / splitSize;
    for (uint32_t i = 0; i < loopCount; ++i) {
        AscendQuantPerChannelIntrinsicsImpl(dstTensor[i * splitSize], srcTensor[i * splitSize],
            sharedTmpBuffer.ReinterpretCast<half>(), scaleTensor[i * splitSize], offset);
    }
    if (calcTail > 0) {
        SetVectorMask<T, MaskMode::COUNTER>(0, calcTail);
        AscendQuantPerChannelIntrinsicsImpl(dstTensor[loopCount * splitSize], srcTensor[loopCount * splitSize],
            sharedTmpBuffer.ReinterpretCast<half>(), scaleTensor[loopCount * splitSize], offset);
    }
}

template <typename T, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuantImplStatic(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor,
    const T offset)
{
    if constexpr(config.scaleCount != 0 && config.calcCount != 0) {
        constexpr uint32_t N = config.calcCount / config.scaleCount;
        if constexpr (IsSameType<T, float>::value) {
            LocalTensor<half> halfScaleTensor =  scaleTensor.template ReinterpretCast<half>();
            UnaryRepeatParams f162s8Param;
            f162s8Param.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
            SetVectorMask<half, MaskMode::COUNTER>(0, config.scaleCount);
            Cast<half, float, false>(halfScaleTensor, scaleTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
                f162s8Param);
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < N; ++i) {
                AscendQuantPerChannelImpl<T, config>(dstTensor[i * config.scaleCount], srcTensor[i * config.scaleCount],
                    sharedTmpBuffer, halfScaleTensor, static_cast<half>(offset), config.scaleCount);
            }
        } else {
            for (uint32_t i = 0; i < N; ++i) {
                AscendQuantPerChannelImpl<T, config>(dstTensor[i * config.scaleCount], srcTensor[i * config.scaleCount],
                    sharedTmpBuffer, scaleTensor, static_cast<half>(offset), config.scaleCount);
            }
        }
    }
}

template <typename T, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuantImplStatic(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor,
    const LocalTensor<T>& offsetTensor)
{
    if constexpr(config.scaleCount != 0 && config.calcCount != 0) {
        constexpr uint32_t N = config.calcCount / config.scaleCount;
        if constexpr (IsSameType<T, float>::value) {
            SetVectorMask<half, MaskMode::COUNTER>(0, config.scaleCount);
            LocalTensor<half> halfScaleTensor =  scaleTensor.template ReinterpretCast<half>();
            UnaryRepeatParams f162s8Param;
            f162s8Param.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
            Cast<half, float, false>(halfScaleTensor, scaleTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
                f162s8Param);
            LocalTensor<half> halfOffsetTensor =  offsetTensor.template ReinterpretCast<half>();
            Cast<half, float, false>(halfOffsetTensor, offsetTensor, RoundMode::CAST_NONE,
                MASK_PLACEHOLDER, 1, f162s8Param);

            for (uint32_t i = 0; i < N; ++i) {
                AscendQuantPerChannelImpl<T, config>(dstTensor[i * config.scaleCount], srcTensor[i * config.scaleCount],
                    sharedTmpBuffer, halfScaleTensor, halfOffsetTensor, config.scaleCount);
            }
        } else {
            for (uint32_t i = 0; i < N; ++i) {
                AscendQuantPerChannelImpl<T, config>(dstTensor[i * config.scaleCount], srcTensor[i * config.scaleCount],
                    sharedTmpBuffer, scaleTensor, offsetTensor, config.scaleCount);
            }
        }
    }
}

#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220
// per tensor intrinsics
__aicore__ inline void AscendQuantIntrinsicsImpl(const LocalTensor<int8_t>& dstTensor,
    const LocalTensor<half>& srcTensor, const LocalTensor<half>& stackBuffer, half scale, half offset)
{
    UnaryRepeatParams unaryParams;
    UnaryRepeatParams f162s8Params;
    f162s8Params.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    Muls<half, false>(stackBuffer, srcTensor, scale, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Adds<half, false>(stackBuffer, stackBuffer, offset, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Cast<int8_t, half, false>(dstTensor, stackBuffer, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, f162s8Params);
    PipeBarrier<PIPE_V>();
}
__aicore__ inline void AscendQuantIntrinsicsImpl(const LocalTensor<int8_t>& dstTensor,
    const LocalTensor<float>& srcTensor, const LocalTensor<half>& stackBuffer, half scale, half offset)
{
    UnaryRepeatParams unaryParams;
    UnaryRepeatParams f162s8Params;
    f162s8Params.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    Cast<half, float, false>(stackBuffer, srcTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, f162s8Params);
    PipeBarrier<PIPE_V>();
    Muls<half, false>(stackBuffer, stackBuffer, scale, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Adds<half, false>(stackBuffer, stackBuffer, offset, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Cast<int8_t, half, false>(dstTensor, stackBuffer, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, f162s8Params);
    PipeBarrier<PIPE_V>();
}

// api impl
template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuantCalc(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset, const uint32_t calCount)
{
    IsQuantValid<T, config>(srcTensor, sharedTmpBuffer, calCount);

    SetMaskCount();
    LocalTensor<half> tmpBuffer = sharedTmpBuffer.ReinterpretCast<half>();
    if constexpr(config.workLocalSize != 0 && config.calcCount != 0) {
        constexpr uint32_t splitSize = config.workLocalSize / sizeof(half) / ONE_BLK_SIZE * ONE_BLK_SIZE;
        ASCENDC_ASSERT((splitSize != 0), {
            KERNEL_LOG(KERNEL_ERROR, "splitSize should not be 0!");
        });
        constexpr uint32_t loopCount = config.calcCount / splitSize;
        SetVectorMask<T, MaskMode::COUNTER>(0, splitSize);
        for (uint32_t i = 0; i < loopCount; ++i) {
            AscendQuantIntrinsicsImpl(dstTensor[splitSize * i], srcTensor[splitSize * i],
                tmpBuffer, static_cast<half>(scale), static_cast<half>(offset));
        }
        if constexpr(config.calcCount % splitSize > 0) {
            SetVectorMask<T, MaskMode::COUNTER>(0, config.calcCount % splitSize);
            AscendQuantIntrinsicsImpl(dstTensor[splitSize * loopCount], srcTensor[splitSize * loopCount],
                tmpBuffer, static_cast<half>(scale), static_cast<half>(offset));
        }
    } else {
        uint32_t splitSize = sharedTmpBuffer.GetSize() / sizeof(half) / ONE_BLK_SIZE * ONE_BLK_SIZE;
        ASCENDC_ASSERT((splitSize != 0), {
            KERNEL_LOG(KERNEL_ERROR, "splitSize should not be 0!");
        });
        uint32_t loopCount = calCount / splitSize;
        SetVectorMask<T, MaskMode::COUNTER>(0, splitSize);
        for (uint32_t i = 0; i < loopCount; ++i) {
            AscendQuantIntrinsicsImpl(dstTensor[splitSize * i], srcTensor[splitSize * i],
                sharedTmpBuffer.ReinterpretCast<half>(), static_cast<half>(scale), static_cast<half>(offset));
        }
        if (calCount % splitSize > 0) {
            SetVectorMask<T, MaskMode::COUNTER>(0, calCount % splitSize);
            AscendQuantIntrinsicsImpl(dstTensor[splitSize * loopCount], srcTensor[splitSize * loopCount],
                sharedTmpBuffer.ReinterpretCast<half>(),
                static_cast<half>(scale), static_cast<half>(offset));
        }
    }

    SetMaskNorm();
    ResetMask();
}

template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuantCalc(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor,
    const T offset, const uint32_t scaleCount, const uint32_t calCount)
{
    IsQuantParamValid(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offset, scaleCount, calCount);
    IsQuantConfigValid<T, config>(srcTensor, sharedTmpBuffer, scaleTensor);
    SetMaskCount();

    if constexpr(config.scaleCount != 0 && config.calcCount != 0) {
        AscendQuantImplStatic<T, config>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offset);
    } else {
        uint32_t N = calCount / scaleCount;
        if constexpr (IsSameType<T, float>::value) {
            // source vector of scale is reused
            LocalTensor<half> halfScaleTensor =  scaleTensor.template ReinterpretCast<half>();
            UnaryRepeatParams f162s8Param;
            f162s8Param.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
            SetVectorMask<half, MaskMode::COUNTER>(0, scaleCount);
            Cast<half, float, false>(halfScaleTensor, scaleTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
                f162s8Param);
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < N; ++i) {
                AscendQuantPerChannelImpl<T, config>(dstTensor[i * scaleCount], srcTensor[i * scaleCount],
                    sharedTmpBuffer, halfScaleTensor, static_cast<half>(offset), scaleCount);
            }
        } else {
            for (uint32_t i = 0; i < N; ++i) {
                AscendQuantPerChannelImpl<T, config>(dstTensor[i * scaleCount], srcTensor[i * scaleCount],
                    sharedTmpBuffer, scaleTensor, static_cast<half>(offset), scaleCount);
            }
        }
    }

    SetMaskNorm();
    ResetMask();
}

template <typename T, bool isReuseSource = false, const AscendQuantConfig& config = ASCEND_QUANT_DEFAULT_CFG>
__aicore__ inline void AscendQuantCalc(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor,
    const LocalTensor<T>& offsetTensor, const uint32_t scaleCount, const uint32_t offsetCount,
    const uint32_t calCount)
{
    IsQuantParamValid(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor,
        scaleCount, offsetCount, calCount);
    IsQuantConfigValid<T, config>(srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor);
    SetMaskCount();

    if constexpr(config.scaleCount != 0 && config.calcCount != 0) {
        AscendQuantImplStatic<T, config>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor);
    } else {
        uint32_t N = calCount / scaleCount;
        if constexpr (IsSameType<T, float>::value) {
            SetVectorMask<half, MaskMode::COUNTER>(0, scaleCount);
            UnaryRepeatParams f162s8Param;
            f162s8Param.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
            // source vector of scale is reused
            LocalTensor<half> halfScaleTensor =  scaleTensor.template ReinterpretCast<half>();
            Cast<half, float, false>(halfScaleTensor, scaleTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
                f162s8Param);
            // source vector of offset is reused
            LocalTensor<half> halfOffsetTensor =  offsetTensor.template ReinterpretCast<half>();
            Cast<half, float, false>(halfOffsetTensor, offsetTensor, RoundMode::CAST_NONE,
                MASK_PLACEHOLDER, 1, f162s8Param);

            for (uint32_t i = 0; i < N; ++i) {
                AscendQuantPerChannelImpl<T, config>(dstTensor[i * scaleCount], srcTensor[i * scaleCount],
                    sharedTmpBuffer, halfScaleTensor, halfOffsetTensor, scaleCount);
            }
        } else {
            for (uint32_t i = 0; i < N; ++i) {
                AscendQuantPerChannelImpl<T, config>(dstTensor[i * scaleCount], srcTensor[i * scaleCount],
                    sharedTmpBuffer, scaleTensor, offsetTensor, scaleCount);
            }
        }
    }

    SetMaskNorm();
    ResetMask();
}
#endif
}
#endif // IMPL_QUANTIZATION_QUANT_ASCEND_QUANT_PRE_IMPL_H