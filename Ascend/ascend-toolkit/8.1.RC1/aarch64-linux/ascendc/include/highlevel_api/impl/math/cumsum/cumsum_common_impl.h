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
 * \file cumsum_common_impl.h
 * \brief
 */
#ifndef IMPL_MATH_CUMSUM_CUMSUM_COMMON_IMPL_H
#define IMPL_MATH_CUMSUM_CUMSUM_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "../../common/check.h"
#if __CCE_AICORE__ >= 200
#include "lib/math/cumsum_utils.h"

namespace AscendC {
struct CumSumInfo {
    uint32_t outter{0};
    uint32_t inner{0};  // 32-byte aligned
};

__aicore__ inline TransDataTo5HDParams ExtractTransDataParam(uint8_t repeatTimes, uint32_t inner, uint16_t alignOutter, 
    uint32_t oneBlockElementNum, uint16_t dstRepStride, uint32_t srcRepStride)
{
    repeatTimes = inner / oneBlockElementNum;
    if (repeatTimes > 1) {
        // For float date types, within a single repeated iteration, a (16, 8) matrix of float values will be
        // transposed into an (8, 16) layout.
        return TransDataTo5HDParams(false, false, repeatTimes, alignOutter, 1);
    } else {
        return TransDataTo5HDParams(false, false, repeatTimes, dstRepStride, srcRepStride);
    }
}

template <typename T>
__aicore__ inline void CumSumLastDim(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    LocalTensor<T> tempBuffer, const CumSumInfo &cumSumInfo)
{
    constexpr uint32_t oneBlockElementNum = ONE_BLK_SIZE / sizeof(T);
    uint16_t alignOutter =
        (cumSumInfo.outter + NCHW_CONV_ADDR_LIST_SIZE - 1) / NCHW_CONV_ADDR_LIST_SIZE * NCHW_CONV_ADDR_LIST_SIZE;
    uint64_t transDataTo5HDDstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t transDataTo5HDSrcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
    uint8_t repeatTimes = 1;
    uint16_t dstRepStride = 0;
    uint16_t srcRepStride = 0;
    // The larger one between the 'inner' and 'outter' is used as the repeat parameter of instruction,
    // while the smaller one serves as the loop count of the instruction.
    if (cumSumInfo.outter == alignOutter && alignOutter > cumSumInfo.inner) {
        repeatTimes = alignOutter / NCHW_CONV_ADDR_LIST_SIZE;
        if (repeatTimes > 1) {
            // For half date types, within a single repeated iteration, a (16, 16) matrix of half values will be
            // transposed into an (16, 16) layout.
            dstRepStride = 1;
            srcRepStride = cumSumInfo.inner;
        }
        TransDataTo5HDParams params(false, false, repeatTimes, dstRepStride, srcRepStride);
        for (int32_t i = 0; i < cumSumInfo.inner / oneBlockElementNum; i++) {
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                transDataTo5HDSrcLocalList[n] =
                    (uint64_t)srcTensor[i * oneBlockElementNum + n * cumSumInfo.inner].GetPhyAddr();
                transDataTo5HDDstLocalList[n] =
                    (uint64_t)tempBuffer[i * oneBlockElementNum * alignOutter + alignOutter * n].GetPhyAddr();
            }
            TransDataTo5HD<T>(transDataTo5HDDstLocalList, transDataTo5HDSrcLocalList, params);
        }
    } else {
        TransDataTo5HDParams params = ExtractTransDataParam(repeatTimes, cumSumInfo.inner, alignOutter,
            oneBlockElementNum, dstRepStride, srcRepStride);
        for (int32_t i = 0; i < alignOutter / NCHW_CONV_ADDR_LIST_SIZE; i++) {
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                transDataTo5HDSrcLocalList[n] = (uint64_t)srcTensor[((i * NCHW_CONV_ADDR_LIST_SIZE +
                    n % (cumSumInfo.outter - i * NCHW_CONV_ADDR_LIST_SIZE)) * cumSumInfo.inner)].GetPhyAddr();
                transDataTo5HDDstLocalList[n] =
                    (uint64_t)tempBuffer[i * NCHW_CONV_ADDR_LIST_SIZE + alignOutter * n].GetPhyAddr();
            }
            TransDataTo5HD<T>(transDataTo5HDDstLocalList, transDataTo5HDSrcLocalList, params);
        }
    }
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(alignOutter * cumSumInfo.inner);
    LocalTensor<float> floatTempBuffer = tempBuffer[alignOutter * cumSumInfo.inner].template ReinterpretCast<float>();
    Cast<float, T, false>(floatTempBuffer, tempBuffer, RoundMode::CAST_NONE, MASK_PLACEHOLDER,
        1, {1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();

    SetVectorMask<float>(0, alignOutter);
    const BinaryRepeatParams binaryParams;
    for (uint32_t row = 1; row < cumSumInfo.inner; ++row) {
        Add<float, false>(floatTempBuffer[row * alignOutter], floatTempBuffer[(row - 1) * alignOutter],
            floatTempBuffer[row * alignOutter], MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();
    }

    SetVectorMask<T, MaskMode::COUNTER>(alignOutter * cumSumInfo.inner);
    Cast<T, float, false>(tempBuffer, floatTempBuffer, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        {1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
    // After the first transpose, the inner axis of tempBuffer must be aligned with NCHW_CONV_ADDR_LIST_SIZE.
    auto tempBuffer2 = tempBuffer[alignOutter * cumSumInfo.inner];
    if (alignOutter > cumSumInfo.inner) {
        repeatTimes = alignOutter / oneBlockElementNum;
        if (repeatTimes > 1) {
            dstRepStride = cumSumInfo.inner;
            srcRepStride = 1;
        } else {
            dstRepStride = 0;
            srcRepStride = 0;
        }
        TransDataTo5HDParams paramsBack(false, false, repeatTimes, dstRepStride, srcRepStride);
        for (int32_t i = 0; i < cumSumInfo.inner / NCHW_CONV_ADDR_LIST_SIZE; i++) {
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                transDataTo5HDSrcLocalList[n] =
                    (uint64_t)tempBuffer[(i * NCHW_CONV_ADDR_LIST_SIZE + n) * alignOutter].GetPhyAddr();
                transDataTo5HDDstLocalList[n] =
                    (uint64_t)tempBuffer2[i * NCHW_CONV_ADDR_LIST_SIZE + n * cumSumInfo.inner].GetPhyAddr();
            }
            TransDataTo5HD<T>(transDataTo5HDDstLocalList, transDataTo5HDSrcLocalList, paramsBack);
        }
    } else {
        repeatTimes = cumSumInfo.inner / oneBlockElementNum;
        if (repeatTimes > 1) {
            srcRepStride = 1;
            dstRepStride = alignOutter;
        } else {
            dstRepStride = 0;
            srcRepStride = 0;
        }
        TransDataTo5HDParams paramsBack(false, false, repeatTimes, srcRepStride, dstRepStride);
        for (int32_t i = 0; i < alignOutter / NCHW_CONV_ADDR_LIST_SIZE; i++) {
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                transDataTo5HDSrcLocalList[n] =
                    (uint64_t)tempBuffer[i * NCHW_CONV_ADDR_LIST_SIZE + alignOutter * n].GetPhyAddr();
                transDataTo5HDDstLocalList[n] =
                    (uint64_t)tempBuffer2[(i * NCHW_CONV_ADDR_LIST_SIZE + n) * cumSumInfo.inner].GetPhyAddr();
            }
            TransDataTo5HD<T>(transDataTo5HDDstLocalList, transDataTo5HDSrcLocalList, paramsBack);
        }
    }
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<T>(0, cumSumInfo.outter * cumSumInfo.inner);
    Adds<T, false>(
        dstTensor, tempBuffer2, 0, MASK_PLACEHOLDER, 1, {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
}

template <>
__aicore__ inline void CumSumLastDim(const LocalTensor<float> &dstTensor, const LocalTensor<float> &srcTensor,
    LocalTensor<float> tempBuffer, const CumSumInfo &cumSumInfo)
{
    constexpr uint32_t oneBlockElementNum = ONE_BLK_SIZE / sizeof(float);
    uint8_t repeatTimes = 1;
    uint16_t dstRepStride = 0;
    uint16_t srcRepStride = 0;
    uint16_t alignOutter =
        (cumSumInfo.outter + NCHW_CONV_ADDR_LIST_SIZE - 1) / NCHW_CONV_ADDR_LIST_SIZE * NCHW_CONV_ADDR_LIST_SIZE;
    uint64_t transDataTo5HDDstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t transDataTo5HDSrcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
    // The larger one between the 'inner' and 'outter' is used as the repeat parameter of instruction,
    // while the smaller one serves as the loop count of the instruction.
    if (cumSumInfo.outter == alignOutter && alignOutter > cumSumInfo.inner) {
        repeatTimes = alignOutter / NCHW_CONV_ADDR_LIST_SIZE;
        if (repeatTimes > 1) {
            // For float date types, within a single repeated iteration, a (16, 8) matrix of float values will be
            // transposed into an (8, 16) layout.
            dstRepStride = 2;                     // 2 is for float transpose
            srcRepStride = cumSumInfo.inner * 2;  // 2 is for float transpose
        }
        TransDataTo5HDParams params(false, false, repeatTimes, dstRepStride, srcRepStride);
        for (int32_t i = 0; i < cumSumInfo.inner / oneBlockElementNum; i++) {
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                transDataTo5HDSrcLocalList[n] =
                    (uint64_t)srcTensor[i * oneBlockElementNum + n * cumSumInfo.inner].GetPhyAddr();
            }
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE / 2; n++) {  // 2 is for float transpose
                transDataTo5HDDstLocalList[n * 2] =
                    (uint64_t)tempBuffer[(i * oneBlockElementNum + n) * alignOutter].GetPhyAddr();
                transDataTo5HDDstLocalList[n * 2 + 1] =
                    (uint64_t)tempBuffer[(i * oneBlockElementNum + n) * alignOutter + oneBlockElementNum].GetPhyAddr();
            }
            TransDataTo5HD<float>(transDataTo5HDDstLocalList, transDataTo5HDSrcLocalList, params);
        }
    } else {
        TransDataTo5HDParams params = ExtractTransDataParam(repeatTimes, cumSumInfo.inner, alignOutter,
            oneBlockElementNum, dstRepStride, srcRepStride);
        for (int32_t i = 0; i < alignOutter / NCHW_CONV_ADDR_LIST_SIZE; i++) {
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE; n++) {
                transDataTo5HDSrcLocalList[n] = (uint64_t)srcTensor[((i * NCHW_CONV_ADDR_LIST_SIZE +
                    n % (cumSumInfo.outter - i * NCHW_CONV_ADDR_LIST_SIZE)) * cumSumInfo.inner)].GetPhyAddr();
            }
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE / 2; n++) {
                transDataTo5HDDstLocalList[n * 2] =
                    (uint64_t)tempBuffer[i * NCHW_CONV_ADDR_LIST_SIZE + n * alignOutter].GetPhyAddr();
                transDataTo5HDDstLocalList[n * 2 + 1] =
                    (uint64_t)tempBuffer[i * NCHW_CONV_ADDR_LIST_SIZE + n * alignOutter + oneBlockElementNum]
                        .GetPhyAddr();
            }
            TransDataTo5HD<float>(transDataTo5HDDstLocalList, transDataTo5HDSrcLocalList, params);
        }
    }
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<float>(0, alignOutter);
    const BinaryRepeatParams binaryParams;
    uint32_t addOffset = alignOutter;
    for (uint32_t row = 1; row < cumSumInfo.inner; ++row) {
        Add<float, false>(tempBuffer[addOffset],
            tempBuffer[addOffset - alignOutter],
            tempBuffer[addOffset],
            MASK_PLACEHOLDER,
            1,
            binaryParams);
        addOffset += alignOutter;
        PipeBarrier<PIPE_V>();
    }
    SetMaskNorm();
    ResetMask();

    auto tempBuffer2 = tempBuffer[alignOutter * cumSumInfo.inner];
    if (alignOutter > cumSumInfo.inner) {
        repeatTimes = alignOutter / NCHW_CONV_ADDR_LIST_SIZE;
        if (repeatTimes > 1) {
            // For float date types, within a single repeated iteration, a (16, 8) matrix of float values will be
            // transposed into an (8, 16) layout.
            dstRepStride = cumSumInfo.inner * 2;
            srcRepStride = 2;
        } else {
            dstRepStride = 0;
            srcRepStride = 0;
        }
        TransDataTo5HDParams paramsBack(false, false, repeatTimes, dstRepStride, srcRepStride);
        for (int32_t i = 0; i < cumSumInfo.inner / oneBlockElementNum; i++) {
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE / 2; n++) {
                transDataTo5HDSrcLocalList[n] =
                    (uint64_t)tempBuffer[i * oneBlockElementNum * alignOutter + n * alignOutter].GetPhyAddr();
                transDataTo5HDSrcLocalList[n + NCHW_CONV_ADDR_LIST_SIZE / 2] =
                    (uint64_t)tempBuffer[i * oneBlockElementNum * alignOutter + n * alignOutter + oneBlockElementNum]
                        .GetPhyAddr();
                transDataTo5HDDstLocalList[n * 2] =
                    (uint64_t)tempBuffer2[i * oneBlockElementNum + n * cumSumInfo.inner].GetPhyAddr();
                transDataTo5HDDstLocalList[n * 2 + 1] =
                    (uint64_t)tempBuffer2[i * oneBlockElementNum + (n + oneBlockElementNum) * cumSumInfo.inner]
                        .GetPhyAddr();
            }
            TransDataTo5HD<float>(transDataTo5HDDstLocalList, transDataTo5HDSrcLocalList, paramsBack);
        }

    } else {
        repeatTimes = cumSumInfo.inner / oneBlockElementNum;
        if (repeatTimes > 1) {
            // For float date types, within a single repeated iteration, a (16, 8) matrix of float values will be
            // transposed into an (8, 16) layout.
            dstRepStride = alignOutter;
            srcRepStride = 1;
        } else {
            dstRepStride = 0;
            srcRepStride = 0;
        }
        TransDataTo5HDParams paramsBack(false, false, repeatTimes, srcRepStride, dstRepStride);
        for (int32_t i = 0; i < alignOutter / NCHW_CONV_ADDR_LIST_SIZE; i++) {
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE / 2; n++) {
                transDataTo5HDSrcLocalList[n] =
                    (uint64_t)tempBuffer[i * NCHW_CONV_ADDR_LIST_SIZE + n * alignOutter].GetPhyAddr();
                transDataTo5HDSrcLocalList[n + NCHW_CONV_ADDR_LIST_SIZE / 2] =
                    (uint64_t)tempBuffer[i * NCHW_CONV_ADDR_LIST_SIZE + n * alignOutter + oneBlockElementNum]
                        .GetPhyAddr();
            }
            for (int32_t n = 0; n < NCHW_CONV_ADDR_LIST_SIZE / 2; n++) {
                transDataTo5HDDstLocalList[n * 2] =
                    (uint64_t)tempBuffer2[(i * NCHW_CONV_ADDR_LIST_SIZE + n) * cumSumInfo.inner].GetPhyAddr();
                transDataTo5HDDstLocalList[n * 2 + 1] =
                    (uint64_t)tempBuffer2[(i * NCHW_CONV_ADDR_LIST_SIZE + (n + NCHW_CONV_ADDR_LIST_SIZE / 2)) *
                                          cumSumInfo.inner]
                        .GetPhyAddr();
            }
            TransDataTo5HD<float>(transDataTo5HDDstLocalList, transDataTo5HDSrcLocalList, paramsBack);
        }
    }
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<float>(0, cumSumInfo.outter * cumSumInfo.inner);
    Adds<float, false>(
        dstTensor, tempBuffer2, 0, MASK_PLACEHOLDER, 1, {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
}

template <typename T>
__aicore__ inline void CumSumFirstDim(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    LocalTensor<uint8_t> &sharedTmpBuffer, const CumSumInfo &cumSumInfo)
{
    if constexpr (sizeof(T) == sizeof(half)) {
        const uint32_t minTmpBufferSize = cumSumInfo.outter * cumSumInfo.inner * sizeof(float);
        const uint32_t tmpBufferSize = sharedTmpBuffer.GetSize();
#if ASCENDC_CPU_DEBUG
        ASCENDC_ASSERT((tmpBufferSize >= minTmpBufferSize), {
            KERNEL_LOG( KERNEL_ERROR, "Insufficient temporary space, current operation is not enough, "
                "but only %u units are available, please check the host tiling.", tmpBufferSize);
        });      
#endif
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(cumSumInfo.outter * cumSumInfo.inner);
        LocalTensor<float> tmpBuffer = sharedTmpBuffer.ReinterpretCast<float>();
        Cast<float, T, false>(tmpBuffer, srcTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
            {1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();

        SetVectorMask<T>(0, cumSumInfo.inner);
        const BinaryRepeatParams binaryParams;
        for (uint32_t row = 1; row < cumSumInfo.outter; ++row) {
            Add<float, false>(tmpBuffer[row * cumSumInfo.inner], tmpBuffer[(row - 1) * cumSumInfo.inner],
                tmpBuffer[row * cumSumInfo.inner], MASK_PLACEHOLDER, 1, binaryParams);
            PipeBarrier<PIPE_V>();
        }

        SetVectorMask<T, MaskMode::COUNTER>(cumSumInfo.outter * cumSumInfo.inner);
        Cast<T, float, false>(dstTensor, tmpBuffer, RoundMode::CAST_NONE, MASK_PLACEHOLDER,
            1, {1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();

    } else {
        SetMaskCount();
        SetVectorMask<T>(0, cumSumInfo.inner);
        Adds<T, false>(
            dstTensor, srcTensor, 0, MASK_PLACEHOLDER, 1, {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams binaryParams;
        for (uint32_t row = 1; row < cumSumInfo.outter; ++row) {
            Add<T, false>(dstTensor[row * cumSumInfo.inner],
                dstTensor[(row - 1) * cumSumInfo.inner],
                srcTensor[row * cumSumInfo.inner],
                MASK_PLACEHOLDER,
                1,
                binaryParams);
            PipeBarrier<PIPE_V>();
        }
        SetMaskNorm();
        ResetMask();
    }
}

template <typename T, const CumSumConfig &config>
__aicore__ inline void CumSumValid(LocalTensor<T> &dstTensor, LocalTensor<T> &lastRowTensor,
    const LocalTensor<T> &srcTensor, LocalTensor<uint8_t> &sharedTmpBuffer, const CumSumInfo &cumSumInfo)
{
    CheckTensorPosition(dstTensor, "dstLocal", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(lastRowTensor, "lastRowTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcLocal", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
#if ASCENDC_CPU_DEBUG
    bool ans = cumSumInfo.inner > 0;
    ASCENDC_ASSERT(ans, {
        KERNEL_LOG(KERNEL_ERROR, "The cumSumInfo.inner parameter cannot be %u, should be bigger than 0.",
            cumSumInfo.inner); });
    ans = (cumSumInfo.inner * sizeof(T) % ONE_BLK_SIZE == 0);
    ASCENDC_ASSERT(ans, {
        KERNEL_LOG(KERNEL_ERROR, "The result of cumSumInfo.inner * sizeof(T) cannot be %u, "
            "should be an integer multiple of 32.", cumSumInfo.inner * sizeof(T));
    });
    ans = srcTensor.GetSize() >= (cumSumInfo.inner * cumSumInfo.outter);
    ASCENDC_ASSERT(ans, {
        KERNEL_LOG(KERNEL_ERROR, "The result of cumSumInfo.inner * cumSumInfo.outter "
            "cannot be  %u, should not be larger than %s size %u.",
            cumSumInfo.inner * cumSumInfo.outter, "srcLocal", srcTensor.GetSize());
    });
    ans = dstTensor.GetSize() >= (cumSumInfo.inner * cumSumInfo.outter);
    ASCENDC_ASSERT(ans, {
        KERNEL_LOG(KERNEL_ERROR, "The result of cumSumInfo.inner * cumSumInfo.outter "
            "cannot be  %u, should not be larger than %s size %u.",
            cumSumInfo.inner * cumSumInfo.outter, "dstLocal", dstTensor.GetSize());
    });
    if (config.outputLastRow) {
        ans = lastRowTensor.GetSize() >= cumSumInfo.inner;
        ASCENDC_ASSERT(ans, { KERNEL_LOG(KERNEL_ERROR,
            "The cumSumInfo.inner parameter cannot be %u, should not be bigger than lastRowTensor size %u "
            "when config.outputLastRow is true.", cumSumInfo.inner, lastRowTensor.GetSize()); });
    }
#endif
}

template <typename T, const CumSumConfig &config>
__aicore__ inline void CumSumImpl(LocalTensor<T> &dstTensor, LocalTensor<T> &lastRowTensor,
    const LocalTensor<T> &srcTensor, LocalTensor<uint8_t> &sharedTmpBuffer, const CumSumInfo &cumSumInfo)
{
    if ASCEND_IS_AIC {
        return;
    }
    CumSumValid<T, config>(dstTensor, lastRowTensor, srcTensor, sharedTmpBuffer, cumSumInfo);

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value), {
        KERNEL_LOG( KERNEL_ERROR, "Failed to check the data types, current api support data types are half/float."); });
    if constexpr (config.isLastAxis) {
        uint32_t minCastTempBufferSize = 0;
        if constexpr (sizeof(T) == sizeof(half)) {
            minCastTempBufferSize = cumSumInfo.inner * NCHW_CONV_ADDR_LIST_SIZE * sizeof(half);
        }
        // Both transpose require tempBuffer
        const uint32_t minTmpBufferSize = minCastTempBufferSize +
                                          NCHW_CONV_ADDR_LIST_SIZE * cumSumInfo.inner * sizeof(T) * 2;
        const uint32_t tmpBufferSize = sharedTmpBuffer.GetSize();
#if ASCENDC_CPU_DEBUG
        ASCENDC_ASSERT((tmpBufferSize >= minTmpBufferSize), {
            KERNEL_LOG( KERNEL_ERROR, "Insufficient temporary space, current operation is not enough, "
                "but only %u units are available, please check the host tiling.", tmpBufferSize);
        });
#endif
        // outter serves as the loop count, process at least 16 rows of data at once.
        const uint32_t oneRepeateSize = tmpBufferSize / minTmpBufferSize * NCHW_CONV_ADDR_LIST_SIZE;
        const uint32_t rangeM = cumSumInfo.outter / oneRepeateSize;
        const uint32_t tailM = cumSumInfo.outter - oneRepeateSize * rangeM;
        uint32_t dstLocalOffset = 0;
        uint32_t srcLocalOffset = 0;
        LocalTensor<T> tmpBuffer = sharedTmpBuffer.ReinterpretCast<T>();
        for (uint32_t i = 0; i < rangeM; i++) {
            CumSumLastDim<T>(
                dstTensor[dstLocalOffset], srcTensor[srcLocalOffset], tmpBuffer, {oneRepeateSize, cumSumInfo.inner});
            dstLocalOffset += cumSumInfo.inner * oneRepeateSize;
            srcLocalOffset += cumSumInfo.inner * oneRepeateSize;
        }

        if (tailM != 0) {
            CumSumLastDim<T>(
                dstTensor[dstLocalOffset], srcTensor[srcLocalOffset], tmpBuffer, {tailM, cumSumInfo.inner});
        }
    } else {
        CumSumFirstDim<T>(dstTensor, srcTensor, sharedTmpBuffer, cumSumInfo);
    }

    if constexpr (config.outputLastRow) {
        SetMaskCount();
        SetVectorMask<T>(0, cumSumInfo.inner);
        Adds<T, false>(lastRowTensor, dstTensor[(cumSumInfo.outter - 1) * cumSumInfo.inner], 0, MASK_PLACEHOLDER,
            1, {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
    }
}
}  // namespace AscendC

#endif

#endif  // IMPL_MATH_CUMSUM_CUMSUM_COMMON_IMPL_H