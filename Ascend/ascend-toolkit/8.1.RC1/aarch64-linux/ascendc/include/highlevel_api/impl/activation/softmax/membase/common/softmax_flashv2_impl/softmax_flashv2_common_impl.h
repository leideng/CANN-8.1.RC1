/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_flashv2_common_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_COMMON_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_COMMON_IMPL_H

namespace AscendC {

template <typename T1, typename T2, bool isUpdate = false, bool isBasicBlock = false>
__aicore__ inline void SoftMaxFlashV2NZImpl(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<T1>& expMaxTensor,
    const LocalTensor<T2>& inSumTensor, const LocalTensor<T2>& inMaxTensor, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    if constexpr (!isUpdate) {
        SoftmaxFlashV2NZNoUpdateImpl<T1, T2, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, workLocal,
            originalSrcShape, tiling);
    } else {
        SoftmaxFlashV2NZUpdateImpl<T1, T2, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor,
            inSumTensor, inMaxTensor, workLocal, originalSrcShape, tiling);
    }
}

template <typename T1, typename T2, bool isUpdate = false, bool isBasicBlock = false>
__aicore__ inline void SoftMaxFlashV2NZImpl(const LocalTensor<half>& dstTensor, const LocalTensor<float>& sumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor, const LocalTensor<half>& expMaxTensor,
    const LocalTensor<float>& inSumTensor, const LocalTensor<float>& inMaxTensor, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    if constexpr (!isUpdate) {
        SoftmaxFlashV2NZNoUpdateImpl<T1, T2, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, workLocal,
            originalSrcShape, tiling);
    } else {
        SoftmaxFlashV2NZUpdateImpl<T1, T2, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor,
            inSumTensor, inMaxTensor, workLocal, originalSrcShape, tiling);
    }
}

template <typename T1, typename T2, bool isBasicBlock = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftmaxFlashV2NoUpdate(const LocalTensor<T1>& dst, const LocalTensor<T1>& expSumTensor,
    const LocalTensor<T1>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    if constexpr (config.oriSrcM == 0 || config.oriSrcK == 0) {
        if constexpr (isBasicBlock) {
            SoftmaxFlashV2NoUpdateBasicBlock(dst, expSumTensor, maxTensor, src, workLocal, tiling);
        } else {
            ReduceLastND reduceParam = { tiling.splitM, originalSrcShape.k, tiling.splitM,
                tiling.splitK, tiling.reduceM,     tiling.reduceK };
            SoftmaxFlashV2NoUpdateExtImpl<T1>(dst, expSumTensor, maxTensor, src, workLocal, originalSrcShape, tiling,
                reduceParam);
        }
    } else {
        constexpr uint32_t basicBlockMaxK = 2048;
        constexpr bool localIsBasicBlock =  config.oriSrcK % FLOAT_REPEAT_SIZE == 0 &&
            config.oriSrcK < basicBlockMaxK && config.oriSrcM % FLOAT_NUM_PER_BLK == 0;
        if constexpr (localIsBasicBlock) {
            SoftmaxFlashV2NoUpdateBasicBlock(dst, expSumTensor, maxTensor, src, workLocal, tiling);
        } else {
            uint32_t splitK = 0;
            ReduceLastND reduceParam;
            if constexpr (config.oriSrcK % FLOAT_NUM_PER_BLK == 0) {
                splitK = config.oriSrcK;
            } else {
                splitK = AlignUp(config.oriSrcK, FLOAT_NUM_PER_BLK);
            }
            if constexpr (SupportType<T1, half>()) {
                reduceParam = { tiling.splitM, config.oriSrcK, tiling.splitM, splitK, tiling.reduceM,
                    DEFAULT_REPEAT_STRIDE * HALF_FACTOR };
            } else if constexpr (SupportType<T1, float>()) {
                reduceParam = { tiling.splitM, config.oriSrcK, tiling.splitM, splitK, tiling.reduceM,
                    DEFAULT_REPEAT_STRIDE };
            }
            SoftmaxFlashV2NoUpdateExtImpl<T1>(dst, expSumTensor, maxTensor, src, workLocal, originalSrcShape, tiling,
                reduceParam);
        }
    }
}

template <typename T1, typename T2, bool isBasicBlock = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftmaxFlashV2NoUpdate(const LocalTensor<half>& dst, const LocalTensor<float>& expSumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    if constexpr (config.oriSrcM == 0 || config.oriSrcK == 0) {
        if constexpr (isBasicBlock) {
            SoftmaxFlashV2NoUpdateBasicBlock(dst, expSumTensor, maxTensor, src, workLocal, tiling);
        } else {
            ReduceLastND reduceParam = { tiling.splitM, originalSrcShape.k, tiling.splitM,
                tiling.splitK, tiling.reduceM,     tiling.reduceK };
            SoftmaxFlashV2NoUpdateExtImpl(dst, expSumTensor, maxTensor, src, workLocal, originalSrcShape, tiling,
                reduceParam);
        }
    } else {
        constexpr uint32_t basicBlockMaxK = 2048;
        constexpr bool localIsBasicBlock =  config.oriSrcK % FLOAT_REPEAT_SIZE == 0 &&
            config.oriSrcK < basicBlockMaxK && config.oriSrcM % FLOAT_NUM_PER_BLK == 0;
        if constexpr (localIsBasicBlock) {
            SoftmaxFlashV2NoUpdateBasicBlock(dst, expSumTensor, maxTensor, src, workLocal, tiling);
        } else {
            uint32_t splitK = 0;
            if constexpr (config.oriSrcK % FLOAT_NUM_PER_BLK == 0) {
                splitK = config.oriSrcK;
            } else {
                splitK = AlignUp(config.oriSrcK, FLOAT_NUM_PER_BLK);
            }
            ReduceLastND reduceParam = { tiling.splitM, config.oriSrcK, tiling.splitM, splitK, tiling.reduceM,
                DEFAULT_REPEAT_STRIDE };
            SoftmaxFlashV2NoUpdateExtImpl(dst, expSumTensor, maxTensor, src, workLocal, originalSrcShape, tiling,
                reduceParam);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void SoftmaxFlashV2NDExtImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& expSumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<T1>& expMaxTensor,
    const LocalTensor<T2>& inExpSumTensor, const LocalTensor<T2>& inMaxTensor, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling, ReduceLastND& reduceParam)
{
    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint32_t splitSize = tiling.splitSize;
    uint32_t reduceSize = tiling.reduceSize;
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i <= tiling.rangeM; i++) {
        SoftmaxFlashV2UpdateImpl(dst, expSumTensor, maxTensor, src, expMaxTensor, inExpSumTensor, inMaxTensor,
            workLocal, reduceParam, tiling, offset1, offset2, splitSize, reduceSize);
        offset1 += tiling.splitSize;
        offset2 += tiling.reduceSize;
        if (i == (tiling.rangeM - 1)) {
            if (tiling.tailM == 0) {
                break;
            }
            offset2 = tiling.rangeM * tiling.reduceSize;
            offset1 = tiling.rangeM * tiling.splitSize;
            splitSize = tiling.tailSplitSize;
            reduceSize = tiling.tailReduceSize;
            reduceParam.originalSrcM = tiling.tailM;
            reduceParam.srcM = tiling.tailM;
            reduceParam.dstM = tiling.tailM;
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T1, typename T2, bool isBasicBlock = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftmaxFlashV2NDImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& expSumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<T1>& expMaxTensor,
    const LocalTensor<T2>& inExpSumTensor, const LocalTensor<T2>& inMaxTensor, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    if constexpr (config.oriSrcM == 0 || config.oriSrcK == 0) {
        if constexpr (isBasicBlock) {
            SoftmaxFlashV2BasicBlockImpl(dst, expSumTensor, maxTensor, src, expMaxTensor, inExpSumTensor, inMaxTensor,
                workLocal, tiling);
        } else {
            ReduceLastND reduceParam = { tiling.splitM, originalSrcShape.k, tiling.splitM,
                tiling.splitK, tiling.reduceM,     tiling.reduceK };
            SoftmaxFlashV2NDExtImpl<T1, T2>(dst, expSumTensor, maxTensor, src, expMaxTensor, inExpSumTensor,
                inMaxTensor, workLocal, originalSrcShape, tiling, reduceParam);
        }
    } else {
        constexpr uint32_t basicBlockMaxK = 2048;
        constexpr bool localIsBasicBlock =  config.oriSrcK % FLOAT_REPEAT_SIZE == 0 &&
            config.oriSrcK < basicBlockMaxK && config.oriSrcM % FLOAT_NUM_PER_BLK == 0;
        if constexpr (localIsBasicBlock) {
            SoftmaxFlashV2BasicBlockImpl(dst, expSumTensor, maxTensor, src, expMaxTensor, inExpSumTensor, inMaxTensor,
                workLocal, tiling);
        } else {
            uint32_t splitK = 0;
            ReduceLastND reduceParam;
            if constexpr (config.oriSrcK % FLOAT_NUM_PER_BLK == 0) {
                splitK = config.oriSrcK;
            } else {
                splitK = AlignUp(config.oriSrcK, FLOAT_NUM_PER_BLK);
            }
            if constexpr (SupportType<T2, half>()) {
                reduceParam = { tiling.splitM, config.oriSrcK, tiling.splitM, splitK, tiling.reduceM,
                    DEFAULT_REPEAT_STRIDE * HALF_FACTOR };
            } else if constexpr (SupportType<T2, float>()) {
                reduceParam = { tiling.splitM, config.oriSrcK, tiling.splitM, splitK, tiling.reduceM,
                    DEFAULT_REPEAT_STRIDE };
            }
            SoftmaxFlashV2NDExtImpl<T1, T2>(dst, expSumTensor, maxTensor, src, expMaxTensor, inExpSumTensor,
                inMaxTensor, workLocal, originalSrcShape, tiling, reduceParam);
        }
    }
}

__aicore__ inline void SoftmaxFlashV2NDExtImpl(const LocalTensor<half>& dst, const LocalTensor<float>& expSumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& src, const LocalTensor<half>& expMaxTensor,
    const LocalTensor<float>& inExpSumTensor, const LocalTensor<float>& inMaxTensor,
    const LocalTensor<float>& workLocal, const LastAxisShapeND& originalSrcShape,
    const SoftMaxTiling& tiling, ReduceLastND& reduceParam)
{
    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint32_t splitSize = tiling.splitSize;
    uint32_t reduceSize = tiling.reduceSize;
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i <= tiling.rangeM; i++) {
        SoftmaxFlashV2UpdateImpl(dst, expSumTensor, maxTensor, src, expMaxTensor, inExpSumTensor, inMaxTensor,
            workLocal, reduceParam, tiling, offset1, offset2, splitSize, reduceSize);
        offset1 += tiling.splitSize;
        offset2 += tiling.reduceSize;
        if (i == (tiling.rangeM - 1)) {
            if (tiling.tailM == 0) {
                break;
            }
            offset2 = tiling.rangeM * tiling.reduceSize;
            offset1 = tiling.rangeM * tiling.splitSize;
            splitSize = tiling.tailSplitSize;
            reduceSize = tiling.tailReduceSize;
            reduceParam.originalSrcM = tiling.tailM;
            reduceParam.srcM = tiling.tailM;
            reduceParam.dstM = tiling.tailM;
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T1, typename T2, bool isBasicBlock = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftmaxFlashV2NDImpl(const LocalTensor<half>& dst, const LocalTensor<float>& expSumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& src, const LocalTensor<half>& expMaxTensor,
    const LocalTensor<float>& inExpSumTensor, const LocalTensor<float>& inMaxTensor,
    const LocalTensor<float>& workLocal, const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    if constexpr (config.oriSrcM == 0 || config.oriSrcK == 0) {
        if constexpr (isBasicBlock) {
            SoftmaxFlashV2BasicBlock(dst, expSumTensor, maxTensor, src, expMaxTensor, inExpSumTensor, inMaxTensor,
                workLocal, tiling);
        } else {
            ReduceLastND reduceParam = { tiling.splitM, originalSrcShape.k, tiling.splitM,
                tiling.splitK, tiling.reduceM,     tiling.reduceK };
            SoftmaxFlashV2NDExtImpl(dst, expSumTensor, maxTensor, src, expMaxTensor, inExpSumTensor, inMaxTensor,
                workLocal, originalSrcShape, tiling, reduceParam);
        }
    } else {
        constexpr uint32_t basicBlockMaxK = 2048;
        constexpr bool localIsBasicBlock =  config.oriSrcK % FLOAT_REPEAT_SIZE == 0 &&
            config.oriSrcK < basicBlockMaxK && config.oriSrcM % FLOAT_NUM_PER_BLK == 0;
        if constexpr (localIsBasicBlock) {
            SoftmaxFlashV2BasicBlock(dst, expSumTensor, maxTensor, src, expMaxTensor, inExpSumTensor, inMaxTensor,
                workLocal, tiling);
        } else {
            uint32_t splitK = 0;
            if constexpr (config.oriSrcK % FLOAT_NUM_PER_BLK == 0) {
                splitK = config.oriSrcK;
            } else {
                splitK = AlignUp(config.oriSrcK, FLOAT_NUM_PER_BLK);
            }
            ReduceLastND reduceParam = { tiling.splitM, config.oriSrcK, tiling.splitM, splitK, tiling.reduceM,
                DEFAULT_REPEAT_STRIDE };
            SoftmaxFlashV2NDExtImpl(dst, expSumTensor, maxTensor, src, expMaxTensor, inExpSumTensor, inMaxTensor,
                workLocal, originalSrcShape, tiling, reduceParam);
        }
    }
}

template <typename T1, typename T2, bool isUpdate = false, bool isBasicBlock = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftmaxFlashV2PostProcess(const LocalTensor<T1>& dstTensor, const LocalTensor<T2>& expSumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& srcTensor, const LocalTensor<T1>& expMaxTensor,
    const LocalTensor<T2>& inExpSumTensor, const LocalTensor<T2>& inMaxTensor, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    if constexpr (!isUpdate) {
        SoftmaxFlashV2NoUpdate<T1, T2, isBasicBlock, config>(dstTensor, expSumTensor, maxTensor, srcTensor, workLocal,
            originalSrcShape, tiling);
    } else {
        SoftmaxFlashV2NDImpl<T1, T2, isBasicBlock, config>(dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            inExpSumTensor, inMaxTensor, workLocal, originalSrcShape, tiling);
    }
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV2_COMMON_IMPL_H