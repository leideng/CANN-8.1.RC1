/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file batch_loop_multi.h
 * \brief
 */


#ifndef IMPL_MATMUL_SCHEDULER_ITERATOR_BATCH_LOOP_BATCH_LOOP_MULTI_H
#define IMPL_MATMUL_SCHEDULER_ITERATOR_BATCH_LOOP_BATCH_LOOP_MULTI_H

#include "batch_loop_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    BatchLoop is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    BatchLoop is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG>
class BatchLoop<IMPL, INPUT_TYPE, MM_CFG,
    enable_if_t<(Impl::Detail::GetCopyCubeInType<INPUT_TYPE, MM_CFG>() == Impl::Detail::CopyCubeInType::BMM) ||
    (Impl::Detail::IsBMMFromL1<INPUT_TYPE, MM_CFG>())>>
{
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    using SrcT = typename INPUT_TYPE::T;

public:
    __aicore__ inline BatchLoop() = default;
    __aicore__ inline ~BatchLoop() = default;

    __aicore__ inline void Init()
    {
        const auto tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        CalcBatchNum(tiling.GetALayoutInfoB(), tiling.GetBLayoutInfoB(), tiling.GetBatchNum(), tiling.GetBatchNum());
        UpdateBatchNumParams();
    }

    __aicore__ inline void SetBatchNum(int32_t batchNumA, int32_t batchNumB)
    {
        CalcBatchNum(batchNumA, batchNumB, batchNumA, batchNumB);
        UpdateBatchNumParams();
    }

    // Outer Loop
    __aicore__ inline void OuterStart()
    {
        outerIdx_ = 0;
        dstOffset_ = 0;
        batchCalcSize_ = batchNum_ * MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreM() *
            MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN();
    }

    __aicore__ inline void OuterNext()
    {
        outerIdx_++;
        dstOffset_ += batchCalcSize_;
    }

    __aicore__ inline bool OuterEnd()
    {
        return outerIdx_ >= batchOuter_;
    }

    __aicore__ inline uint32_t GetOuterIndex() const
    {
        return outerIdx_;
    }

    __aicore__ inline uint32_t GetDstOffset() const
    {
        return dstOffset_;
    }

    __aicore__ inline int32_t GetBatchNum() const
    {
        return batchNum_;
    }

    __aicore__ inline int32_t GetBatchA() const
    {
        return batchA_;
    }

    __aicore__ inline int32_t GetBatchB() const
    {
        return batchB_;
    }

    __aicore__ inline int32_t GetBiasBatchSrcOffset() const
    {
        return outerIdx_ * batchNum_ * MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN();
    }

    // Double Buffer Loop
    __aicore__ inline void SplitStart()
    {
        // Check that the total amount of data to be transferred is less than L1.
        ASSERT((batchA_ * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreM() *
            MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK() +
            batchB_ * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreN() *
            MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetSingleCoreK()) * sizeof(SrcT) <= TOTAL_L1_SIZE);
        splitOuterIdx_ = 0;
        splitBatchIdx_ = 0;
    }

    __aicore__ inline void SplitNext()
    {
        splitOuterIdx_++;
        UpdateSplitParams();
    }

    __aicore__ inline bool SplitEnd()
    {
        return splitOuterIdx_ >= splitSize_;
    }

    __aicore__ inline uint32_t GetSplitIndex() const
    {
        return splitOuterIdx_;
    }

    __aicore__ inline int32_t GetSplitSize() const
    {
        return splitSize_;
    }

    __aicore__ inline int32_t GetSplitBatchNum() const
    {
        return splitBatchNum_;
    }

    // Inner Loop
    __aicore__ inline void InnerStart()
    {
        innerIdx_ = 0;
        UpdateInnerParams();
    }

    __aicore__ inline void InnerNext()
    {
        innerIdx_++;
        UpdateInnerParams();
    }

    __aicore__ inline bool InnerEnd()
    {
        return innerIdx_ >= splitBatchNum_ || splitOuterIdx_ * splitBatchNum_ >= batchNum_;
    }

    __aicore__ inline uint32_t GetInnerIndex() const
    {
        return innerIdx_;
    }

    __aicore__ inline uint32_t GetBatchIndex() const
    {
        return innerBatchIdx_;
    }

private:
    __aicore__ inline void CalcBatchNum(int32_t layoutBatchNumA, int32_t layoutBatchNumB,
        int32_t batchNumA, int32_t batchNumB)
    {
        if constexpr (ToMatmulConfig(MM_CFG).batchMode != BatchMode::BATCH_LARGE_THAN_L1) {
            ASSERT(batchNumA > 0 && batchNumB > 0 &&
                  (batchNumA % batchNumB == 0 || batchNumB % batchNumA == 0));
            batchA_ = batchNumA;
            batchB_ = batchNumB;
            return;
        }

        ASSERT(layoutBatchNumA > 0 && layoutBatchNumB > 0 &&
              (layoutBatchNumA % layoutBatchNumB == 0 || layoutBatchNumB % layoutBatchNumA == 0));
        int32_t aMatrixSingleBatchSize = GetSingleSizeAlignA();
        int32_t bMatrixSingleBatchSize = GetSingleSizeAlignB();
        if ((layoutBatchNumA * aMatrixSingleBatchSize + layoutBatchNumB * bMatrixSingleBatchSize) <= TOTAL_L1_SIZE) {
            batchOuter_ = 1;
            batchA_ = layoutBatchNumA;
            batchB_ = layoutBatchNumB;
            return;
        }
        if (layoutBatchNumA >= layoutBatchNumB) {
            CalcBatchAB(layoutBatchNumA, layoutBatchNumB, aMatrixSingleBatchSize, bMatrixSingleBatchSize);
        } else {
            CalcBatchAB(layoutBatchNumB, layoutBatchNumA, bMatrixSingleBatchSize, aMatrixSingleBatchSize);
        }
    }

    __aicore__ inline int32_t GetSingleSizeAlignA()
    {
        const auto matmulShapeInfo = MATMUL_MODULE(MatmulShapeInfo);
        if (matmulShapeInfo->IsTransposeA()) {
            if constexpr (IsSameType<SrcT, int8_t>::value) {
                return CeilAlign(matmulShapeInfo->GetSingleCoreM(), c0Size_) *
                    CeilAlign(matmulShapeInfo->GetSingleCoreK(), c0Size_) * sizeof(SrcT);
            } else {
                return CeilAlign(matmulShapeInfo->GetSingleCoreM(), c0Size_) *
                    CeilAlign(matmulShapeInfo->GetSingleCoreK(), BLOCK_CUBE) * sizeof(SrcT);
            }
        } else {
            return CeilAlign(matmulShapeInfo->GetSingleCoreM(), BLOCK_CUBE) *
                CeilAlign(matmulShapeInfo->GetSingleCoreK(), c0Size_) * sizeof(SrcT);
        }
    }

    __aicore__ inline int32_t GetSingleSizeAlignB()
    {
        const auto matmulShapeInfo = MATMUL_MODULE(MatmulShapeInfo);
        if (matmulShapeInfo->IsTransposeB()) {
            return CeilAlign(matmulShapeInfo->GetSingleCoreK(), c0Size_) *
                CeilAlign(matmulShapeInfo->GetSingleCoreN(), BLOCK_CUBE) * sizeof(SrcT);
        } else {
            if constexpr (IsSameType<SrcT, int8_t>::value) {
                return CeilAlign(matmulShapeInfo->GetSingleCoreK(), c0Size_) *
                    CeilAlign(matmulShapeInfo->GetSingleCoreN(), c0Size_) * sizeof(SrcT);
            } else {
                return CeilAlign(matmulShapeInfo->GetSingleCoreK(), BLOCK_CUBE) *
                    CeilAlign(matmulShapeInfo->GetSingleCoreN(), c0Size_) * sizeof(SrcT);
            }
        }
    }

    __aicore__ inline void CalcBatchAB(int32_t batchNumLarge, int32_t batchNumLess,
        int32_t largeMatrixSingleBatchSize, int32_t lessMatrixSingleBatchSize)
    {
        int32_t multiples = batchNumLarge / batchNumLess;
        int32_t singleBatchSize = multiples * largeMatrixSingleBatchSize + lessMatrixSingleBatchSize;
        int32_t batchInner = TOTAL_L1_SIZE / singleBatchSize;
        ASSERT(batchInner > 0);
        while (batchNumLess % batchInner != 0 && batchInner > 0) {
            --batchInner;
        }
        batchOuter_ = batchNumLess / batchInner;
        batchA_ = multiples * batchInner;
        batchB_ = batchInner;
    }

    __aicore__ inline void UpdateBatchNumParams()
    {
        batchNum_ = batchA_ > batchB_ ? batchA_ : batchB_;
        splitSize_ = (batchNum_ >= DB_FACTOR) && (batchA_ % DB_FACTOR == 0) &&
            (batchB_ % DB_FACTOR == 0) ? DB_FACTOR : 1;
        splitBatchNum_ = batchNum_ / splitSize_;
    }

    __aicore__ inline void UpdateSplitParams()
    {
        splitBatchIdx_ += batchNum_ / splitSize_;
    }

    __aicore__ inline void UpdateInnerParams()
    {
        innerBatchIdx_ = innerIdx_ + splitBatchIdx_;
    }

    int32_t batchA_;
    int32_t batchB_;
    int32_t batchNum_;
    int32_t batchOuter_ = 1;
    constexpr static int32_t c0Size_ = AuxGetC0Size<typename INPUT_TYPE::T>();

    // outer loop params
    uint32_t outerIdx_;
    int32_t batchCalcSize_;
    uint32_t dstOffset_;

    // split loop params
    uint32_t splitOuterIdx_;
    int32_t splitSize_; // 2 for double buffer, 1 otherwise
    int32_t splitBatchNum_; // batch num per split size
    uint32_t splitBatchIdx_; // global view batch index within split loop

    // inner loop params
    uint32_t innerIdx_;
    uint32_t innerBatchIdx_; // global view batch index within inner loop
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_SCHEDULER_ITERATOR_BATCH_LOOP_BATCH_LOOP_MULTI_H
