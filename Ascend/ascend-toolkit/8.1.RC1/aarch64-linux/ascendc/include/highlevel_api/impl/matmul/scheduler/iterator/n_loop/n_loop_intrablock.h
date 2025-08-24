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
 * \file n_loop_intrablock.h
 * \brief
 */


#ifndef IMPL_MATMUL_SCHEDULER_ITERATOR_N_LOOP_N_LOOP_INTRABLOCK_H
#define IMPL_MATMUL_SCHEDULER_ITERATOR_N_LOOP_N_LOOP_INTRABLOCK_H

#include "n_loop_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    NLoop is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    NLoop is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, const auto& MM_CFG>
class NLoop<IMPL, A_TYPE, MM_CFG, enable_if_t<IsIntrablock<MM_CFG>>>
{
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulSubBlockInfo);
public:
    __aicore__ inline NLoop() = default;
    __aicore__ inline ~NLoop() = default;

    __aicore__ inline void Init(int32_t singleShape)
    {
        SetSingleShape(singleShape);
    }

    __aicore__ inline void SetSingleShape(int32_t singleShape)
    {
        if (MATMUL_MODULE(MatmulSubBlockInfo)->GetSubBlockIdx() == 0) {
            totalIter_ = Ceil(singleShape, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
            tailBaseShape_ = singleShape % MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
            if (tailBaseShape_ == 0) {
                tailBaseShape_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
            }
            ASCENDC_ASSERT((totalIter_ > 0), { KERNEL_LOG(KERNEL_ERROR,
                "invalid singleCoreN, totalIter_ is %d , which should be larger than 0", totalIter_);
            });
        } else {
            v1TotalIter_ = Ceil(singleShape, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
            v1TailBaseShape_ = singleShape % MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
            if (v1TailBaseShape_ == 0) {
                v1TailBaseShape_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
            }
            ASCENDC_ASSERT((v1TotalIter_ > 0), { KERNEL_LOG(KERNEL_ERROR,
                "invalid singleCoreN, v1TotalIter_ is %d , which should be larger than 0", v1TotalIter_);
            });
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetTotalIter() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return v1TotalIter_;
        } else {
            return totalIter_;
        }
    }

    template <bool IS_SPLIT = false, bool IS_INTRA_BLOCK = false>
    __aicore__ inline bool OuterNext()
    {
        if constexpr (IS_SPLIT) {
            innerStartIdx_ += 1;
        } else {
            innerStartIdx_ += innerIter_;
        }
        UpdateOuterParams<IS_SPLIT, IS_INTRA_BLOCK>();
        return !OuterEnd<IS_SPLIT, IS_INTRA_BLOCK>();
    }

    template <bool IS_SPLIT = false, bool IS_INTRA_BLOCK = false>
    __aicore__ inline void OuterStart()
    {
        innerStartIdx_ = 0;
        UpdateOuterParams<IS_SPLIT, IS_INTRA_BLOCK>();
    }

    template <bool IS_SPLIT = false, bool IS_INTRA_BLOCK = false>
    __aicore__ inline bool OuterEnd()
    {
        if constexpr (IS_SPLIT && IS_INTRA_BLOCK) {
            return innerStartIdx_ >= v1TotalIter_;
        } else{
            return innerStartIdx_ >= totalIter_;
        }
    }

    __aicore__ inline uint32_t GetOuterIdx() const
    {
        return Ceil(innerStartIdx_, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepN());
    }

    __aicore__ inline uint32_t GetOuterIter() const
    {
        return Ceil(totalIter_, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepN());
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline int32_t GetTileShape() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return v1BaseShape_;
        } else {
            return baseShape_;
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline int32_t GetTileBlockShape() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return v1BaseBlockShape_;
        } else {
            return baseBlockShape_;
        }
    }

    __aicore__ inline bool InnerNext()
    {
        ++innerIndex_;
        UpdateInnerParams();
        return !InnerEnd();
    }

    __aicore__ inline void InnerStart()
    {
        innerIndex_ = innerStartIdx_;
        UpdateInnerParams();
    }

    __aicore__ inline bool InnerEnd()
    {
        return innerIndex_ >= innerStartIdx_ + innerIter_;
    }

    __aicore__ inline uint32_t GetInnerIdx() const
    {
        return innerIndex_;
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetInnerIter() const
    {
        return innerIter_;
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline int32_t GetBaseShape() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return v1BaseShape_;
        } else {
            return baseShape_;
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline int32_t GetBaseBlockShape() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return v1BaseBlockShape_;
        } else {
            return baseBlockShape_;
        }
    }

private:
    template <bool IS_SPLIT = false, bool IS_INTRA_BLOCK = false>
    __aicore__ inline void UpdateOuterParams()
    {
        if constexpr (IS_SPLIT) {
            if constexpr (IS_INTRA_BLOCK) {
                v1BaseShape_ = (innerStartIdx_ + 1 == v1TotalIter_) ? v1TailBaseShape_ :
                    MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
            } else {
                baseShape_ = (innerStartIdx_ + 1 == totalIter_) ? tailBaseShape_ :
                    MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
            }
        } else {
            innerIter_ = (totalIter_ - innerStartIdx_) > MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepN() ?
                MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepN() : (totalIter_ - innerStartIdx_);
        }
    }

    __aicore__ inline void UpdateInnerParams()
    {
        baseShape_ = (innerIndex_ + 1 == totalIter_) ? tailBaseShape_ : MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        baseBlockShape_ = Ceil(baseShape_, BLOCK_CUBE);
    }

    uint32_t totalIter_;
    // InnerLoop
    uint32_t innerIndex_ = 0;
    uint32_t innerIter_;
    uint32_t innerStartIdx_ = 0;
    // Shape
    int32_t baseShape_;
    int32_t tailBaseShape_;
    int32_t baseBlockShape_;

    // intrablock params
    uint32_t v1TotalIter_;
    // intrablock Shape
    int32_t v1BaseShape_;
    int32_t v1TailBaseShape_;
    int32_t v1BaseBlockShape_;
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _N_LOOP_INTRABLOCK_H_
