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
 * \file m_loop_norm_base.h
 * \brief
 */


#ifndef IMPL_MATMUL_SCHEDULER_ITERATOR_M_LOOP_M_LOOP_NORM_BASE_H
#define IMPL_MATMUL_SCHEDULER_ITERATOR_M_LOOP_M_LOOP_NORM_BASE_H

#include "m_loop_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    MLoopNormBase is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    MLoopNormBase is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, const auto& MM_CFG>
class MLoopNormBase
{
    MATMUL_USE_MODULE(MatmulShapeTiling);
public:
    __aicore__ inline MLoopNormBase() = default;
    __aicore__ inline ~MLoopNormBase() = default;

    __aicore__ inline void Init(int32_t singleShape)
    {
        SetSingleShape(singleShape);
    }

    __aicore__ inline void SetSingleShape(int32_t singleShape)
    {
        if constexpr (ToMatmulConfig(MM_CFG).singleCoreM != 0 && !ToMatmulConfig(MM_CFG).enableSetTail) {
            SetSingleShapeFromCFG();
        } else {
            SetSingleShapeFromTiling(singleShape);
        }
        ASCENDC_ASSERT((totalIter_ > 0), {
            KERNEL_LOG(KERNEL_ERROR, "invalid singleCoreM, totalIter_ is %d , which should be larger than 0",
                totalIter_);
        });
    }

    __aicore__ inline uint32_t GetTotalIter() const
    {
        return totalIter_;
    }

    __aicore__ inline bool OuterNext()
    {
        if constexpr (IsBasicM(MM_CFG)) {
            return false;
        } else {
            innerStartIdx_ += innerIter_;
            UpdateOuterParams();
            return !OuterEnd();
        }
    }

    __aicore__ inline void OuterStart()
    {
        innerStartIdx_ = 0;
        UpdateOuterParams();
    }

    __aicore__ inline bool OuterEnd()
    {
        if constexpr (IsBasicM(MM_CFG)) {
            return true;
        } else {
            return innerStartIdx_ >= totalIter_;
        }
    }

    __aicore__ inline uint32_t GetOuterIdx() const
    {
        return Ceil(innerStartIdx_, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepM());
    }

    __aicore__ inline uint32_t GetOuterIter() const
    {
        return Ceil(totalIter_, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepM());
    }

    __aicore__ inline int32_t GetTileShape() const
    {
        return baseShape_;
    }

    __aicore__ inline int32_t GetTileBlockShape() const
    {
        return baseBlockShape_;
    }

    __aicore__ inline int32_t GetTailShape() const
    {
        return tailBaseShape_;
    }

    __aicore__ inline bool InnerNext()
    {
        if constexpr (IsBasicM(MM_CFG)) {
            return false;
        } else {
            ++innerIndex_;
            UpdateInnerParams();
            return !InnerEnd();
        }
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

    __aicore__ inline uint32_t GetInnerIter() const
    {
        return innerIter_;
    }

    __aicore__ inline int32_t GetBaseShape() const
    {
        return baseShape_;
    }

    __aicore__ inline int32_t GetBaseBlockShape() const
    {
        return baseBlockShape_;
    }

protected:
    __aicore__ inline void SetSingleShapeFromCFG()
    {
        totalIter_ = GetMIter(MM_CFG);
        tailBaseShape_ = ToMatmulConfig(MM_CFG).singleCoreM % ToMatmulConfig(MM_CFG).basicM;
        if (tailBaseShape_ == 0) {
            tailBaseShape_ = ToMatmulConfig(MM_CFG).basicM;
        }
    }

    __aicore__ inline void SetSingleShapeFromTiling(int32_t singleShape)
    {
        totalIter_ = Ceil(singleShape, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM());
        tailBaseShape_ = singleShape % MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
        if (tailBaseShape_ == 0) {
            tailBaseShape_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
        }
    }

    __aicore__ inline void UpdateOuterParams()
    {
        if constexpr (IsBasicM(MM_CFG)) {
            innerIter_ = 1;
        } else {
            innerIter_ = (totalIter_ - innerStartIdx_) > MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepM() ?
                MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepM() : (totalIter_ - innerStartIdx_);
        }
    }

    __aicore__ inline void UpdateInnerParams()
    {
        if constexpr (NoTailM(MM_CFG)) {
            baseShape_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
        } else if constexpr (IsBasicM(MM_CFG)) {
            baseShape_ = tailBaseShape_;
        } else {
            baseShape_ = (innerIndex_ + 1 == totalIter_) ? tailBaseShape_ : MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
        }
        baseBlockShape_ = Ceil(baseShape_, BLOCK_CUBE);
    }

    uint32_t totalIter_;
    // InnerLoop
    uint32_t innerIndex_ = 0;
    uint32_t innerIter_;
    uint32_t innerStartIdx_ = 0;
    // Shape
    int32_t baseShape_;
    int32_t baseBlockShape_;
    int32_t tailBaseShape_;
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _M_LOOP_NORM_BASE_H
