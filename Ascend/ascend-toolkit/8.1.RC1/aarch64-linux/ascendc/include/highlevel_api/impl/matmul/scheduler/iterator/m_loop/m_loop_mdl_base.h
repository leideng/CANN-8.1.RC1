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
 * \file m_loop_mdl_base.h
 * \brief m_loop base class for mdl and mdl_outer_product
 */


#ifndef IMPL_MATMUL_SCHEDULER_ITERATOR_M_LOOP_M_LOOP_MDL_BASE_H
#define IMPL_MATMUL_SCHEDULER_ITERATOR_M_LOOP_M_LOOP_MDL_BASE_H

#include "../../../utils/matmul_module.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    MLoopMDLBase is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    MLoopMDLBase is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, const auto& MM_CFG>
class MLoopMDLBase
{
    MATMUL_USE_MODULE(MatmulShapeTiling);
public:
    __aicore__ inline MLoopMDLBase() = default;
    __aicore__ inline ~MLoopMDLBase() = default;

    __aicore__ inline void Init(int32_t singleShape)
    {
        SetSingleShape(singleShape);
    }

    __aicore__ inline void SetSingleShape(int32_t singleShape)
    {
        if constexpr (ToMatmulConfig(MM_CFG).singleCoreM != 0 && ToMatmulConfig(MM_CFG).basicM != 0 &&
            !ToMatmulConfig(MM_CFG).enableSetTail) {
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
            outerIndex_++;
            UpdateOuterParams();
            return !OuterEnd();
        }
    }

    __aicore__ inline void OuterStart()
    {
        outerIndex_ = 0;
        innerIndex_ = 0;
        UpdateOuterParams();
    }

    __aicore__ inline bool OuterEnd()
    {
        if constexpr (IsBasicM(MM_CFG)) {
            return true;
        } else {
            return outerIndex_ + 1 > outerIter_;
        }
    }

    __aicore__ inline uint32_t GetOuterIdx() const
    {
        return outerIndex_;
    }

    __aicore__ inline uint32_t GetOuterIter() const
    {
        return outerIter_;
    }

    __aicore__ inline int32_t GetTileShape() const
    {
        return tileShape_;
    }

    __aicore__ inline int32_t GetTileShapeOf(int32_t outerIdx) const
    {
        return (outerIdx + 1 >= outerIter_) ? tailTileShape_ : mainTileShape_;
    }

    __aicore__ inline int32_t GetTileBlockShape() const
    {
        return tileBlockShape_;
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

    __aicore__ inline void UpdateInnerParams()
    {
        if constexpr (IsBasicM(MM_CFG)) {
            baseShape_ = tailBaseShape_;
        } else {
            baseShape_ = (innerIndex_ + 1 == totalIter_) ? tailBaseShape_ : MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
        }
        baseBlockShape_ = Ceil(baseShape_, BLOCK_CUBE);
    }

private:
    __aicore__ inline void SetSingleShapeFromCFG()
    {
        totalIter_ = GetMIter(MM_CFG);
        outerIter_ = Ceil(ToMatmulConfig(MM_CFG).singleCoreM,
                            ToMatmulConfig(MM_CFG).basicM * ToMatmulConfig(MM_CFG).stepM);
        tailBaseShape_ = ToMatmulConfig(MM_CFG).singleCoreM % ToMatmulConfig(MM_CFG).basicM;
        if (tailBaseShape_ == 0) {
            tailBaseShape_ = ToMatmulConfig(MM_CFG).basicM;
        }
        mainTileShape_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() *
                        MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepM();
        tailTileShape_ = ToMatmulConfig(MM_CFG).singleCoreM % mainTileShape_;
        if (tailTileShape_ == 0) {
            tailTileShape_ = mainTileShape_;
        }
    }

    __aicore__ inline void SetSingleShapeFromTiling(int32_t singleShape)
    {
        auto tilingBaseM = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
        totalIter_ = singleShape / tilingBaseM;
        tailBaseShape_ = singleShape - totalIter_ * tilingBaseM;
        if (tailBaseShape_ == 0) {
            tailBaseShape_ = tilingBaseM;
        } else {
            totalIter_ += 1;
        }
        mainTileShape_ = tilingBaseM * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepM();
        outerIter_ = Ceil(singleShape, mainTileShape_);
        tailTileShape_ = singleShape % mainTileShape_;
        if (tailTileShape_ == 0) {
            tailTileShape_ = mainTileShape_;
        }
    }

    __aicore__ inline void UpdateOuterParams()
    {
        if constexpr (IsBasicM(MM_CFG)) {
            innerStartIdx_ = 0;
            innerIter_ = 1;
            tileShape_ = tailTileShape_;
        } else {
            innerStartIdx_ = outerIndex_ * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepM();
            innerIter_ = (totalIter_ - innerStartIdx_) > MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepM() ?
                MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepM() : (totalIter_ - innerStartIdx_);
            tileShape_ = (outerIndex_ + 1 >= outerIter_) ? tailTileShape_ : mainTileShape_;
        }
        tileBlockShape_ = Ceil(tileShape_, BLOCK_CUBE);
    }

protected:
    uint32_t totalIter_;
    // OuterLoop
    uint32_t outerIndex_ = 0;
    uint32_t outerIter_;
    // InnerLoop
    uint32_t innerIndex_ = 0;
    uint32_t innerIter_;
    uint32_t innerStartIdx_ = 0;
    // Shape
    int32_t mainTileShape_;
    int32_t tailTileShape_;
    int32_t tileShape_;
    int32_t tileBlockShape_;
    int32_t baseShape_;
    int32_t tailBaseShape_;
    int32_t baseBlockShape_;
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _M_LOOP_MDL_BASE_H_
