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
 * \file m_loop_intrablock.h
 * \brief
 */


#ifndef IMPL_MATMUL_SCHEDULER_ITERATOR_M_LOOP_M_LOOP_INTRABLOCK_H
#define IMPL_MATMUL_SCHEDULER_ITERATOR_M_LOOP_M_LOOP_INTRABLOCK_H

#include "m_loop_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    MLoop is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    MLoop is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class MLoop<IMPL, INPUT_TYPE, MM_CFG, enable_if_t<IsIntrablock<MM_CFG>>>
{
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulSubBlockInfo);
public:
    __aicore__ inline MLoop() = default;
    __aicore__ inline ~MLoop() = default;

    __aicore__ inline void Init(int32_t singleShape)
    {
        SetSingleShape(singleShape);
    }

    __aicore__ inline void SetSingleShape(int32_t singleShape)
    {
        if (MATMUL_MODULE(MatmulSubBlockInfo)->GetSubBlockIdx() == 0) {
            totalIter_ = Ceil(singleShape, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM());
            tailBaseShape_ = singleShape % MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
            if (tailBaseShape_ == 0) {
                tailBaseShape_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
            }
            ASCENDC_ASSERT((totalIter_ > 0), { KERNEL_LOG(KERNEL_ERROR,
                "invalid singleCoreM, totalIter_ is %d , which should be larger than 0", totalIter_);
            });
        } else {
            v1TotalIter_ = Ceil(singleShape, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM());
            v1TailBaseShape_ = singleShape % MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
            if (v1TailBaseShape_ == 0) {
                v1TailBaseShape_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
            }
            ASCENDC_ASSERT((v1TotalIter_ > 0), { KERNEL_LOG(KERNEL_ERROR,
                "invalid singleCoreM, v1TotalIter_ is %d , which should be larger than 0", v1TotalIter_);
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

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline void OuterStart()
    {
        innerStartIdx_ = 0;
        UpdateOuterParams<IS_INTRA_BLOCK>();
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline bool OuterNext()
    {
        if constexpr (IS_INTRA_BLOCK) {
            innerStartIdx_++;
        } else {
            innerStartIdx_ += innerIter_;
        }
        UpdateOuterParams<IS_INTRA_BLOCK>();
        return !OuterEnd();
    }

    __aicore__ inline bool OuterEnd()
    {
        return innerStartIdx_ >= totalIter_;
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOuterIdx() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return innerStartIdx_;
        } else {
            return Ceil(innerStartIdx_, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepM());
        }
    }

    __aicore__ inline uint32_t GetOuterIter() const
    {
        return Ceil(totalIter_, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepM());
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
    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline void UpdateOuterParams()
    {
        if constexpr (IS_INTRA_BLOCK) {
            auto tilingBaseM = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
            v1BaseShape_ = (innerStartIdx_ + 1 == v1TotalIter_) ? v1TailBaseShape_ : tilingBaseM;
            v1BaseBlockShape_ = Ceil(v1BaseShape_, BLOCK_CUBE);
            baseShape_ = (innerStartIdx_ + 1 == totalIter_) ? tailBaseShape_ : tilingBaseM;
            baseBlockShape_ = Ceil(baseShape_, BLOCK_CUBE);
        } else {
            innerIter_ = (totalIter_ - innerStartIdx_) > MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepM() ?
                MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepM() : (totalIter_ - innerStartIdx_);
        }
    }

    __aicore__ inline void UpdateInnerParams()
    {
        baseShape_ = (innerIndex_ + 1 == totalIter_) ? tailBaseShape_ :
            MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
        baseBlockShape_ = Ceil(baseShape_, BLOCK_CUBE);
    }

    uint32_t totalIter_;
    // InnerLoop
    uint32_t innerIndex_ = 0;
    uint32_t innerIter_;
    uint32_t innerStartIdx_ = 0;

    uint32_t v1TotalIter_;
    // Shape
    int32_t baseShape_;
    int32_t tailBaseShape_;
    int32_t baseBlockShape_;
    // Shape
    int32_t v1BaseShape_;
    int32_t v1TailBaseShape_;
    int32_t v1BaseBlockShape_;
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _M_LOOP_INTRABLOCK_H_
