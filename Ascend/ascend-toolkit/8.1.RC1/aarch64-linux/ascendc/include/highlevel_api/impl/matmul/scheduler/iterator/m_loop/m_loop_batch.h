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
 * \file m_loop_batch.h
 * \brief
 */


#ifndef IMPL_MATMUL_SCHEDULER_ITERATOR_M_LOOP_M_LOOP_BATCH_H
#define IMPL_MATMUL_SCHEDULER_ITERATOR_M_LOOP_M_LOOP_BATCH_H

#include "m_loop_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    MLoop is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    MLoop is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, const auto &MM_CFG>
class MLoop<IMPL, A_TYPE, MM_CFG,
    enable_if_t<IsBmmEnableScheduler<A_TYPE, MM_CFG> && (ToMatmulConfig(MM_CFG).scheduleType != ScheduleType::OUTER_PRODUCT)>>
{
    MATMUL_USE_MODULE(MatmulShapeTiling);
public:
    __aicore__ inline MLoop() = default;
    __aicore__ inline ~MLoop() = default;

    __aicore__ inline void Init(int32_t batchShape)
    {
        SetSingleShape(batchShape);
    }

    __aicore__ inline void SetSingleShape(int32_t singleShape)
    {
        if constexpr (ToMatmulConfig(MM_CFG).singleCoreM != 0 && ToMatmulConfig(MM_CFG).basicM != 0) {
            CalcParamsFromCFG();
        } else {
            CalcParamsFromTiling(singleShape);
        }
        ASCENDC_ASSERT((mIter_ > 0), {
            KERNEL_LOG(KERNEL_ERROR, "invalid singleCoreM, mIter_ is %d , which should be larger than 0",
                mIter_);
        });
    }

    __aicore__ inline uint32_t GetTotalIter() const
    {
        return mIter_;
    }

    // About Outer Loop
    // Action Function
    __aicore__ inline bool OuterNext()
    {
        mIdx_++;
        if (OuterEnd()) {
            return false;
        }
        UpdateOuterParams();
        return true;
    }

    __aicore__ inline void OuterStart()
    {
        mIdx_ = 0;
        UpdateOuterParams();
    }

    __aicore__ inline bool OuterEnd()
    {
        return mIdx_ >= mIter_;
    }

    __aicore__ inline void InnerStart()
    {
        OuterStart();
    }

    __aicore__ inline bool InnerNext()
    {
        return OuterNext();
    }

    __aicore__ inline bool InnerEnd()
    {
        return OuterEnd();
    }

    __aicore__ inline uint32_t GetOuterIdx() const
    {
        return mIdx_;
    }

    __aicore__ inline uint32_t GetInnerIdx() const
    {
        return mIdx_;
    }

    __aicore__ inline uint32_t GetOuterIter() const
    {
        return mIter_;
    }

    __aicore__ inline int32_t GetTileShape() const
    {
        return baseShape_;
    }

    __aicore__ inline int32_t GetBaseShape() const
    {
        return baseShape_;
    }

    __aicore__ inline int32_t GetTileBlockShape() const
    {
        return baseBlockShape_;
    }

    __aicore__ inline int32_t GetBaseBlockShape() const
    {
        return baseBlockShape_;
    }
private:
    __aicore__ inline void CalcParamsFromCFG()
    {
        mIter_ = GetMIter(MM_CFG);
        tailM_ = ToMatmulConfig(MM_CFG).singleCoreM % ToMatmulConfig(MM_CFG).basicM;
        if (tailM_ == 0) {
            tailM_ = ToMatmulConfig(MM_CFG).basicM;
        }
    }

    __aicore__ inline void CalcParamsFromTiling(int32_t singleShape)
    {
        mIter_ = Ceil(singleShape, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM());
        tailM_ = singleShape % MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
        if (tailM_ == 0) {
            tailM_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
        }
    }

    __aicore__ inline void UpdateOuterParams()
    {
        if constexpr (NoTailM(MM_CFG)) {
            baseShape_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
        } else if constexpr (IsBasicM(MM_CFG)) {
            baseShape_ = tailM_;
        } else {
            baseShape_ = (mIdx_ + 1 == mIter_) ? tailM_ : MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM();
        }
        baseBlockShape_ = Ceil(baseShape_, BLOCK_CUBE);
    }

    uint32_t mIter_;
    int32_t tailM_;
    // OuterLoop Index
    uint32_t mIdx_;
    // shape
    int32_t baseShape_;
    int32_t baseBlockShape_;
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_SCHEDULER_ITERATOR_BATCH_M_LOOP_M_LOOP_BATCH_H
