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
 * \file n_loop_batch.h
 * \brief
 */


#ifndef IMPL_MATMUL_SCHEDULER_ITERATOR_BATCH_N_LOOP_N_LOOP_BATCH_H
#define IMPL_MATMUL_SCHEDULER_ITERATOR_BATCH_N_LOOP_N_LOOP_BATCH_H

#include "n_loop_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    NLoop is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    NLoop is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG>
class NLoop<IMPL, INPUT_TYPE, MM_CFG,
    enable_if_t<IsBmmEnableScheduler<INPUT_TYPE, MM_CFG> && (ToMatmulConfig(MM_CFG).scheduleType != ScheduleType::OUTER_PRODUCT)>>
{
    MATMUL_USE_MODULE(MatmulShapeTiling);
public:
    __aicore__ inline NLoop() = default;
    __aicore__ inline ~NLoop() = default;

    __aicore__ inline void Init(int32_t batchShape)
    {
        SetSingleShape(batchShape);
    }

    __aicore__ inline void SetSingleShape(int32_t singleShape)
    {
        if constexpr (ToMatmulConfig(MM_CFG).singleCoreN != 0 && ToMatmulConfig(MM_CFG).basicN != 0) {
            CalcParamsFromCFG();
        } else {
            CalcParamsFromTiling(singleShape);
        }
        ASCENDC_ASSERT((nIter_ > 0), {
            KERNEL_LOG(KERNEL_ERROR, "invalid singleCoreN, nIter_ is %d , which should be larger than 0",
                nIter_);
        });
    }

    __aicore__ inline uint32_t GetTotalIter() const
    {
        return nIter_;
    }

    // About Outer Loop
    // Action Function
    __aicore__ inline bool OuterNext()
    {
        nIdx_++;
        if (OuterEnd()) {
            return false;
        }
        UpdateOuterParams();
        return true;
    }

    __aicore__ inline void OuterStart()
    {
        nIdx_ = 0;
        UpdateOuterParams();
    }

    __aicore__ inline bool OuterEnd()
    {
        return nIdx_ >= nIter_;
    }

    __aicore__ inline bool InnerNext()
    {
        return OuterNext();
    }

    __aicore__ inline void InnerStart()
    {
        OuterStart();
    }

    __aicore__ inline bool InnerEnd()
    {
        return OuterEnd();
    }

    // Params Function
    __aicore__ inline uint32_t GetOuterIdx() const
    {
        return nIdx_;
    }

    __aicore__ inline uint32_t GetInnerIdx() const
    {
        return nIdx_;
    }

    __aicore__ inline uint32_t GetOuterIter() const
    {
        return nIter_;
    }

    __aicore__ inline int32_t GetBaseShape() const
    {
        return baseShape_;
    }

    __aicore__ inline int32_t GetBaseBlockShape() const
    {
        return baseBlockShape_;
    }

    __aicore__ inline int32_t GetTileShape() const
    {
        return baseShape_;
    }

    __aicore__ inline int32_t GetTileBlockShape() const
    {
        return baseBlockShape_;
    }

private:
    __aicore__ inline void CalcParamsFromCFG()
    {
        nIter_ = GetNIter(MM_CFG);
        tailN_ = ToMatmulConfig(MM_CFG).singleCoreN % ToMatmulConfig(MM_CFG).basicN;
        if (tailN_ == 0) {
            tailN_ = ToMatmulConfig(MM_CFG).basicN;
        }
    }

    __aicore__ inline void CalcParamsFromTiling(int32_t singleShape)
    {
        nIter_ = Ceil(singleShape, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
        tailN_ = singleShape % MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        if (tailN_ == 0) {
            tailN_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        }
    }

    __aicore__ inline void UpdateOuterParams()
    {
        if constexpr (NoTailN(MM_CFG)) {
            baseShape_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        } else if constexpr (IsBasicN(MM_CFG)) {
            baseShape_ = tailN_;
        } else {
            baseShape_ = (nIdx_ + 1 == nIter_) ? tailN_ : MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        }
        baseBlockShape_ = Ceil(baseShape_, BLOCK_CUBE);
    }

    uint32_t nIter_;
    int32_t tailN_;
    // OuterLoop Index
    uint32_t nIdx_;

    int32_t baseShape_;
    int32_t baseBlockShape_;
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_SCHEDULER_ITERATOR_BATCH_N_LOOP_N_LOOP_BATCH_H
