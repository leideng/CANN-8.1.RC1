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
 * \file n_loop_batch_db.h
 * \brief
 */

#ifndef IMPL_MATMUL_SCHEDULER_ITERATOR_N_LOOP_N_LOOP_BATCH_DB_H
#define IMPL_MATMUL_SCHEDULER_ITERATOR_N_LOOP_N_LOOP_BATCH_DB_H

#include "n_loop_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {

template <typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class NLoop<IMPL, INPUT_TYPE, MM_CFG,
    enable_if_t<(IsBmmEnableScheduler<INPUT_TYPE, MM_CFG> && (ToMatmulConfig(MM_CFG).scheduleType == ScheduleType::OUTER_PRODUCT))>>
{
    MATMUL_USE_MODULE(MatmulShapeTiling);
public:
    __aicore__ inline NLoop() = default;
    __aicore__ inline ~NLoop() = default;

    __aicore__ inline void Init(int32_t batchDbShape)
    {
        SetSingleShape(batchDbShape);
    }

    __aicore__ inline void SetSingleShape(int32_t singleShape)
    {
        nIter_ = Ceil(singleShape, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
        tailN_ = singleShape % MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        if (tailN_ == 0) {
            tailN_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        }
        isL0aDB = 1;
        ASCENDC_ASSERT((nIter_ > 0), {
            KERNEL_LOG(KERNEL_ERROR, "invalid singleCoreN, nIter_ is %d , which should be larger than 0",
                nIter_);
        });
    }

    __aicore__ inline int32_t GetTotalIter() const
    {
        return nIter_;
    }

    __aicore__ inline bool OuterNext()
    {
        if ((tailN_ == MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN() && nIter_ % DB_FACTOR == 0) || nIdx_ < nIter_ - DB_FACTOR) {
            nIdx_ = nIdx_ + DB_FACTOR;
            isL0aDB = DB_FACTOR;
        } else {
            nIdx_ = nIdx_ + 1;
            isL0aDB = 1;
        }
        UpdateOuterParams();
        return !OuterEnd();
    }

    __aicore__ inline void OuterStart()
    {
        nIdx_ = 0;
        if ((tailN_ == MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() && nIter_ % DB_FACTOR == 0) || nIdx_ < nIter_ - DB_FACTOR) {
            isL0aDB = DB_FACTOR;
        } else {
            isL0aDB = 1;
        }
        UpdateOuterParams();
    }

    __aicore__ inline bool OuterEnd()
    {
        return nIdx_ >= nIter_;
    }

    __aicore__ inline bool InnerNext()
    {
        nIdx_ = nIdx_ + 1;
        UpdateOuterParams();
        return !OuterEnd();
    }

    __aicore__ inline void InnerStart()
    {
        nIdx_ = 0;
        isL0aDB = 1;
        UpdateOuterParams();
    }

    __aicore__ inline bool InnerEnd()
    {
        return nIdx_ >= nIter_;
    }

    __aicore__ inline void UpdateOuterParams()
    {
        baseShape_ = (nIdx_ + 1 == nIter_) ? tailN_ : MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN();
        baseBlockShape_ = Ceil(baseShape_, BLOCK_CUBE);
    }

    // Params Function
    __aicore__ inline int32_t GetOuterIdx() const
    {
        return nIdx_;
    }

    __aicore__ inline int32_t GetInnerIdx() const
    {
        return nIdx_;
    }

    __aicore__ inline int32_t GetTileShape() const
    {
        return baseShape_;
    }

    __aicore__ inline int32_t GetTileBlockShape() const
    {
        return baseBlockShape_;
    }

    __aicore__ inline int32_t GetTail() const
    {
        return tailN_;
    }

    __aicore__ inline int32_t GetBaseShape() const
    {
        return baseShape_;
    }

    __aicore__ inline int32_t GetBaseBlockShape() const
    {
        return baseBlockShape_;
    }

    __aicore__ inline int32_t GetL0DBLoopNum() const
    {
        return isL0aDB;
    }

private:
    int32_t nIter_;
    int32_t tailN_;

    int32_t nIdx_;
    int32_t isL0aDB;
    int32_t baseShape_;
    int32_t baseBlockShape_;
};
}
}
}
#endif // IMPL_MATMUL_SCHEDULER_ITERATOR_N_LOOP_N_LOOP_BATCH_DB_H