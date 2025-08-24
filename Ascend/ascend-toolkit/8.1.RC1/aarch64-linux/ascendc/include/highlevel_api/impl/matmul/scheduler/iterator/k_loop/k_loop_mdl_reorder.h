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
* \file k_loop_mdl_reorder.h
* \brief
*/
#ifndef IMPL_MATMUL_SCHEDULER_ITERATOR_K_LOOP_K_LOOP_MDL_REORDER_H
#define IMPL_MATMUL_SCHEDULER_ITERATOR_K_LOOP_K_LOOP_MDL_REORDER_H

#include "k_loop_intf.h"
#include "k_loop_mdl_base.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    KLoop is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    KLoop is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, typename TRANS_T, class A_TYPE, const auto& MM_CFG>
class KLoop<IMPL, TRANS_T, A_TYPE, MM_CFG, enable_if_t<!MatmulFeatureTrait<MM_CFG>::IsNeedUB() && DoMatmulMDL(MM_CFG) && IsKdimReorderLoad<MM_CFG>>>
    : public KLoopMDLBase<IMPL, TRANS_T, A_TYPE, MM_CFG>
{
    MATMUL_USE_MODULE(MatmulShapeTiling);

public:
    using BASE_MODULE = AscendC::Impl::Detail::KLoopMDLBase<IMPL, TRANS_T, A_TYPE, MM_CFG>;
    __aicore__ inline KLoop() = default;
    __aicore__ inline ~KLoop() = default;

    __aicore__ inline void Init(int32_t singleShape)
    {
        BASE_MODULE::SetSingleShape(singleShape);
        InitMinStepKStartIdx();
    }

    __aicore__ inline void InitMinStepKStartIdx()
    {
        int64_t blockIdx = AscendC::GetBlockIdx();
        int32_t stepKa = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKa();
        int32_t stepKb = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKb();
        if (stepKa > stepKb) {
            int32_t maxStepKStartOuterIdx = blockIdx % BASE_MODULE::outerKaIter_;
            minStepKStartOuterIdx_ = maxStepKStartOuterIdx * BASE_MODULE::kaStepFactor_;
        } else {
            int32_t maxStepKStartOuterIdx = blockIdx % BASE_MODULE::outerKbIter_;
            minStepKStartOuterIdx_ = maxStepKStartOuterIdx * BASE_MODULE::kbStepFactor_;
        }
    }

    __aicore__ inline void OuterStart()
    {
        BASE_MODULE::outerIdx_ = minStepKStartOuterIdx_;
        BASE_MODULE::UpdateOuterParams();
    }

    __aicore__ inline bool OuterNext()
    {
        BASE_MODULE::outerIdx_ = (BASE_MODULE::outerIdx_ + 1) % BASE_MODULE::outIter_;
        if (OuterEnd()) {
            return false;
        } else {
            BASE_MODULE::UpdateOuterParams();
            return true;
        }
    }

    __aicore__ inline bool OuterEnd()
    {
        return BASE_MODULE::outerIdx_ % BASE_MODULE::outIter_ == minStepKStartOuterIdx_;
    }

    __aicore__ inline bool FirstOuterIter() const
    {
        return BASE_MODULE::outerIdx_ == minStepKStartOuterIdx_;
    }

    __aicore__ inline bool LastOuterIter() const
    {
        return (BASE_MODULE::outerIdx_ + 1) % BASE_MODULE::outIter_ == minStepKStartOuterIdx_;
    }

    __aicore__ inline bool FirstInnerIter() const
    {
        return BASE_MODULE::innerIdx_ == minStepKStartOuterIdx_ * BASE_MODULE::minStepK_;
    }

    /**
     * @description: Get next ka outer loop index, used for ClearL1BufferCache in SchedulerMDL
     * @param: void
     * @return: return next ka outerIdx
     */
    __aicore__ inline int32_t GetNextOuterKaIdx() const
    {
        return ((BASE_MODULE::outerIdx_ + 1) % BASE_MODULE::outIter_) / BASE_MODULE::kaStepFactor_;
    }

    /**
     * @description: Get next kb outer loop index, used for ClearL1BufferCache in SchedulerMDL
     * @param: void
     * @return: return next kb outerIdx
     */
    __aicore__ inline int32_t GetNextOuterKbIdx() const
    {
        return ((BASE_MODULE::outerIdx_ + 1) % BASE_MODULE::outIter_) / BASE_MODULE::kbStepFactor_;
    }

private:
    int32_t minStepKStartOuterIdx_ {0};
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _K_LOOP_MDL_REORDERH_
