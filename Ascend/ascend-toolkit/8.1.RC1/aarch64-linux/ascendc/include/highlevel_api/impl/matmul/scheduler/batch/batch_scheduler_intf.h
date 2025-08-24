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
 * \file batch_scheduler_intf.h
 * \brief
 */
#ifndef IMPL_MATMUL_SCHEDULER_BATCH_BATCH_SCHEDULER_INTF_H
#define IMPL_MATMUL_SCHEDULER_BATCH_BATCH_SCHEDULER_INTF_H

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    BatchScheduler is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    BatchScheduler is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, typename = void>
class BatchScheduler
{
public:
    __aicore__ inline BatchScheduler() = default;
    __aicore__ inline ~BatchScheduler() = default;

    /**
     * @description: iterate batch(output to GM or VECIN)
     * @param: dst: ouput matrix memory addr (GlobalTensor<DstT> or LocalTensor<DstT>)
     * @param: enPartialSum: whether accumulate partial sum or not
     * @param: enAtomic: whether atomic
     * @param: enSequentialWrite: the data's write type on dst address, continue or flat write
     * @param: matrixStrideA: input matrix stride
     * @param: matrixStrideB: weight matrix stride
     * @param: matrixStrideC: output matrix stride
     * @param: batchA: input matrix batch num
     * @param: batchB: weight matrix batch num
     * @param: batchOuter: batch outer loop count
     * @return: void
     */
    template <class T>
    __aicore__ inline void Schedule(const T& dst, bool enPartialSum, uint8_t enAtomic, bool enSequentialWrite,
        const uint32_t matrixStrideA, const uint32_t matrixStrideB, const uint32_t matrixStrideC) {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "Matching error. This is an empty implementation."); });
    }
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_SCHEDULER_BATCH_BATCH_SCHEDULER_INTF_H