/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
* \file copy_bias_in_batch.h
* \brief copy batch bias data into c1 buffer, only support version V220 and above.
*/

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_BIAS_COPY_BIAS_IN_BATCH_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_BIAS_COPY_BIAS_IN_BATCH_H

#include "copy_bias_in_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {

/**
 * CopyBiasIn: responsible for copy bias data management.
 * This module provides ablities to copy bias data in C1/C2 Buffer.
 * We retain the freedom to make incompatible changes, but do not guarantee the stability.
 * CopyBiasIn is only for internal usage, does not support extension or customized specialization!
 */
template <typename IMPL, class A_TYPE, class BIAS_TYPE, const auto &MM_CFG>
class CopyBiasIn<IMPL, A_TYPE, BIAS_TYPE, MM_CFG, enable_if_t<
    !MatmulFeatureTrait<MM_CFG>::IsNeedUB() &&
    ToMatmulConfig(MM_CFG).enableSetBias &&
    A_TYPE::layout != LayoutMode::NONE &&
    ToMatmulConfig(MM_CFG).batchMode != BatchMode::SINGLE_LARGE_THAN_L1 &&
    (PhyPosIsUB(BIAS_TYPE::pos) || PhyPosIsGM(BIAS_TYPE::pos))>> {
    using BiasT = typename BIAS_TYPE::T;
    using TensorT = typename Conditional<(PhyPosIsGM(BIAS_TYPE::pos) || !MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1()),
                                         GlobalTensor<BiasT>, LocalTensor<BiasT>>::type;

public:
    __aicore__ inline CopyBiasIn() = default;
    __aicore__ inline ~CopyBiasIn() = default;

    __aicore__ inline void
    Copy(LocalTensor<BiasT>& bias, TensorT& srcTensor, int32_t dataLen, int32_t dataNum = 1, int32_t srcOffset = 0)
    {
        BiasCopy(bias, srcTensor, dataLen, dataNum, srcOffset);
    }

private:
    constexpr static int32_t c0Size_ = AuxGetC0Size<BiasT>();

private:
    __aicore__ inline void BiasCopy(LocalTensor<BiasT>& bias, TensorT& srcTensor, int32_t dataLen,
                                    int32_t dataNum, int32_t srcOffset)
    {
        // Check if the bias is batched or not
        if constexpr (!ToMatmulConfig(MM_CFG).isBiasBatch) {
            // Not batched, only copy the data once
            DataCopy(bias, srcTensor, { 1, 1, static_cast<uint16_t>(dataLen), 0, 1, 1, 1, 0 });
        } else {
            // Batched, copy the data one by one
            int32_t dstOffset = 0;
            auto dstStride = CeilAlign(dataLen, c0Size_);
            for (int32_t i = 0; i < dataNum; ++i) {
                DataCopy(bias[dstOffset], srcTensor[srcOffset],
                         { 1, 1, static_cast<uint16_t>(dataLen), 0, 1, 1, 1, 0 });
                srcOffset += dataLen;
                dstOffset += dstStride;
            }
        }
    }
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _COPY_BIAS_IN_BATCH_H_
