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
* \file copy_bias_in_v220.h
* \brief copy bias data into c1 buffer, only support version V220 and above.
*/

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_BIAS_COPY_BIAS_IN_V220_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_BIAS_COPY_BIAS_IN_V220_H

#include "copy_bias_in_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {

/**
 * CopyBiasIn: responsible for copy bias data management.
 * This module provides ablities to copy bias data in C1 Buffer.
 * We retain the freedom to make incompatible changes, but do not guarantee the stability.
 * CopyBiasIn is only for internal usage, does not support extension or customized specialization!
 */
constexpr static int32_t DB_NUM = 2;
template <typename IMPL, class A_TYPE, class BIAS_TYPE, const auto &MM_CFG>
class CopyBiasIn<IMPL, A_TYPE, BIAS_TYPE, MM_CFG, enable_if_t<
    !MatmulFeatureTrait<MM_CFG>::IsNeedUB() &&
    ToMatmulConfig(MM_CFG).enableSetBias &&
    (A_TYPE::layout == LayoutMode::NONE || ToMatmulConfig(MM_CFG).batchMode == BatchMode::SINGLE_LARGE_THAN_L1) &&
    (PhyPosIsUB(BIAS_TYPE::pos) || PhyPosIsGM(BIAS_TYPE::pos)) &&
    (DoMatmulMDL(MM_CFG) || isNormEnableScheduler<A_TYPE, MM_CFG> ||
    IsBmmEnableScheduler<A_TYPE, MM_CFG> || DoMatmulSpecialMDL(MM_CFG) || IsBasicBlockEnable<MM_CFG> ||
    DoMatmulIBShareNorm(MM_CFG))>>
{
    MATMUL_USE_MODULE(NLoop);

    using BiasT = typename BIAS_TYPE::T;
    using TensorT = typename Conditional<(PhyPosIsGM(BIAS_TYPE::pos) || !MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1()),
                                         GlobalTensor<BiasT>, LocalTensor<BiasT>>::type;

public:
    __aicore__ inline CopyBiasIn() = default;
    __aicore__ inline ~CopyBiasIn() = default;

    __aicore__ inline void
    Copy(LocalTensor<BiasT>& bias, TensorT& srcTensor, int32_t dataLen, int32_t dataNum = 1, int32_t srcOffset = 0)
    {
        (void)dataNum;
        if constexpr ((PhyPosIsUB(BIAS_TYPE::pos) && MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1())) {
            uint16_t blockLen = CeilAlign(dataLen, BLOCK_CUBE) / AscendCUtils::GetC0Count(sizeof(BiasT));
            DataCopy(bias, srcTensor[srcOffset],{ (uint16_t)1, blockLen, (uint16_t)0, (uint16_t)0 });
        } else {
            if constexpr (isL0DBScene_) {
                if (MATMUL_MODULE(NLoop)->IsL0DoubleBuffer()) {
                    if constexpr(!isNormEnableScheduler<A_TYPE, MM_CFG>) {
                        dataLen *= DB_NUM;
                    } else {
                        if (MATMUL_MODULE(NLoop)->GetL0DBLoopNum() == DB_NUM) {
                            if (MATMUL_MODULE(NLoop)->GetInnerIdx() + DB_NUM == MATMUL_MODULE(NLoop)->GetTotalIter()) {
                                dataLen = dataLen + MATMUL_MODULE(NLoop)->GetTailShape();
                            } else {
                                dataLen *= DB_NUM;
                            }
                        }
                    }
                }
            }
            DataCopy(bias, srcTensor[srcOffset], { 1, 1, static_cast<uint16_t>(dataLen), 0, 1, 1, 1, 0 });
        }
    }

private:
    constexpr static bool isL0DBScene_ = (MatmulFeatureTrait<MM_CFG>().IsSupportMNL0DB() &&
                                          ToMatmulConfig(MM_CFG).iterateOrder == IterateOrder::ORDER_M);
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _COPY_BIAS_IN_V220_H_
