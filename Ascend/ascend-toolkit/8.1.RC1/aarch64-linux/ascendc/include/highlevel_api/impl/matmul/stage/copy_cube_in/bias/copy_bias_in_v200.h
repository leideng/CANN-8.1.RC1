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
* \file copy_bias_in_v200.h
* \brief copy bias data into c1 buffer, only support version V200.
*/

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_BIAS_COPY_BIAS_IN_V200_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_BIAS_COPY_BIAS_IN_V200_H

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
template <typename IMPL, class A_TYPE, class BIAS_TYPE, const auto &MM_CFG>
class CopyBiasIn<IMPL, A_TYPE, BIAS_TYPE, MM_CFG, enable_if_t<
    MatmulFeatureTrait<MM_CFG>::IsNeedUB() &&
    ToMatmulConfig(MM_CFG).enableSetBias &&
    (PhyPosIsUB(BIAS_TYPE::pos) || PhyPosIsGM(BIAS_TYPE::pos)) &&
    (DoMatmulMDL(MM_CFG) || isNormEnableScheduler<A_TYPE, MM_CFG> ||
    IsBmmEnableScheduler<A_TYPE, MM_CFG> || IsBasicBlockEnable<MM_CFG> || DoMatmulIBShareNorm(MM_CFG))>>
{
    MATMUL_USE_MODULE(CubeOutBuffer);
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(LocalWorkspace);

    using BiasT = typename BIAS_TYPE::T;
    using TensorT = typename Conditional<(PhyPosIsGM(BIAS_TYPE::pos) || !MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1()),
                                         GlobalTensor<BiasT>, LocalTensor<BiasT>>::type;
public:
    __aicore__ inline CopyBiasIn() = default;
    __aicore__ inline ~CopyBiasIn() = default;

    __aicore__ inline void
    Copy(LocalTensor<BiasT>& bias, TensorT& srcTensor, int32_t dataLen, int32_t dataNum = 1, int32_t srcOffset = 0)
    {
        BiasCopy(bias, srcTensor, dataLen, srcOffset);
        // broad cast bias to cmatrix
        BroadCastBiasToL0C(bias);
    }

private:
    __aicore__ inline void BiasCopy(LocalTensor<BiasT>& bias, TensorT& srcTensor, int32_t dataLen, int32_t srcOffset)
    {
        if constexpr (PhyPosIsUB(BIAS_TYPE::pos)) {
            bias = srcTensor[srcOffset];
        } else if constexpr (PhyPosIsGM(BIAS_TYPE::pos)) {
            // get local bias workspace
            bias = MATMUL_MODULE(LocalWorkspace)->GetWorkspaceWithOffset(0).template ReinterpretCast<BiasT>();
            constexpr int32_t c0Size = AuxGetC0Size<BiasT>();
            dataLen = Ceil(dataLen, c0Size) * c0Size;
            bias.SetSize(dataLen * sizeof(BiasT));

            event_t eventIDMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
            DataCopy(bias, srcTensor[srcOffset], dataLen);
            event_t eventIDMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
        }
    }

    __aicore__ inline void BroadCastBiasToL0C(LocalTensor<BiasT>& bias)
    {
        // BroadCast bias to L0C
        const auto& cMatrix = MATMUL_MODULE(CubeOutBuffer)->GetTensor();
        auto blockUseN = MATMUL_MODULE(NLoop)->GetBaseBlockShape();
        auto blockUseM = MATMUL_MODULE(MLoop)->GetBaseBlockShape();
        if (blockUseN <= MAX_REPEAT_TIMES) {
            for (int32_t i = 0; i < blockUseM; ++i) {
                BroadCastVecToMM(cMatrix[i * CUBE_MAX_SIZE], bias, blockUseN, 1, 0, blockUseM - 1);
            }
        } else {
            int32_t loop = blockUseN / MAX_REPEAT_TIMES;
            int32_t loopTail = blockUseN % MAX_REPEAT_TIMES;
            auto rowStride = MAX_REPEAT_TIMES * CUBE_MAX_SIZE;
            auto colStride = blockUseM * CUBE_MAX_SIZE;

            // broadcast bias to cmatrix by element
            int32_t dstOffset = 0;
            int32_t srcOffset = 0;
            for (int32_t i = 0; i < blockUseM; ++i) {
                for (int32_t j = 0; j < loop; ++j) {
                    BroadCastVecToMM(cMatrix[dstOffset], bias[srcOffset], MAX_REPEAT_TIMES, 1, 0, blockUseM - 1);
                    srcOffset += BLOCK_CUBE;
                    dstOffset += colStride;
                }
                if (loopTail) {
                    BroadCastVecToMM(cMatrix[dstOffset], bias[srcOffset], loopTail, 1, 0, blockUseM - 1);
                }
                dstOffset += rowStride;
            }
        }

        // The L0C waits for the completion of the UB copy.
        event_t eventIDVToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_M));
        SetFlag<HardEvent::V_M>(eventIDVToM);
        WaitFlag<HardEvent::V_M>(eventIDVToM);
    }
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _COPY_BIAS_IN_V200_H_
