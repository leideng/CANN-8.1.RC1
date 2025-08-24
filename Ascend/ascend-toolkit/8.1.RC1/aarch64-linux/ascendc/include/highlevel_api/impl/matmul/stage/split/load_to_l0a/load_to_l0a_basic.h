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
 * \file load_to_l0a_basic.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_BASIC_H
#define IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_BASIC_H

#include "load_to_l0a_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, typename A_TYPE, const auto& MM_CFG>
class LoadToL0A<IMPL, A_TYPE, MM_CFG, 
    enable_if_t<GetGemvMode<A_TYPE>() == GemvMode::MATRIX &&
                (DoMatmulBasicBlock(MM_CFG) || DoMatmulSpecialBasicBlock(MM_CFG)) &&
                MatmulFeatureTrait<MM_CFG>::IsSupportLoad3dV2() &&
                !MatmulFeatureTrait<MM_CFG>::IsSupportLoad2dV2()>>
{
    using A_T = typename A_TYPE::T;
public:
    __aicore__ inline LoadToL0A() {};
    __aicore__ inline ~LoadToL0A() {};

    __aicore__ inline void SetScalar(A_T scalar) {};

    __aicore__ inline void Prepare(bool isATranspose, uint16_t aL1K, uint16_t aL1M) const
    {
        if (isATranspose) {
            Load3DSetFMatrixCal(1, aL1K, padList);
        } else {
            // fmatrix w should be 16 aligned
            Load3DSetFMatrixCal(1, ToMatmulConfig(MM_CFG).basicM, padList);
        }
    }

    __aicore__ inline void Load(const LocalTensor<A_T> &l0A, const LocalTensor<A_T> &l1A,
     uint16_t aL1M, uint16_t aL1K, uint16_t madM, uint16_t madK, uint16_t aL1MOffset, uint16_t aL1KOffset,
     bool isATranspose) const
    {
        LoadData3DParamsV2Pro loadData3DV2;
        if (isATranspose) {
            // format(K, M), K, M need to be 16 aligned for f32
            uint16_t usedKAlign; // k value optimization
            if constexpr (ToMatmulConfig(MM_CFG).basicK != 0) {
                constexpr uint16_t align = (ToMatmulConfig(MM_CFG).basicK + HW_M0 - 1) / HW_M0 * HW_M0;
                usedKAlign = align;
            } else {
                usedKAlign = CeilAlign(madK, HW_M0);
            }
            // K_axis is m direction, and M_axis is k direction in load3d intrin
            loadData3DV2.channelSize = ToMatmulConfig(MM_CFG).basicM;
            loadData3DV2.extConfig = ((uint64_t)0 << M_POS_BIT) | ((uint64_t)0 << K_POS_BIT) |
                                ((uint64_t)usedKAlign << M_STEP_BIT) | (uint64_t)ToMatmulConfig(MM_CFG).basicM;
            loadData3DV2.enTranspose = true;
        } else {
            // format(M, K), K_axis is k direction, and M_axis is m direction in load3d intrin
            // k direction need to be 8 aligned for f32
            uint16_t usedKAlign;
            if constexpr (ToMatmulConfig(MM_CFG).basicK != 0) {
                constexpr uint16_t align = (ToMatmulConfig(MM_CFG).basicK + c0Size_ - 1) / c0Size_ * c0Size_;;
                usedKAlign = align;
            } else {
                usedKAlign = CeilAlign(madK, HW_M0);
            }
            loadData3DV2.channelSize = aL1K;
            loadData3DV2.extConfig = ((uint64_t)0 << M_POS_BIT) | ((uint64_t)0 << K_POS_BIT) |
                                ((uint64_t)ToMatmulConfig(MM_CFG).basicM << M_STEP_BIT) | (uint64_t)usedKAlign;
        }
        LoadData<A_T>(l0A[0], l1A[0], loadData3DV2);
    }
private:
    constexpr static int32_t c0Size_ = AuxGetC0Size<A_T>();
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_BASIC_H