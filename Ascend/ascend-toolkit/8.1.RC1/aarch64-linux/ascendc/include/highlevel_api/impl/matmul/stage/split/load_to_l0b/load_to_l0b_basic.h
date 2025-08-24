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
 * \file load_to_l0b_basic.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0B_BASIC_H
#define IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0B_BASIC_H

#include "load_to_l0b_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class LoadToL0B<IMPL, INPUT_TYPE, MM_CFG,
    enable_if_t<(DoMatmulBasicBlock(MM_CFG) || DoMatmulSpecialBasicBlock(MM_CFG)) &&
                MatmulFeatureTrait<MM_CFG>::IsSupportLoad3dV2() &&
                !MatmulFeatureTrait<MM_CFG>::IsSupportLoad2dV2()>>
{
    using TransT = typename INPUT_TYPE::TRANS_T;
public:
    __aicore__ inline LoadToL0B() {};
    __aicore__ inline ~LoadToL0B() {};
    
    __aicore__ inline void Prepare(bool isBTranspose, uint16_t bL1K) const
    {
        if (!isBTranspose) {
            Load3DSetFMatrixBCal(1, bL1K, padList);
        }
    }

    __aicore__ inline void Load(const LocalTensor<TransT> &l0B, const LocalTensor<TransT> &l1B,
     uint16_t bL1N, uint16_t bL1K, uint16_t madN, uint16_t madK, uint16_t bL1NOffset, uint16_t bL1KOffset,
     bool isBTranspose, const LocalTensor<uint8_t> &l1BIndexMatrix = NULL_TENSOR<uint8_t>) const
    {
        constexpr uint16_t typeSize = sizeof(TransT);
        if (isBTranspose) {
            // SET LOAD2D parameters , loop axis: K or M, or 1
            // k is c0Size_ aligned for f32
            constexpr uint16_t nFraC0 = ToMatmulConfig(MM_CFG).basicN / HW_N0;
            uint16_t l0bLoop = 1;
            uint16_t l0bSrcAddrStride = 0;
            uint16_t l0bDstAddrStride = 0;
            uint8_t l0bRepeat = 0;
            uint16_t l0bSrcstride = 1;
            uint16_t l0bDststride = 0;
            if constexpr (ToMatmulConfig(MM_CFG).singleCoreM != 0 && ToMatmulConfig(MM_CFG).singleCoreN != 0) {
                constexpr uint16_t kC0 = (ToMatmulConfig(MM_CFG).basicK + c0Size_ - 1) / c0Size_;
                constexpr uint16_t repeat = kC0 * nFraC0;
                l0bRepeat = repeat;
                if constexpr (nFraC0 * HW_N0 == ToMatmulConfig(MM_CFG).basicN) {
                    l0bLoop = 1;            // loop=1
                } else if constexpr (nFraC0 >= kC0) { // LOOP is K  and repeat is n axis
                    l0bLoop = kC0;
                    constexpr uint16_t srcStride = ToMatmulConfig(MM_CFG).basicN * c0Size_ * typeSize;
                    constexpr uint16_t dstStride = nFraC0 * HW_N0 * c0Size_ * typeSize;
                    l0bSrcAddrStride = srcStride;
                    l0bDstAddrStride = dstStride;
                    l0bRepeat = nFraC0;

                    l0bSrcstride = 1;
                    l0bDststride = 0;
                } else { // LOOP is N  and repeat is K axis
                    l0bLoop = nFraC0;
                    constexpr uint16_t srcStride = HW_N0 * c0Size_ * typeSize;
                    constexpr uint16_t dstStride = HW_N0 * c0Size_ * typeSize;
                    l0bSrcAddrStride = srcStride;
                    l0bDstAddrStride = dstStride;
                    l0bRepeat = kC0;

                    l0bSrcstride = nFraC0;
                    l0bDststride = nFraC0 - 1;
                }
            } else {
                uint16_t madKAlign = CeilAlign<uint16_t>(madK, c0Size_);
                uint16_t kC0 = madKAlign / c0Size_;
                l0bRepeat = kC0 * nFraC0;

                if constexpr (nFraC0 * HW_N0 == ToMatmulConfig(MM_CFG).basicN) {
                    l0bLoop = 1;            // loop=1
                } else if (nFraC0 >= kC0) { // LOOP is K  and repeat is n axis
                    l0bLoop = kC0;
                    l0bSrcAddrStride = ToMatmulConfig(MM_CFG).basicN * c0Size_ * typeSize;
                    l0bDstAddrStride = nFraC0 * HW_N0 * c0Size_ * typeSize;
                    l0bRepeat = nFraC0;

                    l0bSrcstride = 1;
                    l0bDststride = 0;
                } else { // LOOP is N  and repeat is K axis
                    l0bLoop = nFraC0;
                    l0bSrcAddrStride = HW_N0 * c0Size_ * typeSize;
                    l0bDstAddrStride = HW_N0 * c0Size_ * typeSize;
                    l0bRepeat = kC0;

                    l0bSrcstride = ToMatmulConfig(MM_CFG).basicN / HW_N0;
                    l0bDststride = nFraC0 - 1;
                }
            }
            // use load2d for L1_2_L0B
            // startIndex, repeatTimes, srcStride, sid, dstGap, ifTranspose, addrmode
            LoadData2dParams loadDataParams{0, l0bRepeat, l0bSrcstride, 0, l0bDststride, 0, 0};
            uint64_t l1bOffset = 0;
            uint64_t l0bOffset = 0;
            for (uint64_t i = 0; i < l0bLoop; i++) {
                LoadData(l0B[l0bOffset], l1B[l1bOffset], loadDataParams);
                l1bOffset += (l0bSrcAddrStride / typeSize);
                l0bOffset += (l0bDstAddrStride / typeSize);
            }
        } else {
            // use load3dv2 for L1_2_L0B
            // n_axis is K direction, need to be 16 aligned
            // channel size need to be 16 aligned
            // k_axis is M direction, need to be HW_M0 aligned
            uint16_t mAlign;
            if constexpr (ToMatmulConfig(MM_CFG).basicK != 0) {
                constexpr uint16_t align = (ToMatmulConfig(MM_CFG).basicK + HW_M0 - 1) / HW_M0 * HW_M0;
                mAlign = align;
            } else {
                mAlign = CeilAlign(madK, HW_M0);
            }
            // StepN need to be aligned
            LoadData3DParamsV2Pro loadData3DV2;
            loadData3DV2.channelSize = ToMatmulConfig(MM_CFG).basicN;
            loadData3DV2.extConfig = ((uint64_t)0 << M_POS_BIT) | ((uint64_t)0 << K_POS_BIT) |
                                ((uint64_t)mAlign << M_STEP_BIT) | (uint64_t)ToMatmulConfig(MM_CFG).basicN;
            loadData3DV2.fMatrixCtrl = true;
            LoadData<TransT>(l0B[0], l1B[0], loadData3DV2);
        }
    }

private:
    constexpr static int32_t c0Size_ = AuxGetC0Size<TransT>();
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0B_BASIC_H