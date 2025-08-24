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
 * \file load_to_l0b_load2d.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0B_LOAD2D_H
#define IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0B_LOAD2D_H

#include "load_to_l0b_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class LoadToL0B<IMPL, INPUT_TYPE, MM_CFG, 
    enable_if_t<!(DoMatmulBasicBlock(MM_CFG) || DoMatmulSpecialBasicBlock(MM_CFG)) &&
                (GetLoadInstrType<typename INPUT_TYPE::T, MM_CFG>() == LoadInstrType::LOAD2D)>>
{
    using B_T = typename INPUT_TYPE::T;
public:
    __aicore__ inline LoadToL0B() {};
    __aicore__ inline ~LoadToL0B() {};

    __aicore__ inline void Prepare(bool isBTranspose, uint16_t bL1K) const {};

    __aicore__ inline void Load(const LocalTensor<B_T> &dst, const LocalTensor<B_T> &bMatrix,
     uint16_t bL1N, uint16_t bL1K, uint16_t madN, uint16_t madK, uint16_t bL1NOffset, uint16_t bL1KOffset,
     bool isBTranspose, const LocalTensor<uint8_t> &l1BIndexMatrix = {}) const
    {
        uint16_t blockUseN = Ceil(madN, BLOCK_CUBE);
        uint16_t blockUseK = Ceil(madK, c0Size_);
        int srcL1Offset = 0;
        if constexpr(PhyPosIsL1(INPUT_TYPE::pos)) {
            if (isBTranspose) {
                srcL1Offset = bL1N * bL1KOffset + c0Size_ * bL1NOffset;
            } else {
                srcL1Offset = bL1K * bL1NOffset + c0Size_ * bL1KOffset;
            }
        }
        if (isBTranspose) {
            LoadData2dParams loadDataParams;
            int dstOffset = blockUseN * CUBE_MAX_SIZE / factor_;
            int srcOffset = bL1N * c0Size_;
            if constexpr (!PhyPosIsL1(INPUT_TYPE::pos)) {
                srcOffset = blockUseN * BLOCK_CUBE * c0Size_;
            }
            loadDataParams.repeatTimes = blockUseN;
            loadDataParams.srcStride = 1;
            loadDataParams.ifTranspose = false;

            if (blockUseN == 1) {
                loadDataParams.repeatTimes = blockUseK;
                loadDataParams.srcStride = 1;
                LoadData(dst, bMatrix[srcL1Offset], loadDataParams);
            } else {
                for (int i = 0; i < blockUseK; i++) {
                    LoadData(dst[i * dstOffset], bMatrix[srcL1Offset + i * srcOffset], loadDataParams);
                }
            }
        } else {
            LoadData2dParams loadDataParams;
            int dstOffset = blockUseN * CUBE_MAX_SIZE;
            constexpr int srcOffset = CUBE_MAX_SIZE;
            loadDataParams.repeatTimes = blockUseN;
            if constexpr (PhyPosIsL1(INPUT_TYPE::pos)) {
                // alL B matrix is in L1 buffer
                loadDataParams.srcStride = Ceil(bL1K, BLOCK_CUBE);
            } else {
                loadDataParams.srcStride = blockUseK;
            }
            loadDataParams.ifTranspose = true;
            if (blockUseN == 1) {
                loadDataParams.repeatTimes = blockUseK;
                loadDataParams.srcStride = 1;
                LoadData(dst, bMatrix[srcL1Offset], loadDataParams);
            } else {
                for (int i = 0; i < blockUseK; i++) {
                    LoadData(dst[i * dstOffset], bMatrix[srcL1Offset + i * srcOffset], loadDataParams);
                }
            }
        }
    }
private:
    constexpr static int32_t c0Size_ = AuxGetC0Size<B_T>();
    constexpr static int32_t factor_ = AuxGetFactor<B_T>();
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0B_LOAD2D_H