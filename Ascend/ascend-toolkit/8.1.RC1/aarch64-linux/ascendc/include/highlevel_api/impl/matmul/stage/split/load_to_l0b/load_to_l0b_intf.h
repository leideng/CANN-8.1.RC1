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
 * \file load_to_l0b_intf.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0B_INTF_H
#define IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0B_INTF_H

#include "../load_to_l0_utils.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    LoadToL0B is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    LoadToL0B is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class INPUT_TYPE, const auto& MM_CFG, typename = void>
class LoadToL0B
{
        using TransT = typename INPUT_TYPE::TRANS_T;

public:
    __aicore__ inline LoadToL0B() = default;
    __aicore__ inline ~LoadToL0B() = default;

    /**
     * @description: Prepare for LoadToL0B
     * @param: isBTranspose: B matrix transpose status
     * @param: bL1K: the length of K_axis for original bMatrix
     * @return: void
     */
    __aicore__ inline void Prepare(bool isBTranspose, uint16_t bL1K) const {};

    /**
     * @description: load a base block from L1 to L0
     * @param: l0B: dst tensor in L0
     * @param: l1B: src tensor in L1
     * @param: bL1N: the length of N_axis for original bMatrix in L1
     * @param: bL1K: the length of K_axis for original bMatrix in L1
     * @param: madN: the length of N_axis for one base block
     * @param: madK: the length of K_axis for one base block
     * @param: bL1NOffset: Offset of the basic block relative to the original bMatrix in the n direction
     * @param: bL1KOffset: Offset of the basic block relative to the original bMatrix in the k direction
     * @param: isBTranspose: B matrix transpose status
     * @return: void
     */
    __aicore__ inline void Load(const LocalTensor<TransT> &l0B, const LocalTensor<TransT> &l1B,
     uint16_t bL1N, uint16_t bL1K, uint16_t madN, uint16_t madK, uint16_t bL1NOffset, uint16_t bL1KOffset,
     bool isBTranspose, const LocalTensor<uint8_t> &l1BIndexMatrix = {}) const {};
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0B_INTF_H