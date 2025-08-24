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
* \file copy_bias_in_intf.h
* \brief
*/

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_BIAS_COPY_BIAS_IN_INTF_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_BIAS_COPY_BIAS_IN_INTF_H

namespace AscendC {
namespace Impl {
namespace Detail {

/**
 * CopyBiasIn: responsible for copy bias data management.
 * This module provides ablities to copy bias data in C1/C2 Buffer.
 * We retain the freedom to make incompatible changes, but do not guarantee the stability.
 * CopyBiasIn is only for internal usage, does not support extension or customized specialization!
 */
template <typename IMPL, class A_TYPE, class BIAS_TYPE, const auto &MM_CFG, typename = void>
class CopyBiasIn {
    using BiasT = typename BIAS_TYPE::T;
    using TensorT = typename Conditional<(PhyPosIsGM(BIAS_TYPE::pos) || !MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1()),
                                         GlobalTensor<BiasT>, LocalTensor<BiasT>>::type;
public:
    __aicore__ inline CopyBiasIn() = default;
    __aicore__ inline ~CopyBiasIn() = default;

    /**
     * @description: Load bias data to L1
     * @param: bias: L1 bias tensor
     * @param: srcTensor: input bias tensor
     * @param: dataLen: Length of bias blocks
     * @param: dataNum: Number of bias blocks
     * @param: srcOffset: position offset of source data
     * @return: Tensor on L1
     */
    __aicore__ inline void
    Copy(LocalTensor<BiasT>& bias, TensorT& srcTensor, int32_t dataLen, int32_t dataNum = 1, int32_t srcOffset = 0) {}
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _COPY_BIAS_IN_INTF_H_
