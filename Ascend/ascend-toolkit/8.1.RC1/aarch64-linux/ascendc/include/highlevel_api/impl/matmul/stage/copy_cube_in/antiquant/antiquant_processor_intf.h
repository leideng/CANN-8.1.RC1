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
 * \file antiquant_processor_intf.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_ANTIQUANT_ANTIQUANT_PROCESSOR_INTF_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_ANTIQUANT_ANTIQUANT_PROCESSOR_INTF_H

#include "../../../feature_trait/matmul_chip_cap.h"

namespace AscendC {
namespace Impl {
namespace Detail {

template <typename IMPL, class A_TYPE, class B_TYPE, const auto& MM_CFG, typename = void>
class MatmulAntiQuantProcessor
{
    using TransT = typename A_TYPE::T;
    using SrcT = typename B_TYPE::T;
public:
    __aicore__ inline MatmulAntiQuantProcessor() {}
    __aicore__ inline ~MatmulAntiQuantProcessor() {}

    /**
     * @description: Set anti-quant Scalar mode and Scalar params
     * @param: offsetScalar: anti-quant offset
     * @param: offsetScalar: anti-quant scale
     * @return: void
     */
    __aicore__ inline void SetAntiQuantScalar(const TransT offsetScalar, const TransT scaleScalar)
    {
        ASCENDC_ASSERT((false),
        { KERNEL_LOG(KERNEL_ERROR, "Do not support set anti-quant mode."); });
    }

    /**
     * @description: Set anti-quant VectorMode and vector tensor params
     * @param: offsetTensor: anti-quant offset
     * @param: scaleTensor: anti-quant scale
     * @return: void
     */
    __aicore__ inline void SetAntiQuantVector(const LocalTensor<TransT>& offsetTensor,
            const LocalTensor<TransT>& scaleTensor)
    {
        ASCENDC_ASSERT((false),
            { KERNEL_LOG(KERNEL_ERROR, "Do not support set anti-quant mode."); });
    }

    /**
     * @description: Trans input tensor to specified type
     * @param: quantOut: tensor after changing type
     * @param: quantIn: origin input tensor
     * @param: isBankConflict: if current tiling params is bank-conflict
     * @param: isTranspose: if current input tensor is tranposed
     * @return: void
     */
    __aicore__ inline void AntiQuantCompute(const LocalTensor<TransT>& quantOut, const LocalTensor<SrcT>& quantIn,
        bool isBankConflict, bool isTranspose)
    {
        ASCENDC_ASSERT((false),
            { KERNEL_LOG(KERNEL_ERROR, "Do not support set anti-quant mode."); });
    }
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_OUT_QUANT_QUANT_PROCESSOR_INTF_H