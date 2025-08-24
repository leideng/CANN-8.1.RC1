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
* \file c2_buffer_intf.h
* \brief
*/
#ifndef IMPL_MATMUL_RESOURCE_BIAS_BUFFER_C2_BUFFER_C2_BUFFER_INTF_H
#define IMPL_MATMUL_RESOURCE_BIAS_BUFFER_C2_BUFFER_C2_BUFFER_INTF_H

#include "../../../utils/matmul_module.h"
#include "../../../utils/matmul_utils.h"

namespace AscendC {
namespace Impl {
namespace Detail {

/**
 * C2Buffer: responsible for C2 buffer management.
 * This module provides ablities to allocate or free one C2 buffer block, and pipeline syncronization.
 * We retain the freedom to make incompatible changes, but do not guarantee the stability.
 * CopyBiasIn is only for internal usage, does not support extension or customized specialization!
 */
template <typename IMPL, typename L0cT, class A_TYPE, const auto& MM_CFG, typename = void>
class C2Buffer {
public:
    __aicore__ inline C2Buffer() {}
    __aicore__ inline ~C2Buffer() {}
    /**
     * @description: Init of TBuf, should be called when matmul is inited.
     * @return: void
     */
    __aicore__ inline void Init() {}
    /**
     * @description: Allocate one block of buffer, should be called only when load bias
     * @return: null tensor on l1
     */
    __aicore__ inline LocalTensor<L0cT> Allocate()
    {
        return NULL_TENSOR<L0cT>;
    }
    /**
     * @description: Free tensor, should be called after AllocTensor
     * @param: tensor: tensor allocated by AllocTensor or NULL_TENSOR
     * @return: void
     */
    __aicore__ inline void Free() {}
    /**
     * @description: Put tensor to buffer que
     * @param: tensor: target tensor on L1
     * @return: void
     */
    __aicore__ inline void EnQue() {}
    /**
     * @description: Fetch tensor from que
     * @param: void
     * @return: bias tensor on L1
     */
    __aicore__ inline void DeQue() {}
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _C2_BUFFER_INTF_H_
