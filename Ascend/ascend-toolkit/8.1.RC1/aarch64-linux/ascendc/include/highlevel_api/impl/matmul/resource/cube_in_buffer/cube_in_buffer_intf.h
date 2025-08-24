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
* \file cube_in_buffer_intf.h
* \brief
*/
#ifndef IMPL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_INTF_H
#define IMPL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_INTF_H

#include "../../utils/matmul_module.h"
#include "../../utils/matmul_utils.h"

#include "cube_in_buffer_utils.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    CubeInBuffer is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    CubeInBuffer is only for internal usage, does not support extension or customized specialization!
*/
/*
CubeInBuffer: responsible for L1 buffer management.
This module provides ablities to allocate or free one l1 buffer block, and pipeline syncronization.
*/
template <typename IMPL, class INPUT_TYPE, const auto& MM_CFG, typename = void>
class CubeInBuffer {
    using TransT = typename INPUT_TYPE::TRANS_T;
public:
    __aicore__ inline CubeInBuffer() {}
    __aicore__ inline ~CubeInBuffer() {}
    /**
     * @description: Init of buffer, should be called when matmul is inited.
     * @param: baseBlockSize: element nums of basic block when loading to L1
     * @param: cacheNum: describe the nums of basic block when loading to L1
     * @return: void
     */
    __aicore__ inline void Init(int32_t baseBlockSize, int32_t cacheNum) {}

    /**
     * @description: Reset all should be called when matmul end
     * @param: void
     * @return: void
     */
    __aicore__ inline void Destroy() {}

    /**
     * @description: Judge if data of current iteration is already in buffer
     * @param: iterIndex: current index of iteration
     * @param: bufferPos: current buffer position
     * @return: true if already in buffer, else false
     */
    __aicore__ inline bool Hit(int32_t iterIndex, int32_t bufferPos = -1)
    {
        return false;
    }

    /**
     * @description: Get buffer only when hit
     * @param: iterIndex: current index of iteration
     * @param: bufferPos: current buffer position
     * @return: tensor on L1
     */
    __aicore__ inline LocalTensor<TransT> GetBuffer(int32_t iterIndex, int32_t bufferPos = -1)
    {
        return NULL_TENSOR<TransT>;
    }

    /**
     * @description: Allocate one block of buffer, should be called only when current iterindex does not hit
     * @param: bufferPos: current buffer position
     * @return: void
     */
    __aicore__ inline LocalTensor<TransT> AllocTensor(int32_t bufferPos = -1)
    {
        return NULL_TENSOR<TransT>;
    }

    /**
     * @description: Free tensor, should be called after AllocTensor
     * @param: bufferPos: current buffer position
     * @param: tensor: tensor allocated by AllocTensor or NULL_TENSOR
     * @return: void
     */
    __aicore__ inline void FreeTensor(int32_t bufferPos = -1, const LocalTensor<TransT>& tensor = NULL_TENSOR<TransT>) {}

    /**
     * @description: Reset the status of que in CubeInBuffer
     * @return: void
     */
    __aicore__ inline void Reset() {}

    /**
     * @description: Put tensor to buffer que
     * @param: tensor: target tensor on L1
     * @param: iterIndex: current index of iteration
     * @return: void
     */
    __aicore__ inline void EnQue(LocalTensor<TransT>& tensor) {}

    /**
     * @description: Fetch tensor from que
     * @param: void
     * @return: void
     */
    __aicore__ inline void DeQue() {}
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _CUBE_IN_BUFFER_INTF_H_
