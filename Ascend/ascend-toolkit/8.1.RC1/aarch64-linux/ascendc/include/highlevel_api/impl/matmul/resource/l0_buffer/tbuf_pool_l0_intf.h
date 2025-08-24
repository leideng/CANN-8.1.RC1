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
 * \file tbuf_pool_l0_intf.h
 * \brief
 */
#ifndef IMPL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_INTF_H
#define IMPL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_INTF_H

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    TBufPoolL0 is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    TBufPoolL0 is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, typename A_TYPE, typename B_TYPE, const auto& MM_CFG, typename = void>
class TBufPoolL0 {
public:
    __aicore__ inline TBufPoolL0() = default;
    __aicore__ inline ~TBufPoolL0() = default;

    /**
     * @description: Init TBufPoolL0
     * @param: isL0Db: if l0Db is enable
     * @return: void
     */
    __aicore__ inline void Init() {};

    /**
     * @description: Set l0Db
     * @param: isL0Db: if l0Db is enable
     * @return: void
     */
    __aicore__ inline void SetDBFlag(bool isL0Db = true) {};

    /**
     * @description: allocate buffer
     * @param: isL0Db: if l0Db is enable
     * @return: a reference to this TBufPoolL0
     */
    __aicore__ inline TBufPoolL0& Allocate() {
        return *this;
    }

    /**
     * @description: get buffer
     * @param: Pos: tensor logical position
     * @param: T: tensor dtype
     * @return: tensor on l0
     */
    template <TPosition Pos, typename T>
    __aicore__ inline LocalTensor<T> GetBuffer()
    {
        LocalTensor<T> tempTensor;
        return tempTensor;
    }

    /**
     * @description: judge if hit l0cache
     * @param: Pos: tensor logical position
     * @param: pos: abs pos in single
     * @param: T: tensor dtype
     * @return: if hit l0cache
     */
    template <TPosition Pos>
    __aicore__ inline bool Hit(uint32_t pos = 0) {
        return false;
    };

    /**
     * @description: reset l0Cache
     * @return: void
     */
    __aicore__ inline void ResetCache() {};

    /**
     * @description: Put tensor to buffer que
     * @return: void
     */
    __aicore__ inline void EnQue() {};
    
    /**
     * @description: Fetch tensor from que
     * @return: void
     */
    __aicore__ inline void DeQue() {};

    /**
     * @description: Free tensor, should be called after Allocate
     * @return: void
     */
    __aicore__ inline void Free() {};
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_RESOURCE_L0_BUFFER_TBUF_POOL_L0_INTF_H
