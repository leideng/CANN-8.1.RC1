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
 * \file matmul_server_utils.h
 * \brief
 */
#ifndef IMPL_MATMUL_KFC_MATMUL_SERVER_UTILS_H
#define IMPL_MATMUL_KFC_MATMUL_SERVER_UTILS_H

#include "lib/matmul/matmul.h"
#include "kernel_operator.h"

namespace AscendC {

template <bool IS_IBSHARE> struct IBShareCache {
    __aicore__ inline IBShareCache() {};
};

template <>
struct IBShareCache<false> {
    __aicore__ inline IBShareCache() {};
    using ShareCache = uint16_t;
};

template <>
struct IBShareCache<true> {
    __aicore__ inline IBShareCache() {};
    using ShareCache = Impl::Detail::GlobalCache;
};
template <class A_TYPE, class B_TYPE> __aicore__ constexpr bool IsIBShare()
{
    if constexpr (A_TYPE::ibShare == true) {
        return true;
    }
    if constexpr (B_TYPE::ibShare == true) {
        return true;
    }
    return false;
}

struct MatmulMsg {
    uint32_t setOrgShape : 1;
    uint32_t orgM;
    uint32_t orgN;
    uint32_t orgKa;
    uint32_t orgKb;
    uint32_t orgKc;
};

struct ShareMatmulBase {
    __aicore__ inline ShareMatmulBase() {};
};

struct ShareMatmul : ShareMatmulBase {
    __aicore__ inline ShareMatmul(){};
    MatmulMsg msg0;
    MatmulMsg msg1;
};

template <bool SHARED>
struct ShareMatmulAux {
    __aicore__ inline ShareMatmulAux(){};
};

template <>
struct ShareMatmulAux<false> {
    __aicore__ inline ShareMatmulAux(){};
    using MSG = ShareMatmulBase;
};

template <>
struct ShareMatmulAux<true> {
    __aicore__ inline ShareMatmulAux(){};
    using MSG = ShareMatmul;
};

template <const auto& MM_CFG = CFG_NORM>
__aicore__ inline constexpr bool IsSharedMatmul()
{
    if constexpr (!AscendC::ToMatmulConfig(MM_CFG).enableInit ||
        AscendC::ToMatmulConfig(MM_CFG).enableMixDualMaster) {
        return true;
    }
    return false;
}
} // namespace AscendC
#endif // _MATMUL_SERVER_H_