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
 * \file dfx_chain_handler.h
 * \brief
 */

#ifndef MATMUL_DFX_CHAIN_HANDLER_H
#define MATMUL_DFX_CHAIN_HANDLER_H

namespace AscendC {
namespace Impl {
namespace Detail {

struct DfxFuncInfo;

template <typename ...HANDLERS> 
struct DfxChainHandler {
    template <typename... Args>
    __aicore__ inline static void PreCall(const DfxFuncInfo& info, Args&&... args) {
        (HANDLERS::PreCall(info, std::forward<Args>(args)...), ...);
    }

    template <typename RT>
    __aicore__ inline static void PostCall(const DfxFuncInfo& info, const RT& ret) {
        (HANDLERS::PostCall(info, ret), ...);
    }

    __aicore__ inline static void PostCall(const DfxFuncInfo& info) {
        (HANDLERS::PostCall(info), ...);
    }
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _DFX_CHAIN_HANDLER_H_
