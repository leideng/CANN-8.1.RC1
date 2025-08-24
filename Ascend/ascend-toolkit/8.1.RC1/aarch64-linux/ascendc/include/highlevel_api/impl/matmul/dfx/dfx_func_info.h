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
 * \file dfx_func_info.h
 * \brief
 */

#ifndef MATMUL_DFX_FUNC_INFO_H
#define MATMUL_DFX_FUNC_INFO_H

namespace AscendC {
namespace Impl {
namespace Detail {
struct DfxFuncInfo {
    __aicore__ inline DfxFuncInfo(__gm__ const char* module, __gm__ const char* func, uint32_t funcId)
    :module(module), func(func), funcId(funcId) {
    }
    __gm__ const char* module;
    __gm__ const char* func;
    uint32_t funcId;
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _DFX_FUNC_INFO_H_
