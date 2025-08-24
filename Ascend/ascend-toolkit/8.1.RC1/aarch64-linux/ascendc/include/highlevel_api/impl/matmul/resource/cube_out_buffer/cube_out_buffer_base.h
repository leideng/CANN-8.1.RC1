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
* \file cube_out_buffer_base.h
* \brief
*/
#ifndef IMPL_MATMUL_RESOURCE_CUBE_OUT_BUFFER_CUBE_OUT_BUFFER_BASE_H
#define IMPL_MATMUL_RESOURCE_CUBE_OUT_BUFFER_CUBE_OUT_BUFFER_BASE_H

namespace AscendC {
namespace Impl {
namespace Detail {
enum class UNIT_FLAG_CTRL : uint8_t {
    DISABLE,
    RESERVED,
    CHECK,
    SET,
};

// L0cType
template <bool ENABLE_UNITFLAG>
struct L0cType {
    __aicore__ inline L0cType() {};
};

template <>
struct L0cType<true> {
    __aicore__ inline L0cType() {};
    using BUFFER = TBuf<TPosition::CO1>;
};

template <>
struct L0cType<false> {
    __aicore__ inline L0cType() {};
    using BUFFER = TQue<TPosition::CO1, QUEUE_DEPTH>;
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_RESOURCE_CUBE_OUT_BUFFER_CUBE_OUT_BUFFER_BASE_H