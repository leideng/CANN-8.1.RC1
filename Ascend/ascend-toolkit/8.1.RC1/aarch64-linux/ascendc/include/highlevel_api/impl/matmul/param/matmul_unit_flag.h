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
 * \file matmul_unit_flag.h
 * \brief matmul unit flag manager
 */

#ifndef IMPL_MATMUL_PARAM_MATMUL_UNIT_FLAG_H
#define IMPL_MATMUL_PARAM_MATMUL_UNIT_FLAG_H

#include "../utils/matmul_module.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, const auto &MM_CFG, typename = void>
class MatmulUnitFlag
{
public:
    __aicore__ inline uint8_t GetUnitFlag(bool isLast)
    {
        if constexpr (EnUnitFlag(MM_CFG)) {
            return isLast ? UNIT_FLAG_SET : UNIT_FLAG_CHECK;
        } else {
            return 0;
        }
    }
};

template <typename IMPL, const auto &MM_CFG>
class MatmulUnitFlag<IMPL, MM_CFG, enable_if_t<!MatmulFeatureTrait<MM_CFG>::IsUnitFlagEnabled()>>
{
public:
    __aicore__ inline uint8_t GetUnitFlag(bool isLast)
    {
        return 0;
    }
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_PARAM_MATMUL_UNIT_FLAG_H
