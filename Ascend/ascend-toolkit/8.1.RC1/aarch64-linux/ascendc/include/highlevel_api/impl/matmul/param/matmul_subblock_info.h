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
 * \file matmul_subblock_info.h
 * \brief matmul variable manager
 */

#ifndef IMPL_MATMUL_PARAM_MATMUL_SUBBLOCK_INFO_H
#define IMPL_MATMUL_PARAM_MATMUL_SUBBLOCK_INFO_H

#include "../utils/matmul_module.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, const auto &MM_CFG>
class MatmulSubBlockInfo {
public:
    __aicore__ inline void SetSubBlockIdx(uint8_t subBlockIdx)
    {
#if __CCE_AICORE__ == 220
        ASCENDC_ASSERT((subBlockIdx < MIX_NUM),
            { KERNEL_LOG(KERNEL_ERROR, "subBlockIdx is %d , which should only be [0,%d) ", subBlockIdx, MIX_NUM); });
#endif
        subBlockIdx_ = subBlockIdx;
    }

    __aicore__ inline void SetFakeMsg(bool fakeMsg)
    {
        fakeMsg_ = fakeMsg;
    }

    __aicore__ inline uint8_t GetSubBlockIdx() const
    {
        return subBlockIdx_;
    }

    __aicore__ inline bool GetFakeMsg() const
    {
        return fakeMsg_;
    }

    __aicore__ inline bool IsFakeIntraBlock() const
    {
        return fakeMsg_ || subBlockIdx_ == 0;
    }

private:
    uint8_t subBlockIdx_ {0};
    bool fakeMsg_ {false};
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_PARAM_MATMUL_SUBBLOCK_INFO_H
