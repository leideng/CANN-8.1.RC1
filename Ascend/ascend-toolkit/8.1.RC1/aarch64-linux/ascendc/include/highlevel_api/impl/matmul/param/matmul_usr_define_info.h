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
* \file matmul_usr_define_info.h
* \brief
*/
#ifndef IMPL_MATMUL_PARAM_MATMUL_USER_DEFINE_INFO_H
#define IMPL_MATMUL_PARAM_MATMUL_USER_DEFINE_INFO_H

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, const auto &MM_CFG, typename = void>
class MatmulUserDefineInfo {
public:
    __aicore__ inline void SetSelfDefineData(uint64_t dataPtr)
    {}

    __aicore__ inline void SetUserDefineInfo(uint64_t tilingPtr)
    {}

    __aicore__ inline uint64_t GetSelfDefineData() const
    {}

    __aicore__ inline uint64_t GetUserDefineInfo() const
    {}

private:
    uint64_t dataPtr_;
    uint64_t tilingPtr_;
};

template <typename IMPL, const auto &MM_CFG>
class MatmulUserDefineInfo<IMPL, MM_CFG, enable_if_t<
    MatmulFeatureTrait<MM_CFG>::IsSupportUserDefine()>> {
public:
    __aicore__ inline void SetSelfDefineData(uint64_t dataPtr)
    {
        dataPtr_ = dataPtr;
    }

    __aicore__ inline void SetUserDefineInfo(uint64_t tilingPtr)
    {
        tilingPtr_ = tilingPtr;
    }

    __aicore__ inline uint64_t GetSelfDefineData() const
    {
        return dataPtr_;
    }

    __aicore__ inline uint64_t GetUserDefineInfo() const
    {
        return tilingPtr_;
    }

private:
    uint64_t dataPtr_;
    uint64_t tilingPtr_;
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _MATMUL_USER_DEFINE_INFO_H_