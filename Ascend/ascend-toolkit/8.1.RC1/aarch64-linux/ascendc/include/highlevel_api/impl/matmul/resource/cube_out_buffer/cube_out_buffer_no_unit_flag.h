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
* \file cube_out_buffer_no_unit_flag.h
* \brief
*/
#ifndef IMPL_MATMUL_MODULES_RESOURCE_CUBE_OUT_BUFFER_CUBE_OUT_BUFFER_NO_UNIT_FLAG_H
#define IMPL_MATMUL_MODULES_RESOURCE_CUBE_OUT_BUFFER_CUBE_OUT_BUFFER_NO_UNIT_FLAG_H

#include "lib/matmul/tiling.h"

#include "../../utils/matmul_utils.h"
#include "cube_out_buffer_base.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    CubeOutBuffer is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    CubeOutBuffer is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, typename L0cT, const auto& MM_CFG, typename = void>
class CubeOutBuffer
{
    MATMUL_USE_MODULE(MatmulShapeTiling)
public:
    __aicore__ inline CubeOutBuffer() {};
    __aicore__ inline ~CubeOutBuffer() {};
    __aicore__ inline void Init(int32_t cacheSize = 1, uint32_t lenFactor = 1)
    {
        constexpr int32_t DB_NUM = 2;
        if constexpr (ToMatmulConfig(MM_CFG).scheduleType == ScheduleType::OUTER_PRODUCT || DoMatmulSpecialMDL(MM_CFG)) {
            lenFactor = DB_NUM;
        }
        if (MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetDbL0C() == DB_NUM) {
            GetTPipePtr()->InitBuffer(
                CO1_, DB_NUM, lenFactor * cacheSize * sizeof(L0cT));
        } else {
            GetTPipePtr()->InitBuffer(
                CO1_, 1, lenFactor * cacheSize * sizeof(L0cT));
        }
    }

    __aicore__ inline LocalTensor<L0cT> AllocTensor()
    {
        cMatrix_ = CO1_.template AllocTensor<L0cT>();
        return cMatrix_;
    }

    __aicore__ inline LocalTensor<L0cT> GetTensor()
    {
        return cMatrix_;
    }

    __aicore__ inline void EnQue(LocalTensor<L0cT>& tensor)
    {
        CO1_.EnQue(tensor);
    }

    __aicore__ inline LocalTensor<L0cT> DeQue()
    {
        return CO1_.template DeQue<L0cT>();
    }

    __aicore__ inline void FreeTensor(LocalTensor<L0cT>& co1Local)
    {
        CO1_.FreeTensor(co1Local);
    }

    __aicore__ inline void Destroy()
    {
        CO1_.FreeAllEvent();
    }

private:
    typename L0cType<false>::BUFFER CO1_;
    LocalTensor<L0cT> cMatrix_;
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_MODULES_RESOURCE_CUBE_OUT_BUFFER_CUBE_OUT_BUFFER_NO_UNIT_FLAG_H