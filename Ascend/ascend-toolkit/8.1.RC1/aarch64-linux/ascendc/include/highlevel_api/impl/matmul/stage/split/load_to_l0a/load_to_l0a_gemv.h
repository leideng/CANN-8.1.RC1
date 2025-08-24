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
 * \file load_to_l0a_gemv.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_GEMV_H
#define IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_GEMV_H

#include "load_to_l0a_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, typename A_TYPE, const auto& MM_CFG>
class LoadToL0A<IMPL, A_TYPE, MM_CFG, 
enable_if_t<GetGemvMode<A_TYPE>() == GemvMode::SCALAR && !MatmulFeatureTrait<MM_CFG>::IsNeedUB()>>
{
    using A_T = typename A_TYPE::T;
public:
    __aicore__ inline LoadToL0A() {};
    __aicore__ inline ~LoadToL0A() {};

    __aicore__ inline void SetScalar(A_T scalar)
    {
        // A/B does not come from GM with IBShare is not support
        if constexpr (DoMatmulIBShareNorm(MM_CFG) && A_TYPE::ibShare) {
            ASCENDC_ASSERT((false), {
                KERNEL_LOG(KERNEL_ERROR, "It is not allowed to set matrix A with scalar when matmul A is ibShare.");
            });
        }
        aScalar_ = scalar;
    }
    
    __aicore__ inline void Prepare(bool isATranspose, uint16_t aL1K, uint16_t aL1M) const {};

    __aicore__ inline void Load(const LocalTensor<A_T> &l0A, const LocalTensor<A_T> &l1A,
     uint16_t aL1M, uint16_t aL1K, uint16_t madM, uint16_t madK, uint16_t aL1MOffset, uint16_t aL1KOffset,
     bool isATranspose) const
    {
        ASSERT(madM == 1);
        InitConstValueParams initConstValueParams {1, (uint16_t)ConstCeil(madK, BLOCK_CUBE * c0Size_),
                                                    0, aScalar_};
        InitConstValue(l0A, initConstValueParams);
        return;
    }
private:
    A_T aScalar_;
    constexpr static int32_t c0Size_ = AuxGetC0Size<A_T>();
};

template <typename IMPL, typename A_TYPE, const auto& MM_CFG>
class LoadToL0A<IMPL, A_TYPE, MM_CFG, 
enable_if_t<GetGemvMode<A_TYPE>() == GemvMode::VECTOR>>
{
    using A_T = typename A_TYPE::T;
    public:
    __aicore__ inline LoadToL0A() {};
    __aicore__ inline ~LoadToL0A() {};

    __aicore__ inline void SetScalar(A_T scalar) {};

    __aicore__ inline void Prepare(bool isATranspose, uint16_t aL1K, uint16_t aL1M) {};

    __aicore__ inline void Load(LocalTensor<A_T> &l0A, const LocalTensor<A_T> &l1A,
     uint16_t aL1M, uint16_t aL1K, uint16_t madM, uint16_t madK, uint16_t aL1MOffset, uint16_t aL1KOffset,
     bool isATranspose)
    {
        int FracSize = BYTE_PER_FRACTAL / sizeof(A_T);
        int repeat = Ceil(madK, FracSize);
        LoadData2dParams loadDataParams;
        loadDataParams.repeatTimes = repeat;
        loadDataParams.srcStride = 1;
        LoadData(l0A[0], l1A[aL1KOffset], loadDataParams);
        return;
    }
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_SPLIT_LOAD_TO_L0A_GEMV_H