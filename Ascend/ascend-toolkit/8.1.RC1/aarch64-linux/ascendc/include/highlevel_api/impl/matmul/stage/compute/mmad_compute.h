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
 * \file mmad_compute.h
 * \brief
 */
#ifndef IMPL_MATMUL_STAGE_COMPUTE_MMAD_COMPUTE_H
#define IMPL_MATMUL_STAGE_COMPUTE_MMAD_COMPUTE_H

#include "../../utils/matmul_module.h"
#include "../../utils/matmul_param.h"
#include "../../feature_trait/matmul_feature_trait.h"

namespace AscendC {
namespace Impl {
namespace Detail {

template <typename IMPL, typename C_T, typename A_T, typename B_T, const auto& MM_CFG>
class MmadCompute {
public:
    template <bool hasSpIdx = false>
    inline __aicore__ void Compute(const LocalTensor<C_T>& cMatrix, const LocalTensor<A_T>& l0A, const LocalTensor<B_T>& l0B,
                 uint16_t mmadM, uint16_t mmadK, uint16_t mmadN, bool isATrans, bool isBTrans,
                 uint8_t unitFlag = 0, bool cmatrixSource = false, bool cmatrixInitVal = true, bool isBias = false)
    {
        MmadParams mmadParams;
        mmadParams.m = mmadM;
        mmadParams.k = mmadK;
        mmadParams.n = mmadN;
        if constexpr (MatmulFeatureTrait<MM_CFG>::IsUnitFlagEnabled()) {
            mmadParams.unitFlag = unitFlag;
        }
        if constexpr (IsSameType<C_T, float>::value && IsSameType<A_T, float>::value) {
            if (isATrans) {
                mmadParams.kDirectionAlign = 1;
            }
        }

        if constexpr (!MatmulFeatureTrait<MM_CFG>::IsSupportCmatrixInitVal()) {
            mmadParams.cmatrixSource = cmatrixSource;
            mmadParams.cmatrixInitVal = cmatrixInitVal;
        } else {
            mmadParams.isBias = isBias;
        }
        if constexpr (hasSpIdx) {
            MmadWithSparse(cMatrix, l0A, l0B, mmadParams);
        } else {
            Mmad(cMatrix, l0A, l0B, mmadParams);
        }

        if ((mmadM / ALIGN_NUM) * (mmadN / ALIGN_NUM) < LIMIT_MNSIZE) { // add pipe_M required by aicore
            PipeBarrier<PIPE_M>();
        }
    }

private:
    static constexpr uint16_t LIMIT_MNSIZE = 10;
    static constexpr uint16_t ALIGN_NUM = 16;
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC

#endif // IMPL_MATMUL_STAGE_COMPUTE_MMAD_COMPUTE_H
