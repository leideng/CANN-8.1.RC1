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
* \file matmul_feature_trait.h
* \brief
*/
#ifndef IMPL_MATMUL_FEATURE_TRAIT_MATMUL_FEATURE_TRAIT_H
#define IMPL_MATMUL_FEATURE_TRAIT_MATMUL_FEATURE_TRAIT_H

#include "../utils/matmul_utils.h"
#include "matmul_chip_cap.h"

namespace AscendC {
namespace Impl {
namespace Detail {

template <const auto& MM_CFG = CFG_NORM>
class MatmulFeatureTrait {
public:
    __aicore__ inline constexpr static bool IsUnitFlagEnabled()
    {
        return EnUnitFlag(MM_CFG) && MatmulChipCap::GetFeatures().supportUnitFlag;
    }

    __aicore__ inline constexpr static bool IsNeedUB()
    {
        return MatmulChipCap::GetFeatures().ifNeedUB;
    }

    __aicore__ inline constexpr static bool IsSupportUBToL1()
    {
        return MatmulChipCap::GetFeatures().ifSupportUBToL1;
    }

    __aicore__ inline constexpr static bool IsSupportMNL0DB()
    {
        // only V220 and MDL and NORM scene support L0DB
        return MatmulChipCap::GetFeatures().supportMNL0DB &&
               ToMatmulConfig(MM_CFG).scheduleType == ScheduleType::OUTER_PRODUCT &&
               (DoMatmulMDL(MM_CFG) || DoMatmulNorm(MM_CFG));
    }
    __aicore__ inline constexpr static FixpipeParamsType GetFixpipeParamsType()
    {
        return MatmulChipCap::GetFeatures().fixpipeParamsType;
    }

    __aicore__ inline constexpr static bool IsSupportLoad3dV2()
    {
        return MatmulChipCap::GetFeatures().ifSupportLoad3dV2;
    }

    __aicore__ inline constexpr static bool IsSupportLoad2dTranspose()
    {
        return MatmulChipCap::GetFeatures().ifSupportLoad2dTranspose;
    }

    __aicore__ inline constexpr static bool IsSupportLoad2dV2()
    {
        return MatmulChipCap::GetFeatures().ifSupportLoad2dV2;
    }

    __aicore__ inline constexpr static bool IsSupportCmatrixInitVal()
    {
        return MatmulChipCap::GetFeatures().ifSupportCmatrixInitVal;
    }

    __aicore__ inline constexpr static bool IsSupportFmatrixB()
    {
        return MatmulChipCap::GetFeatures().ifSupportFmatrixB;
    }

    __aicore__ inline constexpr static bool IsSupportUserDefine()
    {
        return MatmulChipCap::GetFeatures().ifSupportUserDefine;
    }
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _MATMUL_FEATURE_TRAIT_H_
