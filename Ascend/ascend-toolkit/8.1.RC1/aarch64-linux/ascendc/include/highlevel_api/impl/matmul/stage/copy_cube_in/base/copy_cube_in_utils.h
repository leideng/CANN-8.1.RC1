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
* \file copy_cube_in_utils.h
* \brief
*/
#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_CUBE_IN_UTILS_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_CUBE_IN_UTILS_H

#include "../../../feature_trait/matmul_feature_trait.h"

namespace AscendC {
namespace Impl {
namespace Detail {
enum class CopyCubeInType : uint8_t {
    NONE = 0,
    NORMAL = 1,
    MDL,
    BMM,
    FROM_L1,
    SPARSE_MDL,
};

template <typename INPUT_TYPE, const auto& MM_CFG>
__aicore__ inline constexpr bool IsSameABTemplate()
{
    return DoMatmulIBShareNorm(MM_CFG) && INPUT_TYPE::ibShare;
}

template <typename INPUT_TYPE, const auto& MM_CFG>
__aicore__ inline constexpr bool IsCopyFromUB()
{
    return PhyPosIsUB(INPUT_TYPE::pos) && MatmulFeatureTrait<MM_CFG>().IsSupportUBToL1();
}

template <typename INPUT_TYPE, const auto& MM_CFG>
__aicore__ inline constexpr bool IsBMMFromL1()
{
    return PhyPosIsL1(INPUT_TYPE::pos) && (INPUT_TYPE::layout == LayoutMode::NORMAL) &&
           (ToMatmulConfig(MM_CFG).batchMode != BatchMode::SINGLE_LARGE_THAN_L1);
}

template <typename INPUT_TYPE, const auto& MM_CFG>
__aicore__ inline constexpr CopyCubeInType GetCopyCubeInType()
{
    if constexpr (PhyPosIsL1(INPUT_TYPE::pos)) {
        return CopyCubeInType::FROM_L1;
    } else if constexpr (DoMatmulIBShareNorm(MM_CFG)) {
        return CopyCubeInType::NORMAL;
    } else if constexpr (DoMatmulNorm(MM_CFG)) {
        if constexpr (INPUT_TYPE::layout != LayoutMode::NONE &&
            ToMatmulConfig(MM_CFG).batchMode != BatchMode::SINGLE_LARGE_THAN_L1) {
            return CopyCubeInType::BMM;
        } else {
            return CopyCubeInType::NORMAL;
        }
    } else if constexpr (DoMatmulMDL(MM_CFG) || DoMatmulSpecialMDL(MM_CFG)) {
        if constexpr (HasSparseIndex<INPUT_TYPE>()) {
            return CopyCubeInType::SPARSE_MDL;
        } else {
            return CopyCubeInType::MDL;
        }
    } else if constexpr (DoMatmulBasicBlock(MM_CFG) || DoMatmulSpecialBasicBlock(MM_CFG)) {
        return CopyCubeInType::NORMAL;
    } else {
        return CopyCubeInType::NONE;
    }
}

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _COPY_CUBE_IN_UTILS_H_