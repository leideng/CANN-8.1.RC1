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
* \file matmul_chip_cap.h
* \brief
*/

#ifndef IMPL_MATMUL_FEATURE_TRAIT_MATMUL_CHIP_CAP_H
#define IMPL_MATMUL_FEATURE_TRAIT_MATMUL_CHIP_CAP_H

namespace AscendC {
namespace Impl {
namespace Detail {

enum class FixpipeParamsType: int8_t {
    V220,
    V300,
    V310,
    NONE
};

class MatmulChipCap
{
public:
    struct Feature {
        bool supportUnitFlag;
        bool ifNeedUB;
        bool ifSupportUBToL1;
        bool supportMNL0DB;
        FixpipeParamsType fixpipeParamsType;
        bool ifSupportLoad3dV2;
        bool ifSupportLoad2dTranspose;
        bool ifSupportLoad2dV2;
        bool ifSupportCmatrixInitVal;
        bool ifSupportFmatrixB;
        bool ifSupportUserDefine;
    };

    __aicore__ constexpr static const Feature& GetFeatures()
    {
        return features[GetChipType()];
    }

private:
    enum {
        CHIP_TYPE_100,
        CHIP_TYPE_200,
        CHIP_TYPE_220,
        CHIP_TYPE_300,
        CHIP_TYPE_310,
        CHIP_TYPE_MAX,
    };

    __aicore__ inline constexpr static uint8_t GetChipType()
    {
        #if __CCE_AICORE__ == 100
            return CHIP_TYPE_100;
        #elif __CCE_AICORE__ == 200
            return CHIP_TYPE_200;
        #elif __CCE_AICORE__ == 220
            return CHIP_TYPE_220;
        #elif __CCE_AICORE__ == 300
            return CHIP_TYPE_300;
        #elif __CCE_AICORE__ == 310
            return CHIP_TYPE_310;
        #else
            return CHIP_TYPE_MAX;
        #endif
    }

private:
    constexpr static Feature features[CHIP_TYPE_MAX] = {
        /*       supportUnitFlag, ifNeedUB, ifSupportUBToL1, supportMNL0DB, fixpipeParamsType, ifSupportLoad3dV2, ifSupportLoad2dTranspose, ifSupportLoad2dV2, ifOnlyUseIsBiasForMmad, ifSupportFmatrixB, ifSupportUserDefine*/
        /*100*/ {false, true, true, false, FixpipeParamsType::NONE, false, false, false, true, false, false},
        /*200*/ {false, true, true, false, FixpipeParamsType::NONE, true, false, false, false, false, false},
        /*220*/ {true, false, false, true, FixpipeParamsType::V220, true, true, false, false, true, true},
        /*300*/ {true, false, true, false, FixpipeParamsType::V220, true, true, false, false, true, false},
        /*310*/ {true, false, false, false, FixpipeParamsType::V310, true, true, true, false, true, false}};
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _MATMUL_CHIP_CAP_H_
