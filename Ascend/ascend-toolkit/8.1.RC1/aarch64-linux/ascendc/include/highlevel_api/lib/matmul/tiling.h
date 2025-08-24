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
 * \file tiling.h
 * \brief
 */
#ifndef LIB_MATMUL_TILING_H
#define LIB_MATMUL_TILING_H
#include "../../impl/matmul/utils/matmul_config_impl.h"

__aicore__ constexpr MatmulConfig GetNormalConfig(const bool intrinsicsLimit = false, const bool batchLoop = false,
    const bool isVecND2NZ = false, const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1,
    const bool isMsgReuse = true, const IterateOrder iterateOrder = IterateOrder::UNDEF,
    const ScheduleType scheduleType = ScheduleType::INNER_PRODUCT, const bool enUnitFlag = true,
    const bool enableMixDualMaster = false)
{
    return {
        .doNorm = true,
        .doBasicBlock = false,
        .doMultiDataLoad = false,
        .basicM = 0,
        .basicN = 0,
        .basicK = 0,
        .intrinsicsCheck = intrinsicsLimit,
        .isNBatch = batchLoop,
        .enVecND2NZ = isVecND2NZ,
        .doSpecialBasicBlock = false,
        .doMTE2Preload = 0,
        .singleCoreM = 0,
        .singleCoreN = 0,
        .singleCoreK = 0,
        .stepM = 0,
        .stepN = 0,
        .baseMN = 0,
        .singleCoreMN = 0,
        .enUnitFlag = enUnitFlag,
        .isPerTensor = false,
        .hasAntiQuantOffset = false,
        .doIBShareNorm = false,
        .doSpecialMDL = false,
        .enableInit = true,
        .batchMode = bmmMode,
        .enableEnd = true,
        .enableGetTensorC = true,
        .enableSetOrgShape = true,
        .enableSetBias = true,
        .enableSetTail = true,
        .enableQuantVector = true,
        .enableSetDefineData = true,
        .iterateMode = IterateMode::ITERATE_MODE_DEFAULT,
        .enableReuse = isMsgReuse,
        .enableUBReuse = true,
        .enableL1CacheUB = false,
        .intraBlockPartSum = false,
        .iterateOrder = iterateOrder,
        .scheduleType = scheduleType,
        .enableDoubleCache = false,
        .isBiasBatch = true,
        .enableStaticPadZeros = false,
        .isPartialOutput = false,
        .enableMixDualMaster = enableMixDualMaster,
        .isA2B2Shared = false,
        .isEnableChannelSplit = false, .enableKdimReorderLoad = false
    };
}

__aicore__ constexpr MatmulConfig GetMDLConfig(const bool intrinsicsLimit = false, const bool batchLoop = false,
    const uint32_t doMTE2Preload = 0, const bool isVecND2NZ = false, bool isPerTensor = false,
    bool hasAntiQuantOffset = false, const bool enUnitFlag = false, const bool isMsgReuse = true,
    const bool enableUBReuse = true, const bool enableL1CacheUB = false, const bool enableMixDualMaster = false, const bool enableKdimReorderLoad = false)
{
    return {
        .doNorm = false,
        .doBasicBlock = false,
        .doMultiDataLoad = true,
        .basicM = 0,
        .basicN = 0,
        .basicK = 0,
        .intrinsicsCheck = intrinsicsLimit,
        .isNBatch = batchLoop,
        .enVecND2NZ = isVecND2NZ,
        .doSpecialBasicBlock = false,
        .doMTE2Preload = doMTE2Preload,
        .singleCoreM = 0,
        .singleCoreN = 0,
        .singleCoreK = 0,
        .stepM = 0,
        .stepN = 0,
        .baseMN = 0,
        .singleCoreMN = 0,
        .enUnitFlag = enUnitFlag,
        .isPerTensor = isPerTensor,
        .hasAntiQuantOffset = hasAntiQuantOffset,
        .doIBShareNorm = false,
        .doSpecialMDL = false,
        .enableInit = true,
        .batchMode = BatchMode::NONE,
        .enableEnd = true,
        .enableGetTensorC = true,
        .enableSetOrgShape = true,
        .enableSetBias = true,
        .enableSetTail = true,
        .enableQuantVector = true,
        .enableSetDefineData = true,
        .iterateMode = IterateMode::ITERATE_MODE_DEFAULT,
        .enableReuse = isMsgReuse,
        .enableUBReuse = enableUBReuse,
        .enableL1CacheUB = enableL1CacheUB,
        .intraBlockPartSum = false,
        .iterateOrder = IterateOrder::UNDEF,
        .scheduleType = ScheduleType::INNER_PRODUCT,
        .enableDoubleCache = false,
        .isBiasBatch = true,
        .enableStaticPadZeros = false,
        .isPartialOutput = false,
        .enableMixDualMaster = enableMixDualMaster,
        .isA2B2Shared = false,
        .isEnableChannelSplit = false, .enableKdimReorderLoad = enableKdimReorderLoad
    };
}

__aicore__ constexpr MatmulConfig GetSpecialMDLConfig(const bool intrinsicsLimit = false, const bool batchLoop = false,
    const uint32_t doMTE2Preload = 0, const bool isVecND2NZ = false, bool isPerTensor = false,
    bool hasAntiQuantOffset = false)
{
    return {
        .doNorm = false,
        .doBasicBlock = false,
        .doMultiDataLoad = false,
        .basicM = 0,
        .basicN = 0,
        .basicK = 0,
        .intrinsicsCheck = intrinsicsLimit,
        .isNBatch = batchLoop,
        .enVecND2NZ = isVecND2NZ,
        .doSpecialBasicBlock = false,
        .doMTE2Preload = doMTE2Preload,
        .singleCoreM = 0,
        .singleCoreN = 0,
        .singleCoreK = 0,
        .stepM = 0,
        .stepN = 0,
        .baseMN = 0,
        .singleCoreMN = 0,
        .enUnitFlag = false,
        .isPerTensor = isPerTensor,
        .hasAntiQuantOffset = hasAntiQuantOffset,
        .doIBShareNorm = false,
        .doSpecialMDL = true,
        .enableInit = true,
        .batchMode = BatchMode::NONE,
        .enableEnd = true,
        .enableGetTensorC = true,
        .enableSetOrgShape = true,
        .enableSetBias = true,
        .enableSetTail = true,
        .enableQuantVector = true,
        .enableSetDefineData = true,
        .iterateMode = IterateMode::ITERATE_MODE_DEFAULT,
        .enableReuse = true,
        .enableUBReuse = true,
        .enableL1CacheUB = false,
        .intraBlockPartSum = false,
        .iterateOrder = IterateOrder::UNDEF,
        .scheduleType = ScheduleType::INNER_PRODUCT,
        .enableDoubleCache = false,
        .isBiasBatch = true,
        .enableStaticPadZeros = false,
        .isPartialOutput = false,
        .enableMixDualMaster = false,
        .isA2B2Shared = false,
        .isEnableChannelSplit = false, .enableKdimReorderLoad = false
    };
}

__aicore__ constexpr MatmulConfig GetBasicConfig(const uint32_t basicM, const uint32_t basicN,
    const uint32_t basicK, const bool intrinsicsLimit = false, const bool batchLoop = false,
    const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1)
{
    return {
        .doNorm = false,
        .doBasicBlock = true,
        .doMultiDataLoad = false,
        .basicM = basicM,
        .basicN = basicN,
        .basicK = basicK,
        .intrinsicsCheck = intrinsicsLimit,
        .isNBatch = batchLoop,
        .enVecND2NZ = false,
        .doSpecialBasicBlock = false,
        .doMTE2Preload = 0,
        .singleCoreM = 0,
        .singleCoreN = 0,
        .singleCoreK = 0,
        .stepM = 0,
        .stepN = 0,
        .baseMN = 0,
        .singleCoreMN = 0,
        .enUnitFlag = false,
        .isPerTensor = false,
        .hasAntiQuantOffset = false,
        .doIBShareNorm = false,
        .doSpecialMDL = false,
        .enableInit = true,
        .batchMode = bmmMode,
        .enableEnd = true,
        .enableGetTensorC = true,
        .enableSetOrgShape = true,
        .enableSetBias = true,
        .enableSetTail = true,
        .enableQuantVector = true,
        .enableSetDefineData = true,
        .iterateMode = IterateMode::ITERATE_MODE_DEFAULT,
        .enableReuse = true,
        .enableUBReuse = true,
        .enableL1CacheUB = false,
        .intraBlockPartSum = false,
        .iterateOrder = IterateOrder::UNDEF,
        .scheduleType = ScheduleType::INNER_PRODUCT,
        .enableDoubleCache = false,
        .isBiasBatch = true,
        .enableStaticPadZeros = false,
        .isPartialOutput = false,
        .enableMixDualMaster = false,
        .isA2B2Shared = false,
        .isEnableChannelSplit = false, .enableKdimReorderLoad = false
    };
}

__aicore__ constexpr MatmulConfig GetSpecialBasicConfig(const uint32_t basicM, const uint32_t basicN,
    const uint32_t basicK, const uint32_t singleCoreM, const uint32_t singleCoreN, const uint32_t singleCoreK,
    const uint32_t stepM, const uint32_t stepN, const bool intrinsicsLimit = false, const bool batchLoop = false,
    const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1)
{
    return {
        .doNorm = false,
        .doBasicBlock = false,
        .doMultiDataLoad = false,
        .basicM = basicM,
        .basicN = basicN,
        .basicK = basicK,
        .intrinsicsCheck = intrinsicsLimit,
        .isNBatch = batchLoop,
        .enVecND2NZ = false,
        .doSpecialBasicBlock = true,
        .doMTE2Preload = 0,
        .singleCoreM = singleCoreM,
        .singleCoreN = singleCoreN,
        .singleCoreK = singleCoreK,
        .stepM = stepM,
        .stepN = stepN,
        .baseMN = basicM * basicN,
        .singleCoreMN = singleCoreM * singleCoreN,
        .enUnitFlag = false,
        .isPerTensor = false,
        .hasAntiQuantOffset = false,
        .doIBShareNorm = false,
        .doSpecialMDL = false,
        .enableInit = true,
        .batchMode = bmmMode,
        .enableEnd = true,
        .enableGetTensorC = true,
        .enableSetOrgShape = true,
        .enableSetBias = true,
        .enableSetTail = true,
        .enableQuantVector = true,
        .enableSetDefineData = true,
        .iterateMode = IterateMode::ITERATE_MODE_DEFAULT,
        .enableReuse = true,
        .enableUBReuse = true,
        .enableL1CacheUB = false,
        .intraBlockPartSum = false,
        .iterateOrder = IterateOrder::UNDEF,
        .scheduleType = ScheduleType::INNER_PRODUCT,
        .enableDoubleCache = false,
        .isBiasBatch = true,
        .enableStaticPadZeros = false,
        .isPartialOutput = false,
        .enableMixDualMaster = false,
        .isA2B2Shared = false,
        .isEnableChannelSplit = false, .enableKdimReorderLoad = false
    };
}

__aicore__ constexpr MatmulConfig GetIBShareNormConfig(const bool intrinsicsLimit = false, const bool batchLoop = false,
    const bool isVecND2NZ = false, const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1,
    const bool isDoubleCache = false, const bool enUnitFlag = true)
{
    return {
        .doNorm = false,
        .doBasicBlock = false,
        .doMultiDataLoad = false,
        .basicM = 0,
        .basicN = 0,
        .basicK = 0,
        .intrinsicsCheck = intrinsicsLimit,
        .isNBatch = batchLoop,
        .enVecND2NZ = isVecND2NZ,
        .doSpecialBasicBlock = false,
        .doMTE2Preload = false,
        .singleCoreM = 0,
        .singleCoreN = 0,
        .singleCoreK = 0,
        .stepM = 0,
        .stepN = 0,
        .baseMN = 0,
        .singleCoreMN = 0,
        .enUnitFlag = enUnitFlag,
        .isPerTensor = false,
        .hasAntiQuantOffset = false,
        .doIBShareNorm = true,
        .doSpecialMDL = false,
        .enableInit = true,
        .batchMode = bmmMode,
        .enableEnd = true,
        .enableGetTensorC = true,
        .enableSetOrgShape = true,
        .enableSetBias = true,
        .enableSetTail = true,
        .enableQuantVector = true,
        .enableSetDefineData = true,
        .iterateMode = IterateMode::ITERATE_MODE_DEFAULT,
        .enableReuse = true,
        .enableUBReuse = true,
        .enableL1CacheUB = false,
        .intraBlockPartSum = false,
        .iterateOrder = IterateOrder::UNDEF,
        .scheduleType = ScheduleType::INNER_PRODUCT,
        .enableDoubleCache = isDoubleCache,
        .isBiasBatch = true,
        .enableStaticPadZeros = false,
        .isPartialOutput = false,
        .enableMixDualMaster = false,
        .isA2B2Shared = false,
        .isEnableChannelSplit = false, .enableKdimReorderLoad = false
    };
}

constexpr MatmulConfig CFG_NORM = GetNormalConfig();
constexpr MatmulConfig CFG_MDL = GetMDLConfig();
constexpr MatmulConfig MM_CFG_BB = GetBasicConfig(128, 128, 128);
constexpr MatmulConfig CFG_IBSHARE_NORM = GetIBShareNormConfig();

template <MatmulConfigMode configMode, typename... ArgTypes>
__aicore__ inline constexpr MatmulConfig GetMMConfig(ArgTypes&&... args) {
    MatmulConfig mmConfig = CFG_NORM;
    if constexpr (configMode == MatmulConfigMode::CONFIG_MDL) {
        mmConfig = CFG_MDL;
    } else if constexpr (configMode == MatmulConfigMode::CONFIG_SPECIALMDL) {
        mmConfig = GetSpecialMDLConfig();
    } else if constexpr (configMode == MatmulConfigMode::CONFIG_IBSHARE) {
        mmConfig = CFG_IBSHARE_NORM;
    }
    GetMMConfigImpl(mmConfig, args...);
    return mmConfig;
}

struct MatmulApiStaticTiling {
    int32_t usedCoreNum = -1;
    int32_t M = -1;
    int32_t N = -1;
    int32_t Ka = -1;
    int32_t Kb = -1;
    int32_t singleCoreM = -1;
    int32_t singleCoreN = -1;
    int32_t singleCoreK = -1;
    int32_t baseM = -1;
    int32_t baseN = -1;
    int32_t baseK = -1;
    int32_t depthA1 = -1;
    int32_t depthB1 = -1;
    int32_t stepM = -1;
    int32_t stepN = -1;
    int32_t isBias = -1;
    int32_t transLength = -1;
    int32_t iterateOrder = -1;
    int32_t shareMode = -1;
    int32_t shareL1Size = -1;
    int32_t shareL0CSize = -1;
    int32_t shareUbSize = -1;
    int32_t stepKa = -1;
    int32_t stepKb = -1;
    int32_t depthAL1CacheUB = -1;
    int32_t depthBL1CacheUB = -1;
    int32_t dbL0A = -1;
    int32_t dbL0B = -1;
    int32_t dbL0C = -1;
    int32_t ALayoutInfoB = -1;
    int32_t ALayoutInfoS = -1;
    int32_t ALayoutInfoN = -1;
    int32_t ALayoutInfoG = -1;
    int32_t ALayoutInfoD = -1;
    int32_t BLayoutInfoB = -1;
    int32_t BLayoutInfoS = -1;
    int32_t BLayoutInfoN = -1;
    int32_t BLayoutInfoG = -1;
    int32_t BLayoutInfoD = -1;
    int32_t CLayoutInfoB = -1;
    int32_t CLayoutInfoS1 = -1;
    int32_t CLayoutInfoN = -1;
    int32_t CLayoutInfoG = -1;
    int32_t CLayoutInfoS2 = -1;
    int32_t BatchNum = -1;
    MatmulConfig cfg = CFG_NORM;
};

#endif // LIB_MATMUL_TILING_H