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
 * \file matmul_config.h
 * \brief
 */
#ifndef LIB_MATMUL_MATMUL_CONFIG_H
#define LIB_MATMUL_MATMUL_CONFIG_H

#include "kernel_tiling/kernel_tiling.h"

#if ASCENDC_CPU_DEBUG
#define DEBUG_CODE(T) T
#else
#define DEBUG_CODE(T)
#endif

#define ITERATE_SIZE 2

enum class CubeFormat {
    ND = 0,
    NZ,
    ZN,
    ZZ,
    NN,
    ND_ALIGN,
    SCALAR,
    VECTOR,
};

enum class LayoutMode {
    NONE = 0,
    BSNGD,
    SBNGD,
    BNGS1S2,
    NORMAL
};

enum class BatchMode {
    NONE = 0,
    BATCH_LESS_THAN_L1,
    BATCH_LARGE_THAN_L1,
    SINGLE_LARGE_THAN_L1
};

enum class IterateOrder {
    ORDER_M = 0,
    ORDER_N,
    UNDEF,
};
 
enum class ScheduleType {
    INNER_PRODUCT = 0, // k loop, default type
    OUTER_PRODUCT,     // m/n loop, depends on IterateOrder
};

enum class MatmulVersion {
    NORMAL = 0,
    MULTI_DATA_LOAD,
    BASIC_BLOCK,
    IBSHARE_NORM,
};

enum IterateMode : uint8_t {
    ITERATE_MODE_NORMAL  = 0b00000001,
    ITERATE_MODE_ALL     = 0b00000010,
    ITERATE_MODE_BATCH   = 0b00000100,
    ITERATE_MODE_N_BATCH = 0b00001000,
    ITERATE_MODE_DEFAULT = 0b11111111,
};

struct MatmulConfig {
    bool doNorm;
    bool doBasicBlock;
    bool doMultiDataLoad;
    // basic MNK could only be valid in basic block mode
    uint32_t basicM;
    uint32_t basicN;
    uint32_t basicK;
    bool intrinsicsCheck;
    bool isNBatch;
    bool enVecND2NZ;
     // only be valid in special basic block mode
    bool doSpecialBasicBlock;
    uint32_t doMTE2Preload;
    uint32_t singleCoreM;
    uint32_t singleCoreN;
    uint32_t singleCoreK;
    uint32_t stepM;
    uint32_t stepN;
    uint32_t baseMN;
    uint32_t singleCoreMN;
    bool enUnitFlag = true;
    // AntiQuant Param
    bool isPerTensor;
    bool hasAntiQuantOffset;
    bool doIBShareNorm;
    // MDL support stepN == 2
    bool doSpecialMDL;
    bool enableInit = true;
    BatchMode batchMode;

    // Add for process performance
    bool enableEnd = true;
    bool enableGetTensorC = true;
    bool enableSetOrgShape = true;
    bool enableSetBias = true;
    bool enableSetTail = true;
    bool enableQuantVector = true;
    bool enableSetDefineData = true;
    uint8_t iterateMode = IterateMode::ITERATE_MODE_DEFAULT;
    bool enableReuse = true;
    // enable UB reuse(ND2NZ & ND2NZ) for V200
    bool enableUBReuse;
    bool enableL1CacheUB;
    // for intra-block l0c add
    bool intraBlockPartSum = false;
    // MDL support M/N db
    IterateOrder iterateOrder { IterateOrder::UNDEF };
    ScheduleType scheduleType;
    bool enableDoubleCache;
    bool isBiasBatch = true;
    bool enableStaticPadZeros = false;
    bool isPartialOutput = false;
    bool enableMixDualMaster = false;
    bool isA2B2Shared = false;
    bool isEnableChannelSplit = false;
    bool enableKdimReorderLoad = false;
};

enum class MatmulConfigMode {
    CONFIG_NORM,
    CONFIG_MDL,
    CONFIG_SPECIALMDL,
    CONFIG_IBSHARE
};

struct MatmulShapeParams {
    uint32_t singleCoreM;
    uint32_t singleCoreN;
    uint32_t singleCoreK;
    uint32_t basicM;
    uint32_t basicN;
    uint32_t basicK;
};

struct MatmulQuantParams {
    bool isPerTensor;
    bool hasAntiQuantOffset;
};

struct MatmulBatchParams {
    bool isNBatch;
    BatchMode batchMode;
    bool isBiasBatch = true;
};

struct MatmulFuncParams {
    bool intrinsicsCheck;
    bool enVecND2NZ;
    bool enableDoubleCache;
    bool enableL1CacheUB;
    uint32_t doMTE2Preload;
    IterateOrder iterateOrder;
    ScheduleType scheduleType;
    bool enableReuse = true;
    bool enableUBReuse;
    bool isPartialOutput = false;
    bool isA2B2Shared = false;
    bool isEnableChannelSplit = false;
    bool enableKdimReorderLoad = false;
};

struct MatrixOffset {
    int32_t offset;
    int32_t row, col;
    int32_t height, width;
};

extern int blockidx_;

#endif // LIB_MATMUL_MATMUL_CONFIG_H