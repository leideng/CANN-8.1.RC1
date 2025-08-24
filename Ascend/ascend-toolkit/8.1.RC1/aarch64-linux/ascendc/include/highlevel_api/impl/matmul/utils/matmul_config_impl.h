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
 * \file matmul_config_impl.h
 * \brief
 */
#ifndef IMPL_MATMUL_UTILS_MATMUL_CONFIG_IMPL_H
#define IMPL_MATMUL_UTILS_MATMUL_CONFIG_IMPL_H

#include "lib/matmul/matmul_config.h"

template <typename ArgType>
__aicore__ inline constexpr void GetMMConfigImpl(MatmulConfig& cfg, ArgType arg) {
    if constexpr (AscendC::IsSameType<ArgType, MatmulShapeParams>::value) {
        cfg.singleCoreM = arg.singleCoreM;
        cfg.singleCoreN = arg.singleCoreN;
        cfg.singleCoreK = arg.singleCoreK;
        cfg.basicM = arg.basicM;
        cfg.basicN = arg.basicN;
        cfg.basicK = arg.basicK;
    } else if constexpr (AscendC::IsSameType<ArgType, MatmulQuantParams>::value) {
        cfg.isPerTensor = arg.isPerTensor;
        cfg.hasAntiQuantOffset = arg.hasAntiQuantOffset;
    } else if constexpr (AscendC::IsSameType<ArgType, MatmulBatchParams>::value) {
        cfg.isNBatch = arg.isNBatch;
        cfg.batchMode = arg.batchMode;
        cfg.isBiasBatch = arg.isBiasBatch;
    } else if constexpr (AscendC::IsSameType<ArgType, MatmulFuncParams>::value) {
        cfg.intrinsicsCheck = arg.intrinsicsCheck;
        cfg.enVecND2NZ = arg.enVecND2NZ;
        cfg.enableDoubleCache = arg.enableDoubleCache;
        cfg.enableL1CacheUB = arg.enableL1CacheUB;
        cfg.doMTE2Preload = arg.doMTE2Preload;
        cfg.iterateOrder = arg.iterateOrder;
        cfg.scheduleType = arg.scheduleType;
        cfg.enableReuse = arg.enableReuse;
        cfg.enableUBReuse = arg.enableUBReuse;
        cfg.isA2B2Shared = arg.isA2B2Shared;
        cfg.isEnableChannelSplit = arg.isEnableChannelSplit;
        cfg.enableKdimReorderLoad = arg.enableKdimReorderLoad;
    }
}

template <typename T, typename... ArgTypes>
__aicore__ inline constexpr void GetMMConfigImpl(MatmulConfig& cfg, T arg, ArgTypes&&... args) {
    GetMMConfigImpl(cfg, arg);
    GetMMConfigImpl(cfg, args...);
}

#endif // _MATMUL_CONFIG_IMPL_H_