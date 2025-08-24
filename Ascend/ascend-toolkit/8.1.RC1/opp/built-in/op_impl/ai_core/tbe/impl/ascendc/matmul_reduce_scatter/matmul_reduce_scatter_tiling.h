/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file matmul_reduce_scatter_tiling.h
 * \brief
 */
#ifndef MATMUL_REDUCE_SCATTER_TILING_H
#define MATMUL_REDUCE_SCATTER_TILING_H

#include "kernel_tiling/kernel_tiling.h"
#include "../common/mc2_tiling_struct.h"

struct ReduceScatterSoc {
    uint32_t commAlg;
	uint32_t isA3;
    uint32_t isStep;
    uint32_t isND2NZ;
    uint32_t baseBD;
    uint32_t baseBN;
};

struct L2cacheUseInfo {
    uint32_t l2CacheFlag;
};

class MatmulReduceScatterTilingData {
public:
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    RCSTiling param;
    TCubeTiling tileTiling;
    TCubeTiling tailTiling;
    TCubeTiling localTiling;
    TileL2Tiling tileL2Tiling;
    TileL2Tiling tailL2Tiling;
    TileL2Tiling localL2Tiling;
	ReduceScatterSoc socParam;
    L2cacheUseInfo l2cacheUseInfo;
};

#endif //__MATMUL_REDUCE_SCATTER_TILING_H__