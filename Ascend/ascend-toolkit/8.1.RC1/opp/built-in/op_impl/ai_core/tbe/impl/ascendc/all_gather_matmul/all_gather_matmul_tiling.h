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
 * \file all_gather_matmul_tiling.h
 * \brief
 */

#ifndef __ALL_GATHER_MATMUL_TILING_H__
#define __ALL_GATHER_MATMUL_TILING_H__

#pragma once
#include "kernel_tiling/kernel_tiling.h"
#include "../common/mc2_tiling_struct.h"

struct AllGatherSoc {
    uint32_t commAlg;
    uint32_t isA3;
    uint32_t isStep;
    uint32_t isND2NZ;
};

class AllGatherMatmulTilingData {
    public:
        Mc2InitTiling mc2InitTiling;
        Mc2CcTiling mc2CcTiling;
        TCubeTiling tileTiling;
        TCubeTiling tailTiling;
        TCubeTiling localTiling;
        TileL2Tiling tileL2Tiling;
        TileL2Tiling tailL2Tiling;
        TileL2Tiling localL2Tiling;
        RCSTiling param;
        AllGatherSoc socParam;
};

#endif //__ALL_GATHER_MATMUL_TILING_H__