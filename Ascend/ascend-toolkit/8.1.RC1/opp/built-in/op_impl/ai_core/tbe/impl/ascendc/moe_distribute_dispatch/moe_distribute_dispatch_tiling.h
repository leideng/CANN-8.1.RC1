/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file moe_distribute_dispatch_tiling.h
 * \brief
 */

#ifndef ASCENDC_MOE_DISTRIBUTE_DISPATCH_TILING_H
#define ASCENDC_MOE_DISTRIBUTE_DISPATCH_TILING_H
struct MoeDistributeDispatchA2Info {
    uint32_t epWorldSize;                // epWorldSize
    uint32_t tpWorldSize;                // tpWorldSize
    uint32_t epRankId;                   // epRankId
    uint32_t tpRankId;                   // tpRankId
    uint32_t expertSharedType;           // expert type
    uint32_t sharedExpertRankNum;        // shared expert number
    uint32_t moeExpertNum;               // moe expert number
    uint32_t quantMode;                  // quant mode
    uint32_t globalBs;                   // globalBs = BS * worldSize
    uint32_t bs;                         // bs
    uint32_t k;                          // k
    uint32_t h;                          // h
    uint32_t aivNum;                     // aivNum
    bool isQuant;                        // whether quant or not
    bool reserved1;                      // reserved
    bool reserved2;                      // reserved
    bool reserved3;                      // reserved
    uint64_t totalUbSize;                // epWorldSize
    uint32_t hcclBufferSize;             // HCCL windows, unit:B
    uint32_t expertTokenNumsType;        // expert token nums type, support 0: cumsum mode, 1: count mode
};

struct MoeDistributeDispatchA2TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MoeDistributeDispatchA2Info moeDistributeDispatchInfo;
};

struct MoeDistributeDispatchInfo {
    uint32_t epWorldSize;                // epWorldSize
    uint32_t tpWorldSize;                // tpWorldSize
    uint32_t epRankId;                   // epRankId
    uint32_t tpRankId;                   // tpRankId
    uint32_t expertShardType;            // expert type
    uint32_t sharedExpertRankNum;        // shared expert number
    uint32_t moeExpertNum;               // moe expert number
    uint32_t quantMode;                  // quant mode
    uint32_t globalBs;                   // globalBs = BS * worldSize
    uint32_t bs;                         // bs
    uint32_t k;                          // k
    uint32_t h;                          // h
    uint32_t aivNum;                     // aivNum
    bool isQuant;                        // whether quant or not
    bool reserved1;                      // reserved
    bool reserved2;                      // reserved
    bool reserved3;                      // reserved
    uint64_t totalUbSize;                // epWorldSize
    uint64_t totalWinSize;
    uint32_t expertTokenNumsType;        // expert token nums type, support 0: cumsum mode, 1: count mode
};

struct MoeDistributeDispatchTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    Mc2CcTiling mc2CcTiling2;
    MoeDistributeDispatchInfo moeDistributeDispatchInfo;
};

#endif