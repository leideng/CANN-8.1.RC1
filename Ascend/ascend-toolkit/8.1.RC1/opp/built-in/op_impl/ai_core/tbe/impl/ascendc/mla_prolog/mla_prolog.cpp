/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License. 
 */

/*!
 * \file mla_prolog.cpp
 * \brief
 */

#include "kernel_mla_prolog_vec_s1_cub_s2.h"

using namespace MlaProlog; 

#ifdef __DAV_C220_CUBE__
#define COPY_TILING_DATA_BASE_PARAMS(tiling)                                                                           \
    GET_TILING_DATA_MEMBER(MlaPrologTilingData, baseParams, tilingDataIn, tiling);                                            \
    const MlaPrologTilingData* __restrict tilingData = nullptr;                                                  \
    const MlaPrologBaseParams *__restrict tilingDataBaseParams = &tilingDataIn;
#else
#define COPY_TILING_DATA_BASE_PARAMS(tiling)                                                                           \
    GET_TILING_DATA_MEMBER(MlaPrologTilingData, baseParams, tilingDataIn, tiling);                                     \
    const MlaPrologTilingData* __restrict tilingData = nullptr;                                                        \
    const MlaPrologBaseParams *__restrict tilingDataBaseParams = &tilingDataIn;
#endif

#define INVOKE_MLA_PROLOG_NO_KFC_OP_IMPL(templateClass, ...)                                                           \
    do {                                                                                                               \
        COPY_TILING_DATA_BASE_PARAMS(tiling);                                                                          \
        templateClass<MLAPType<__VA_ARGS__>> op(&pipe, tilingData, tilingDataBaseParams);                              \
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,         \
                ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,        \
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, queryOut, queryRopeOut, workspace);  \
        op.Process();                                                                                                  \
    } while (0)

extern "C" __global__ __aicore__ void mla_prolog(__gm__ uint8_t *tokenX, __gm__ uint8_t *weightDq,
                                                 __gm__ uint8_t *weightUqQr, __gm__ uint8_t *weightUk,
                                                 __gm__ uint8_t *weightDkvKr, __gm__ uint8_t *rmsnormGammaCq,
                                                 __gm__ uint8_t *rmsnormGammaCkv, __gm__ uint8_t *ropeSin,
                                                 __gm__ uint8_t *ropeCos, __gm__ uint8_t *cacheIndex,
                                                 __gm__ uint8_t *kvCache, __gm__ uint8_t *krCache,
                                                 __gm__ uint8_t *dequantScaleX, __gm__ uint8_t *dequantScaleWDq,
                                                 __gm__ uint8_t *dequantScaleWUqQr, __gm__ uint8_t *dequantScaleWDkvKr,
                                                 __gm__ uint8_t *quantScaleCkv, __gm__ uint8_t *quantScaleCkr,
                                                 __gm__ uint8_t *smoothScalesCq,
                                                 __gm__ uint8_t *queryOut, __gm__ uint8_t *queryRopeOut,
                                                 __gm__ uint8_t *kvCacheOut, __gm__ uint8_t *krCacheOut,
                                                 __gm__ uint8_t *workspace, __gm__ uint8_t *tiling) {

    TPipe pipe;
    // 个位代表 CACHE_MOD 0-CACHE_MODE_BNSD   1-PA_BSND    2-PA_NZ
    // 十位代表场景    0-FP16(预留)     1-BF16      2-量化场景
    // 百位代表量化场景     0-MMQcQr量化    1-MMQcQr量化+KVcache量化
    // MlaProlog<inputType1, inputType2, input3Type3, CACHE_MODE>
    // inputType1: token_x weight_dq weight_dkv_kr
    // inputType2: weight_uq_qr
    // input3Type3: kv_cache kr_cache

    if (TILING_KEY_IS(10000000000000010)) {
        // input BF16, cache_mode BNSD
        INVOKE_MLA_PROLOG_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, bfloat16_t, bfloat16_t, CACHE_MODE::CACHE_MODE_BNSD);
    } else if (TILING_KEY_IS(10000000000000011)) {
        // input BF16, cache_mode PA_BSND
        INVOKE_MLA_PROLOG_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, bfloat16_t, bfloat16_t, CACHE_MODE::CACHE_MODE_PA_BSND);
    } else if (TILING_KEY_IS(10000000000000012)) {
        // input BF16, cache_mode PA_NZ
        INVOKE_MLA_PROLOG_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, bfloat16_t, bfloat16_t, CACHE_MODE::CACHE_MODE_PA_NZ);
    } else if (TILING_KEY_IS(10000000000000020)) {
        // quant scenario MMQcQr量化, cache_mode BNSD
        INVOKE_MLA_PROLOG_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, bfloat16_t, CACHE_MODE::CACHE_MODE_BNSD);
    } else if (TILING_KEY_IS(10000000000000021)) {
        // quant scenario MMQcQr量化, cache_mode PA_BSND
        INVOKE_MLA_PROLOG_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, bfloat16_t, CACHE_MODE::CACHE_MODE_PA_BSND);
    } else if (TILING_KEY_IS(10000000000000022)) {
        // quant scenario MMQcQr量化, cache_mode PA_NZ
        INVOKE_MLA_PROLOG_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, bfloat16_t, CACHE_MODE::CACHE_MODE_PA_NZ);
    } else if (TILING_KEY_IS(10000000000000120)) {
        // quant scenario MMQcQr量化+KVcache量化, cache_mode BNSD
        INVOKE_MLA_PROLOG_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, int8_t, CACHE_MODE::CACHE_MODE_BNSD);
    } else if (TILING_KEY_IS(10000000000000121)) {
        // quant scenario MMQcQr量化+KVcache量化, cache_mode PA_BSND
        INVOKE_MLA_PROLOG_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, int8_t, CACHE_MODE::CACHE_MODE_PA_BSND);
    } else if (TILING_KEY_IS(10000000000000122)) {
        // quant scenario MMQcQr量化+KVcache量化, cache_mode PA_NZ
        INVOKE_MLA_PROLOG_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, int8_t, CACHE_MODE::CACHE_MODE_PA_NZ);
    }   
}