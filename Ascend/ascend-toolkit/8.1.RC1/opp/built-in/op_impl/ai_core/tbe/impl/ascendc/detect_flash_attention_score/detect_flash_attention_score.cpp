/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file detect_flash_attention_score.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "detect_flash_attention_score_empty_tensor.h"
#include "detect_flash_attention_score_drop_mask_adapter.h"
#include "detect_flash_attention_score_s1s2_bn2gs1.h"
#include "detect_flash_attention_score_s1_bn2gs1.h"
#include "detect_flash_attention_score_bn2gs1s2_b.h"
#include "detect_flash_attention_var_len_score.h"

using namespace AscendC;

#ifdef __DAV_C220_CUBE__ // CUBE 实现

#define COPY_TILING_DATA(tiling)                                                                                       \
    GET_TILING_DATA_MEMBER(DetectFlashAttentionScoreGeneralTilingData, bmm1TilingData, bmm1TilingDataVar, tiling);           \
    GET_TILING_DATA_MEMBER(DetectFlashAttentionScoreGeneralTilingData, bmm2TilingData, bmm2TilingDataVar, tiling);           \
    const DetectFlashAttentionScoreGeneralTilingData *__restrict tilingData = nullptr;                                       \
    const TCubeTiling *__restrict bmm1tiling = &bmm1TilingDataVar;                                                     \
    const TCubeTiling *__restrict bmm2tiling = &bmm2TilingDataVar;

#define INVOKE_FA_GENERAL_OP_IMPL(templateClass, ...)                                                                  \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);                     \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(templateClass, ...)                                                           \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,           \
                          bmm2tiling);                                                                                 \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(templateClass, ...)                                                           \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling, op.bmm2Nz,           \
                          bmm2tiling);                                                                                \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(templateClass, ...)                                                          \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,           \
                          bmm2tiling);                                                                                 \
    } while (0)

#else // VECTOR 实现

#define COPY_TILING_DATA(tiling)                                                                                       \
    GET_TILING_DATA_WITH_STRUCT(DetectFlashAttentionScoreGeneralTilingData, tilingDataIn, tiling);                           \
    const DetectFlashAttentionScoreGeneralTilingData *__restrict tilingData = &tilingDataIn;                                 \
    const TCubeTiling *__restrict bmm1tiling = &(tilingData->bmm1TilingData);                                          \
    const TCubeTiling *__restrict bmm2tiling = &(tilingData->bmm2TilingData);

#define INVOKE_FA_GENERAL_OP_IMPL(templateClass, ...)                                                                  \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
            tPipe.Reset();                                                                                             \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);                 \
            op.Init(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, softmaxMax, softmaxSum,          \
                    softmaxOut, attentionOut, user, tilingData, &tPipe);                                               \
            op.Process();                                                                                              \
        } else {                                                                                                       \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);                 \
            op.Init(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, softmaxMax, softmaxSum,          \
                    softmaxOut, attentionOut, user, tilingData, &tPipe);                                               \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(templateClass, ...)                                                           \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
            tPipe.Destroy();                                                                                           \
            TPipe tPipeOp;                                                                                             \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipeOp, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,     \
                              bmm2tiling);                                                                             \
            op.Init(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, softmaxMax, softmaxSum,          \
                    softmaxOut, attentionOut, user, tilingData, &tPipeOp);                                             \
            op.Process();                                                                                              \
        } else {                                                                                                       \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,       \
                              bmm2tiling);                                                                             \
            op.Init(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, softmaxMax, softmaxSum,          \
                    softmaxOut, attentionOut, user, tilingData, &tPipe);                                               \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(templateClass, ...)                                                           \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
            tPipe.Destroy();                                                                                           \
            TPipe tPipeOp;                                                                                             \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipeOp, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling, op.bmm2Nz,       \
                              bmm2tiling);                                                                             \
            op.Init(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, softmaxMax, softmaxSum,          \
                    softmaxOut, attentionOut, user, tilingData, &tPipeOp);                                             \
            op.Process();                                                                                              \
        } else {                                                                                                       \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling, op.bmm2Nz,       \
                              bmm2tiling);                                                                             \
            op.Init(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, softmaxMax, softmaxSum,          \
                    softmaxOut, attentionOut, user, tilingData, &tPipe);                                               \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(templateClass, ...)                                                          \
    do {                                                                                                               \
        COPY_TILING_DATA(tiling);                                                                                      \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
        }                                                                                                              \
        tPipe.Reset();                                                                                                 \
        templateClass<__VA_ARGS__> op;                                                                                 \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,           \
                          bmm2tiling);                                                                                 \
        op.UnpackInit(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, actualSeqLengths,              \
                      actualSeqLengthsKv, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe); \
        op.Process();                                                                                                  \
    } while (0)

#endif

extern "C" __global__ __aicore__ void
detect_flash_attention_score(
                      __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                      __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
                      __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
                      __gm__ uint8_t *qStartIdx, __gm__ uint8_t *kvStartIdx, __gm__ uint8_t *softmaxMax,
                      __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut,
                      __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    set_mask_norm();

    if (TILING_KEY_IS(90) || TILING_KEY_IS(92) || TILING_KEY_IS(94)) {
        GET_TILING_DATA_WITH_STRUCT(DetectFlashAttentionScoreTilingData, tiling_data_in, tiling);
        const DetectFlashAttentionScoreTilingData *__restrict tiling_data = &tiling_data_in;
        if (TILING_KEY_IS(90)) {
            FlashAttentionScoreEmptyTensor<half> op;
            op.Init(softmaxMax, softmaxSum, attentionOut, tiling_data);
            op.Process();
        } else if (TILING_KEY_IS(92)) {
            FlashAttentionScoreEmptyTensor<float> op;
            op.Init(softmaxMax, softmaxSum, attentionOut, tiling_data);
            op.Process();
        } else if (TILING_KEY_IS(94)) {
            FlashAttentionScoreEmptyTensor<bfloat16_t> op;
            op.Init(softmaxMax, softmaxSum, attentionOut, tiling_data);
            op.Process();
        }
        return;
    }

#if (ORIG_DTYPE_QUERY == DT_FLOAT16)             // 3
    if (TILING_KEY_IS(10000001010220230943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION s1s2L1Reuse
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true, CubeFormat::ND,
                                  true);
        return;
    } else if (TILING_KEY_IS(10000000001021230099UL)) { // SplitBbDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, half, float, true, LayoutMode::SBNGD);
        return;
    } 
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16) // 2
    if (TILING_KEY_IS(10000000010220220943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_SBH, false, true, false, bfloat16_t, float, true);
        return;
    } 
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT) // 1
    if (TILING_KEY_IS(10000000000220110943UL)) { // SplitS1S2HighPerf: FLOAT32_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BSH, false, false, false, float, float, true);
        return;
    }
#endif
}
