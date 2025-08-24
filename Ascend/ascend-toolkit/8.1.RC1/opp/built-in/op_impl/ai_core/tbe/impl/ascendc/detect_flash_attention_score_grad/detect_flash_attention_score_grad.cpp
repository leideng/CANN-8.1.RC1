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
 * \file detect_flash_attention_score_grad.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "detect_flash_attention_score_grad_empty_tensor.h"
#include "detect_flash_attention_score_grad_post.h"
#include "detect_flash_attention_score_grad_s1s2_bn2gs1s2.h"
#include "detect_flash_attention_score_grad_pre.h"
#include "detect_flash_attention_score_grad_sfmg.h"
#include "detect_flash_attention_score_grad_s1s2_bn2.h"
#include "detect_flash_attention_score_grad_ngs1s2_bn.h"
#include "detect_flash_attention_score_grad_bngs1s2_b.h"
#include "detect_flash_attention_score_grad_s1s2_bn2gs1s2_sab.h"

constexpr MatmulConfig MM_CFG_EXCEED = GetNormalConfig(true);
constexpr MatmulConfig MM_CFG_NORMAL = GetNormalConfig(false);
constexpr CubeFormat MM_NZ_OUT_FORMAT = CubeFormat::NZ;
constexpr CubeFormat MM_ND_OUT_FORMAT = CubeFormat::ND_ALIGN;
constexpr CubeFormat MM_ND_OUT_NOALIGN = CubeFormat::ND;
constexpr uint64_t INPUT_NONE = 0;
constexpr uint64_t INPUT_EXIST = 1;
constexpr uint32_t INPUT_DISABLE = 0;
constexpr uint32_t INPUT_ENABLE = 1;

constexpr static uint32_t ND = 0;
constexpr static uint32_t NZ = 1;

constexpr static const uint32_t BNGSD = 0;
constexpr static const uint32_t SBNGD = 1;
constexpr static const uint32_t BSNGD = 2;
constexpr static const uint32_t TND = 3;

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(INPUT_TYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT, \
                                              MM2_OUT_FORMAT)                                                          \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(DetectFlashAttentionScoreGradTilingDataS1s2Bn2gs1s2, tiling_data_in, tiling_data);       \
        const DetectFlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *__restrict tilingData = &tiling_data_in;                  \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1TilingData);                                       \
        const TCubeTiling *__restrict bmm2tiling = &(tilingData->mm2TilingData);                                       \
        const TCubeTiling *__restrict bmm3tiling = &(tilingData->mm3TilingData);                                       \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, DetectFlashAttentionScoreGradTilingDataS1s2Bn2gs1s2, true> opPre;      \
        opPre.Init(dq, dk, dv, drop_mask, user, tilingData, &pipeIn);                                                  \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
                                                                                                                       \
        TPipe pipeBase;                                                                                                \
        FlashAttentionScoreGradS1s2Bn2gs1s2<INPUT_TYPE, float, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,          \
                                            INPUT_LAYOUT, MM2_OUT_FORMAT>                                              \
            op;                                                                                                        \
        REGIST_MATMUL_OBJ(&pipeBase, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm3, bmm2tiling, op.mm4,             \
                          bmm3tiling);                                                                                 \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum,       \
                prefix, actual_seq_qlen, actual_seq_kvlen, dq, dk, dv, dpse, user, tilingData, &pipeBase);             \
        op.Process();                                                                                                  \
        op.SyncALLCores();                                                                                             \
        pipeBase.Destroy();                                                                                            \
        TPipe pipePost;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, DetectFlashAttentionScoreGradTilingDataS1s2Bn2gs1s2, true, INPUT_LAYOUT,     \
                                    input_format>                                                                      \
            opPost;                                                                                                    \
        opPost.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipePost);                       \
        opPost.Process();                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(INPUT_TYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,        \
                                              INPUT_LAYOUT, MM2_OUT_FORMAT, IS_DTM)                                    \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(DetectFlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb, tiling_data_in, tiling_data); \
        const DetectFlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb *__restrict tilingData = &tiling_data_in;            \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1TilingData);                                       \
        const TCubeTiling *__restrict bmm2tiling = &(tilingData->mm2TilingData);                                       \
        const TCubeTiling *__restrict bmm3tiling = &(tilingData->mm3TilingData);                                       \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, DetectFlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb, true> opPre;\
        opPre.Init(dq, dk, dv, drop_mask, user, tilingData, &pipeIn);                                                  \
        opPre.Process();                                                                                               \
        pipeIn.Destroy();                                                                                              \
        if ASCEND_IS_AIV {                                                                                             \
            TPipe pipeSfmg;                                                                                            \
            FlashAttentionScoreGradSfmg<INPUT_TYPE, float, DetectFlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb,        \
                INPUT_LAYOUT> opSfmg;                                                                                  \
            opSfmg.Init(dy, attention_in, actual_seq_qlen, dq, dk, dv, drop_mask, user, tilingData, &pipeSfmg);        \
            opSfmg.Process();                                                                                          \
            pipeSfmg.Destroy();                                                                                        \
        }                                                                                                              \
        TPipe pipeBase;                                                                                                \
        FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<INPUT_TYPE, float, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,    \
                                            INPUT_LAYOUT, MM2_OUT_FORMAT, IS_DTM> op;                                  \
        REGIST_MATMUL_OBJ(&pipeBase, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm3, bmm2tiling, op.mm4,             \
                          bmm3tiling);                                                                                 \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum,       \
                prefix, actual_seq_qlen, actual_seq_kvlen, dq, dk, dv, dpse, user, tilingData);                        \
        op.ProcessFirstMM();                                                                                           \
        op.InitBuffer(&pipeBase);                                                                                      \
        op.Process();                                                                                                  \
        op.SyncALLCores();                                                                                             \
        pipeBase.Destroy();                                                                                            \
        TPipe pipePost;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, DetectFlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb, true, INPUT_LAYOUT,\
                                    input_format>                                                                      \
            opPost;                                                                                                    \
        opPost.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipePost);                       \
        opPost.Process();                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(INPUT_TYPE, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,     \
                                         INPUT_LAYOUT, MM2_OUT_FORMAT)                                                 \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(DetectFlashAttentionScoreGradTilingDataS1s2Bn2, tiling_data_in, tiling_data);            \
        const DetectFlashAttentionScoreGradTilingDataS1s2Bn2 *__restrict tilingData = &tiling_data_in;                       \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1TilingData);                                       \
        const TCubeTiling *__restrict bmm31tiling = &(tilingData->mm31TilingData);                                     \
        const TCubeTiling *__restrict bmm4tiling = &(tilingData->mm4TilingData);                                       \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, DetectFlashAttentionScoreGradTilingDataS1s2Bn2, false> opPre;          \
        opPre.Init(dq, dk, dv, drop_mask, user, tilingData, &pipeIn);                                                  \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeOp;                                                                                                  \
        FlashAttentionScoreGradS1s2Bn2<INPUT_TYPE, float, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG,             \
                                       DROPOUT_CFG, INPUT_LAYOUT, MM2_OUT_FORMAT>                                      \
            op;                                                                                                        \
        REGIST_MATMUL_OBJ(&pipeOp, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm4, bmm4tiling, op.mm3_1,             \
                          bmm31tiling);                                                                                \
        op.Init(query, key, value, dy, pse_shift, drop_mask, padding_mask, atten_mask, softmax_max, softmax_sum,       \
                prefix, softmax_in, actual_seq_qlen, actual_seq_kvlen, attention_in, dq, dk, dv, dpse, user,           \
                tilingData, &pipeOp);                                                                                  \
        op.Process();                                                                                                  \
        pipeOp.Destroy();                                                                                              \
        TPipe pipeCast;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, DetectFlashAttentionScoreGradTilingDataS1s2Bn2, true, INPUT_LAYOUT,          \
        input_format> opCast;                                                                                          \
        opCast.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipeCast);                       \
        opCast.Process();                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(INPUT_TYPE, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG,       \
                                                    DROPOUT_CFG, INPUT_LAYOUT, MM2_OUT_FORMAT)                         \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(DetectFlashAttentionScoreGradTilingDataS1s2Bn2, tiling_data_in, tiling_data);            \
        const DetectFlashAttentionScoreGradTilingDataS1s2Bn2 *__restrict tilingData = &tiling_data_in;                       \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1TilingData);                                       \
        const TCubeTiling *__restrict bmm31tiling = &(tilingData->mm31TilingData);                                     \
        const TCubeTiling *__restrict bmm4tiling = &(tilingData->mm4TilingData);                                       \
        FlashAttentionScoreGradS1s2Bn2<INPUT_TYPE, float, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG,             \
                                       DROPOUT_CFG, INPUT_LAYOUT, MM2_OUT_FORMAT>                                      \
            op;                                                                                                        \
        REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm4, bmm4tiling, op.mm3_1,             \
                          bmm31tiling);                                                                                \
        op.Init(query, key, value, dy, pse_shift, drop_mask, padding_mask, atten_mask, softmax_max, softmax_sum,       \
                prefix, softmax_in, actual_seq_qlen, actual_seq_kvlen, attention_in, dq, dk, dv, dpse, user,           \
                tilingData, &pipeIn);                                                                                  \
        op.Process();                                                                                                  \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeCast;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, DetectFlashAttentionScoreGradTilingDataS1s2Bn2, true, INPUT_LAYOUT,          \
        input_format> opCast;                                                                                          \
        opCast.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipeCast);                       \
        opCast.Process();                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(INPUT_TYPE, INPUT_LAYOUT, layout, MM_CONFIG, MM_OUT_FORMAT,             \
                                               MM2_OUT_FORMAT)                                                         \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(DetectFlashAttentionScoreGradUbngs1s2BbTilingData, tiling_data_in, tiling_data);         \
        const DetectFlashAttentionScoreGradUbngs1s2BbTilingData *__restrict tilingData = &tiling_data_in;                    \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, DetectFlashAttentionScoreGradUbngs1s2BbTilingData, false> opPre;       \
        opPre.Init(dq, dk, dv, drop_mask, user, tilingData, &pipeIn);                                                  \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeOp;                                                                                                  \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1AndMm2TilingData);                                 \
        const TCubeTiling *__restrict bmm3tiling = &(tilingData->mm31TilingData);                                      \
        const TCubeTiling *__restrict bmm4tiling = &(tilingData->mm32AndMm4TilingData);                                \
        FlashAttentionScoreGradUngs1s2Bb<INPUT_TYPE, float, MM_CONFIG, INPUT_LAYOUT, MM_OUT_FORMAT, MM2_OUT_FORMAT> op;\
        REGIST_MATMUL_OBJ(&pipeOp, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm31, bmm3tiling,                      \
                          op.mm32, bmm4tiling, op.mm4, bmm4tiling);                                                    \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq,   \
                dk, dv, user, tilingData, &pipeOp);                                                                    \
        op.Process();                                                                                                  \
        pipeOp.Destroy();                                                                                              \
        TPipe pipeMuls;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, DetectFlashAttentionScoreGradUbngs1s2BbTilingData, false,                    \
        layout, input_format> opMuls;                                                                                  \
        opMuls.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipeMuls);                       \
        opMuls.Process();                                                                                              \
        pipeMuls.Destroy();                                                                                            \
    } while (0)

#define INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(INPUT_TYPE, INPUT_LAYOUT, layout, MM_CONFIG, MM_OUT_FORMAT, MM2_OUT_FORMAT)  \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(DetectFlashAttentionScoreGradTilingDataUngs1s2Bbn, tiling_data_in, tiling_data);         \
        const DetectFlashAttentionScoreGradTilingDataUngs1s2Bbn *__restrict tilingData = &tiling_data_in;                    \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, DetectFlashAttentionScoreGradTilingDataUngs1s2Bbn, false> opPre;       \
        opPre.Init(dq, dk, dv, drop_mask, user, tilingData, &pipeIn);                                                  \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeOp;                                                                                                  \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1AndMm2TilingData);                                 \
        const TCubeTiling *__restrict bmm3tiling = &(tilingData->mm31TilingData);                                      \
        const TCubeTiling *__restrict bmm4tiling = &(tilingData->mm32AndMm4TilingData);                                \
        FlashAttentionScoreGradUngs1s2Bbn<INPUT_TYPE, float, MM_CONFIG, true, INPUT_LAYOUT, MM_OUT_FORMAT,             \
                                          MM2_OUT_FORMAT> op;                                                          \
        REGIST_MATMUL_OBJ(&pipeOp, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm31, bmm3tiling,                      \
                          op.mm32, bmm4tiling, op.mm4, bmm4tiling);                                                    \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq,   \
                dk, dv, user, tilingData, &pipeOp);                                                                    \
        op.Process();                                                                                                  \
        pipeOp.Destroy();                                                                                              \
        TPipe pipeMuls;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, DetectFlashAttentionScoreGradTilingDataUngs1s2Bbn, false,                    \
        layout, input_format> opMuls;                                                                                  \
        opMuls.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipeMuls);                       \
        opMuls.Process();                                                                                              \
        pipeMuls.Destroy();                                                                                            \
    } while (0)

// implementation of kernel function
extern "C" __global__ __aicore__ void detect_flash_attention_score_grad(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dy, __gm__ uint8_t *pse_shift,
    __gm__ uint8_t *drop_mask, __gm__ uint8_t *padding_mask, __gm__ uint8_t *atten_mask, __gm__ uint8_t *softmax_max,
    __gm__ uint8_t *softmax_sum, __gm__ uint8_t *softmax_in, __gm__ uint8_t *attention_in, __gm__ uint8_t *prefix,
    __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen, __gm__ uint8_t *q_start_idx,
    __gm__ uint8_t *kv_start_idx, __gm__ uint8_t *dq, __gm__ uint8_t *dk,
    __gm__ uint8_t *dv, __gm__ uint8_t *dpse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling_data)
{
    TPipe pipeIn;
    set_mask_norm();
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

// --------------------------------------------float16 tilingkey------------------------------------------------------
#if (ORIG_DTYPE_QUERY == DT_FLOAT16)
    // -----------------------SameAB start---------------------------------
    if (TILING_KEY_IS(10000001000111003434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                                     MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    }
    // -----------------------SameAB end---------------------------------

    // -----------------------1.1 start---------------------------------
    else if (TILING_KEY_IS(10000000000111033434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    }
    // -----------------------1.1 end---------------------------------

    // -----------------------1.2 start---------------------------
    else if (TILING_KEY_IS(10000000000010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse单独存在
    }
    // -----------------------1.2 end---------------------------------

    // -----------------------3.1 start---------------------------------
    else if (TILING_KEY_IS(10000000000000003199UL)) { // BSH BSNGD & FLOAT16_PRECISION
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                          MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    }
    // -----------------------3.1 end---------------------------------

    // -----------------------4.1 start---------------------------------
    else if (TILING_KEY_IS(10000000000000103099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                               MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    }
    // -----------------------4.1 end---------------------------------
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16)
    // -----------------------SameAB start---------------------------------
    if (TILING_KEY_IS(10000001100100032434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                                     TND, MM_ND_OUT_NOALIGN, INPUT_ENABLE);
        return;
    }
    // -----------------------1.1 start---------------------------------
    else if (TILING_KEY_IS(10000000001100032434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    }
    // -----------------------1.1 end---------------------------------

    // -----------------------1.2 start---------------------------
    else if (TILING_KEY_IS(10000000000010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_FORMAT);
        return;
        // pse单独存在
    }
    // -----------------------1.2 end---------------------------------

    // -----------------------3.1 start---------------------------------
    else if (TILING_KEY_IS(10000000000000002199UL)) { // BSH BSNGD & BFLOAT16
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                          MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    }
    // -----------------------3.1 end---------------------------------

    // -----------------------4.1 start---------------------------------
    else if (TILING_KEY_IS(10000000000000102099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                               MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    }
// -----------------------4.1 end---------------------------------
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT)
    if (TILING_KEY_IS(10000001000000001434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                                     MM_ND_OUT_NOALIGN, BSNGD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    }
    // -----------------------1.1 start---------------------------------
    else if (TILING_KEY_IS(10000000000111001434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    }
    // -----------------------1.1 end---------------------------------

    // -----------------------1.2 start---------------------------
    else if (TILING_KEY_IS(10000000000000001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_NOALIGN, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse单独存在
    }
    // -----------------------1.2 end---------------------------------

    // -----------------------3.1 start---------------------------------
    else if (TILING_KEY_IS(10000000000000001199UL)) { // BSH BSNGD & FLOAT16_PRECISION
        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(float, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                          MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    }
    // -----------------------3.1 end---------------------------------

    // -----------------------4.1 start---------------------------------
    else if (TILING_KEY_IS(10000000000000101099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(float, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                               MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN);
        return;
    }
// -----------------------4.1 end---------------------------------
#endif

    GET_TILING_DATA_WITH_STRUCT(DetectFlashAttentionScoreGradTilingData, tiling_data_in, tiling_data);
    const DetectFlashAttentionScoreGradTilingData *__restrict tilingData = &tiling_data_in;

    if (TILING_KEY_IS(90)) {
        FlashAttentionScoreGradEmptyTensor<DTYPE_DQ> op;
        op.Init(dq, dk, dv, dpse, tilingData);
        op.Process();
    }
}
