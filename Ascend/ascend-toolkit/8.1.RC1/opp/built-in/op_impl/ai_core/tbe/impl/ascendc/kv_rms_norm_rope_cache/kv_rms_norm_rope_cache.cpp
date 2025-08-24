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
 * \file kv_rms_norm_rope_cache.cpp
 * \brief
 */

#include "kv_rms_norm_rope_cache_b16_mtp.h"
#include "kv_rms_norm_rope_cache_b16_b1sd.h"
#include "kv_rms_norm_rope_cache_b16_pa_blk_nz.h"
#include "kv_rms_norm_rope_cache_b16_pa_bnsd_quant.h"
#include "kv_rms_norm_rope_cache_b16_pa_blk_bnsd_quant.h"
#include "kv_rms_norm_rope_cache_b16_pa_nz_quant.h"
#include "kv_rms_norm_rope_cache_b16_pa_blk_nz_quant.h"
#include "kv_rms_norm_rope_cache_b16_pa.h"

using namespace KvRmsNormRopeCache;
#define KV_RMS_NORM_ROPE_CACHE_B16_NORM 1000
#define KV_RMS_NORM_ROPE_CACHE_B16_NORM_MTP 1001
#define KV_RMS_NORM_ROPE_CACHE_B16_PA 2000
#define KV_RMS_NORM_ROPE_CACHE_B16_PA_MTP 2001
#define KV_RMS_NORM_ROPE_CACHE_B16_B1SD_NORM 3000
#define KV_RMS_NORM_ROPE_CACHE_B16_B1SD_PA 3001
#define KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_NZ 4000
#define KV_RMS_NORM_ROPE_CACHE_B16_PA_NZ 4001
#define KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_BNSD 5000
#define KV_RMS_NORM_ROPE_CACHE_B16_PA_BNSD 5001
#define KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_NZ_QUANT 4010
#define KV_RMS_NORM_ROPE_CACHE_B16_PA_NZ_QUANT 4011
#define KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_BNSD_QUANT 5010
#define KV_RMS_NORM_ROPE_CACHE_B16_PA_BNSD_QUANT 5011

extern "C" __global__ __aicore__ void kv_rms_norm_rope_cache(GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin,
                                                             GM_ADDR index, GM_ADDR k_cache, GM_ADDR ckv_cache,
                                                             GM_ADDR k_rope_scale, GM_ADDR c_kv_scale, GM_ADDR k_rope_offset, GM_ADDR c_kv_offset,
                                                             GM_ADDR k_cache_out, GM_ADDR c_kv_offset_out,
                                                             GM_ADDR k_rope, GM_ADDR c_kv,
                                                             GM_ADDR workspace, GM_ADDR tiling) {
  TPipe pipe;
  GET_TILING_DATA(tilingData, tiling);
  if (TILING_KEY_IS(KV_RMS_NORM_ROPE_CACHE_B16_NORM)) {
    KernelKvRmsNormRopeCacheB16MTP<false, DTYPE_KV> op(&pipe, &tilingData);
    op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache);
    op.Process();
  } else if (TILING_KEY_IS(KV_RMS_NORM_ROPE_CACHE_B16_NORM_MTP)) {
    KernelKvRmsNormRopeCacheB16MTP<false, DTYPE_KV> op(&pipe, &tilingData);
    op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache);
    op.Process();
  } else if (TILING_KEY_IS(KV_RMS_NORM_ROPE_CACHE_B16_PA)) {
    KernelKvRmsNormRopeCacheB16MTP<true, DTYPE_KV> op(&pipe, &tilingData);
    op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache);
    op.Process();
  } else if (TILING_KEY_IS(KV_RMS_NORM_ROPE_CACHE_B16_PA_MTP)) {
    KernelKvRmsNormRopeCacheB16MTP<true, DTYPE_KV> op(&pipe, &tilingData);
    op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache);
    op.Process();
  } else if (TILING_KEY_IS(KV_RMS_NORM_ROPE_CACHE_B16_B1SD_NORM)) {
    KernelKvRmsNormRopeCacheB16B1SD<false, DTYPE_KV> op(&pipe, &tilingData);
    op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache);
    op.Process();
  } else if (TILING_KEY_IS(KV_RMS_NORM_ROPE_CACHE_B16_B1SD_PA)) {
    KernelKvRmsNormRopeCacheB16B1SD<true, DTYPE_KV> op(&pipe, &tilingData);
    op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache);
    op.Process();
  } else if (TILING_KEY_IS(KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_NZ)) {
    KernelKvRmsNormRopeCacheB16PABLKNZ<true, DTYPE_KV> op(&pipe, &tilingData);
    op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope, c_kv);
    op.Process();
  } else if (TILING_KEY_IS(KV_RMS_NORM_ROPE_CACHE_B16_PA_NZ_QUANT)) {
    KernelKvRmsNormRopeCacheB16PANZQUANT<true, DTYPE_KV> op(&pipe, &tilingData);
    op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope, c_kv, k_rope_scale, c_kv_scale);
    op.Process();
  } else if (TILING_KEY_IS(KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_BNSD_QUANT)) {
    KernelKvRmsNormRopeCacheB16PABLKBNSDQUANT<true, DTYPE_KV> op(&pipe, &tilingData);
    op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope, c_kv, k_rope_scale, c_kv_scale);
    op.Process();
  } else if (TILING_KEY_IS(KV_RMS_NORM_ROPE_CACHE_B16_PA_BNSD_QUANT)) {
    KernelKvRmsNormRopeCacheB16BNSDQUANT<true, DTYPE_KV> op(&pipe, &tilingData);
    op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope, c_kv, k_rope_scale, c_kv_scale);
    op.Process();
  } else if (TILING_KEY_IS(KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_NZ_QUANT)) {
    KernelKvRmsNormRopeCacheQuantB16PABLKNZ<true, DTYPE_KV> op(&pipe, &tilingData);
    op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope, c_kv, k_rope_scale, c_kv_scale);
    op.Process();
  } else if (TILING_KEY_IS(KV_RMS_NORM_ROPE_CACHE_B16_PA_NZ)) {
    KernelKvRmsNormRopeCacheB16PA<true, DTYPE_KV, PA_NZ_NO_QUANT> op(&pipe, &tilingData);
    op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope, c_kv);
    op.Process();
  } else if (TILING_KEY_IS(KV_RMS_NORM_ROPE_CACHE_B16_PA_BLK_BNSD)) {
    KernelKvRmsNormRopeCacheB16PA<true, DTYPE_KV, PA_BLK_BNSD_NO_QUANT> op(&pipe, &tilingData);
    op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope, c_kv);
    op.Process();
  } else if (TILING_KEY_IS(KV_RMS_NORM_ROPE_CACHE_B16_PA_BNSD)) {
    KernelKvRmsNormRopeCacheB16PA<true, DTYPE_KV, PA_BNSD_NO_QUANT> op(&pipe, &tilingData);
    op.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope, c_kv);
    op.Process();
  }
}