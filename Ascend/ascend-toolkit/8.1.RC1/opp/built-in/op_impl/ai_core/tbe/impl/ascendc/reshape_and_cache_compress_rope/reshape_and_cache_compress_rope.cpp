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
 * \file reshape_and_cache_compress_rope.cpp
 * \brief
 */
#include "reshape_and_cache_compress_rope.h"
using namespace ReshapeAndCache;

extern "C" __global__ __aicore__ void reshape_and_cache_compress_rope(
    GM_ADDR keyIn, GM_ADDR valueIn, GM_ADDR keyCacheIn, GM_ADDR valueCacheIn, GM_ADDR slotMapping,
    GM_ADDR winsIn, GM_ADDR seqLenIn, GM_ADDR offsetIdx, GM_ADDR keyCacheOut, GM_ADDR valueCacheOut, 
    GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(330000000)) {
        ReshapeAndCacheCompressRope<bfloat16_t> op;
        op.Init(&pipe, &tilingData);
        op.Method(keyIn, valueIn, keyCacheIn, valueCacheIn, slotMapping,
            winsIn, seqLenIn, offsetIdx, keyCacheOut, valueCacheOut);
    }
    else if (TILING_KEY_IS(230000000)) {
        ReshapeAndCacheCompressRope<half> op;
        op.Init(&pipe, &tilingData);
        op.Method(keyIn, valueIn, keyCacheIn, valueCacheIn, slotMapping,
            winsIn, seqLenIn, offsetIdx, keyCacheOut, valueCacheOut);
    }
}