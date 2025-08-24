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
 * \file iou_v2.cpp
 * \brief
 */

#include "iou_v2_align_iof.h"
#include "iou_v2_align_iou.h"
#include "iou_v2_not_align_iof.h"
#include "iou_v2_not_align_iou.h"

using namespace AscendC;
using namespace IouV2;

extern "C" __global__ __aicore__ void iou_v2(
    GM_ADDR bboxes,
    GM_ADDR gtboxes,
    GM_ADDR overlap,
    GM_ADDR workspace,
    GM_ADDR tiling
) {
    GET_TILING_DATA(tiling_data, tiling);
#define INIT_AND_PROCESS \
    op.Init(bboxes, gtboxes, overlap, &tiling_data); \
    op.Process()

    if (TILING_KEY_IS(4)) {
        KernelIouV2Align<float> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(7)) {
        KernelIouV2NotAlign<float> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(14)) {
        KernelIofV2Align<float> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(17)) {
        KernelIofV2NotAlign<float> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(5)) {
        KernelIouV2Align<half> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(8)) {
        KernelIouV2NotAlign<half> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(15)) {
        KernelIofV2Align<half> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(18)) {
        KernelIofV2NotAlign<half> op;
        INIT_AND_PROCESS;
#if __CCE_AICORE__ != 200
    } else if (TILING_KEY_IS(6)) {
        KernelIouV2Align<bfloat16_t> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(9)) {
        KernelIouV2NotAlign<bfloat16_t> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(16)) {
        KernelIofV2Align<bfloat16_t> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(19)) {
        KernelIofV2NotAlign<bfloat16_t> op;
        INIT_AND_PROCESS;
#endif
    }
}
