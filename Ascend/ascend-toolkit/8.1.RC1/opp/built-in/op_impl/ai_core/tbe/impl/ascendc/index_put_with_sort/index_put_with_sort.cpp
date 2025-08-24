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
 * \file index_put_with_sort.cpp
 * \brief
 */

#include "index_put_with_sort.h"
#include "index_put_with_sort_accumulate.h"
#include "index_put_with_sort_determinist.h"
#include "index_put_with_sort_determinist_cast.h"

extern "C" __global__ __aicore__ void index_put_with_sort(GM_ADDR self, GM_ADDR linearIndex,
    GM_ADDR posIdx, GM_ADDR values, GM_ADDR output, GM_ADDR workSpace, GM_ADDR tiling) {
    if (workSpace == nullptr) {
        return;
    }
    GM_ADDR user = AscendC::GetUserWorkspace(workSpace);
    if (user == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe tpipe;
    uint32_t accumulate = tilingData.params.accumulate;
    if (TILING_KEY_IS(0) || TILING_KEY_IS(10)) {
        AscendC::IndexPutWithSortKernel<float> op(self, linearIndex, posIdx, values, output, workSpace, tilingData, tpipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        AscendC::IndexPutWithSortAccumulateKernel<float> accumulateOp(self, linearIndex, posIdx, values, output, workSpace, tilingData, tpipe);
        accumulateOp.Process();
    } else if (TILING_KEY_IS(11)) {
        AscendC::IndexPutWithSortDeterministKernel<float> determinOp(self, linearIndex, posIdx, values, output, workSpace, tilingData, tpipe);
        determinOp.Process();
    } else if (TILING_KEY_IS(100) || TILING_KEY_IS(110)) {
        AscendC::IndexPutWithSortKernel<half> op(self, linearIndex, posIdx, values, output, workSpace, tilingData, tpipe);
        op.Process();
    } else if (TILING_KEY_IS(101)) {
        AscendC::IndexPutWithSortDeterministCastKernel<half> accumulateOp(self, linearIndex, posIdx, values, output, workSpace, tilingData, tpipe);
        accumulateOp.Process();
    } else if (TILING_KEY_IS(111)) {
        AscendC::IndexPutWithSortDeterministCastKernel<half> determinOp(self, linearIndex, posIdx, values, output, workSpace, tilingData, tpipe);
        determinOp.Process();
    } else if (TILING_KEY_IS(200) || TILING_KEY_IS(210)) {
        AscendC::IndexPutWithSortKernel<bfloat16_t> op(self, linearIndex, posIdx, values, output, workSpace, tilingData, tpipe);
        op.Process();
    } else if (TILING_KEY_IS(201)) {
        AscendC::IndexPutWithSortDeterministCastKernel<bfloat16_t> accumulateOp(self, linearIndex, posIdx, values, output, workSpace, tilingData, tpipe);
        accumulateOp.Process();
    } else if (TILING_KEY_IS(211)) {
        AscendC::IndexPutWithSortDeterministCastKernel<bfloat16_t> determinOp(self, linearIndex, posIdx, values, output, workSpace, tilingData, tpipe);
        determinOp.Process();
    }
}