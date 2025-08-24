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
 * \file linear_index_v2.cpp
 * \brief
 */

#include "linear_index_v2.h"

namespace AscendC {
extern "C" __global__ __aicore__ void linear_index_v2(GM_ADDR indexList,
                                                      GM_ADDR stride,
                                                      GM_ADDR valueSize,
                                                      GM_ADDR output,
                                                      GM_ADDR workSpace,
                                                      GM_ADDR tiling) {
    if (workSpace == nullptr) {
        return;
    }
    GM_ADDR user = AscendC::GetUserWorkspace(workSpace);
    if (user == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(0)) {
        LinearIndexKernelV2<int64_t> op(indexList, stride, valueSize, output, workSpace, tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        LinearIndexKernelV2<int32_t> op(indexList, stride, valueSize, output, workSpace, tilingData, pipe);
        op.Process();
    }
}
}