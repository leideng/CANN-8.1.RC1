/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file scatter_list.cpp
 * \brief
 */
#include "scatter_list_rsbse.h"
#include "scatter_list_rsble.h"
#include "scatter_list_rlbse.h"
#include "scatter_list_rlbse_pad.h"
#include "scatter_list_rlble.h"
#include "scatter_list_rlble_pad.h"
#include "scatter_list_neg_large.h"
#include "scatter_list_neg_large_e.h"
#include "scatter_list_neg_more.h"
#include "scatter_list_neg_small.h"
#include "scatter_list_transpose_large.h"
#include "scatter_list_transpose_more.h"
#include "scatter_list_transpose_small.h"
#include "scatter_list_neg_dim2_large.h"

using namespace ScatterList;

extern "C" __global__ __aicore__ void scatter_list(GM_ADDR var, GM_ADDR indice, GM_ADDR updates, GM_ADDR mask,
                                                   GM_ADDR varOut, GM_ADDR workspace, GM_ADDR tiling) {
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(100)) {
        if (sizeof(DTYPE_VAR) == sizeof(int8_t)) {
            ScatterList::ScatterListTransposeSmall<int8_t, DTYPE_INDICE> op;
            op.Init(var, indice, updates, mask, varOut, userWS, &tilingData);
            op.Process();
        }
        if (sizeof(DTYPE_VAR) == sizeof(int16_t)) {
            ScatterList::ScatterListTransposeSmall<half, DTYPE_INDICE> op;
            op.Init(var, indice, updates, mask, varOut, userWS, &tilingData);
            op.Process();
        }
        if (sizeof(DTYPE_VAR) == sizeof(int32_t)) {
            ScatterList::ScatterListTransposeSmall<float, DTYPE_INDICE> op;
            op.Init(var, indice, updates, mask, varOut, userWS, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(101)) {
        if (sizeof(DTYPE_VAR) == sizeof(int8_t)) {
            ScatterList::ScatterListTransposeMore<int8_t, DTYPE_INDICE> op;
            op.Init(var, indice, updates, mask, varOut, userWS, &tilingData);
            op.Process();
        }
        if (sizeof(DTYPE_VAR) == sizeof(int16_t)) {
            ScatterList::ScatterListTransposeMore<half, DTYPE_INDICE> op;
            op.Init(var, indice, updates, mask, varOut, userWS, &tilingData);
            op.Process();
        }
        if (sizeof(DTYPE_VAR) == sizeof(int32_t)) {
            ScatterList::ScatterListTransposeMore<float, DTYPE_INDICE> op;
            op.Init(var, indice, updates, mask, varOut, userWS, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(102)) {
        if (sizeof(DTYPE_VAR) == sizeof(int8_t)) {
            ScatterList::ScatterListTransposeLarge<int8_t, DTYPE_INDICE> op;
            op.Init(var, indice, updates, mask, varOut, userWS, &tilingData);
            op.Process();
        }
        if (sizeof(DTYPE_VAR) == sizeof(int16_t)) {
            ScatterList::ScatterListTransposeLarge<half, DTYPE_INDICE> op;
            op.Init(var, indice, updates, mask, varOut, userWS, &tilingData);
            op.Process();
        }
        if (sizeof(DTYPE_VAR) == sizeof(int32_t)) {
            ScatterList::ScatterListTransposeLarge<float, DTYPE_INDICE> op;
            op.Init(var, indice, updates, mask, varOut, userWS, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(103)) {
        ScatterList::ScatterListNegSmall<DTYPE_VAR, DTYPE_INDICE> op;
        op.Init(var, indice, updates, mask, varOut, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(104)) {
        ScatterList::ScatterListNegMore<DTYPE_VAR, DTYPE_INDICE> op;
        op.Init(var, indice, updates, mask, varOut, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(105)) {
        ScatterList::ScatterListNegLarge<DTYPE_VAR, DTYPE_INDICE> op;
        op.Init(var, indice, updates, mask, varOut, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(106)) {
        ScatterList::ScatterListNegLargeE<DTYPE_VAR, DTYPE_INDICE> op;
        op.Init(var, indice, updates, mask, varOut, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(107)) {
        ScatterList::ScatterListNegDim2Large<DTYPE_VAR, DTYPE_INDICE> op;
        op.Init(var, indice, updates, mask, varOut, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(200)) {
        ScatterList::ScatterListRSBSE<DTYPE_VAR, DTYPE_INDICE> op;
        op.Init(var, indice, updates, mask, varOut, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(210)) {
        ScatterList::ScatterListRLBSE<DTYPE_VAR, DTYPE_INDICE> op;
        op.Init(var, indice, updates, mask, varOut, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(211)) {
        ScatterList::ScatterListRLBSEPad<DTYPE_VAR, DTYPE_INDICE> op;
        op.Init(var, indice, updates, mask, varOut, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(220)) {
        ScatterList::ScatterListRSBLE<DTYPE_VAR, DTYPE_INDICE> op;
        op.Init(var, indice, updates, mask, varOut, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(230)) {
        ScatterList::ScatterListRLBLE<DTYPE_VAR, DTYPE_INDICE> op;
        op.Init(var, indice, updates, mask, varOut, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(231)) {
        ScatterList::ScatterListRLBLEPad<DTYPE_VAR, DTYPE_INDICE> op;
        op.Init(var, indice, updates, mask, varOut, userWS, &tilingData);
        op.Process();
    } else {
        return;
    }
}
