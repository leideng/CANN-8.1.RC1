/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file reverse_sequence.cpp
 * \brief
 */
#include "reverse_sequence_batch_0.h"
#include "reverse_sequence_batch_1.h"

#define BATCH_DIM_0_C_SMALL 101
#define BATCH_DIM_0_C_BIG 201
#define BATCH_DIM_1_C_SMALL 301
#define BATCH_DIM_1_C_BIG 401

using namespace ReverseSequence;

extern "C" __global__ __aicore__ void reverse_sequence(GM_ADDR x, GM_ADDR seq_lengths, GM_ADDR y, GM_ADDR workspace,
                                                       GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(BATCH_DIM_0_C_SMALL)) {
        ReverseSequenceBatch0<DTYPE_SEQ_LENGTHS> op(&tilingData);
        op.Init(x, seq_lengths, y, workspace);
        op.Process<ReverseSequenceBatch0<DTYPE_SEQ_LENGTHS>, &ReverseSequenceBatch0<DTYPE_SEQ_LENGTHS>::ReverseSeq>(
            &op);
    } else if (TILING_KEY_IS(BATCH_DIM_0_C_BIG)) {
        ReverseSequenceBatch0<DTYPE_SEQ_LENGTHS, false> op(&tilingData);
        op.Init(x, seq_lengths, y, workspace);
        op.Process<ReverseSequenceBatch0<DTYPE_SEQ_LENGTHS, false>,
                   &ReverseSequenceBatch0<DTYPE_SEQ_LENGTHS, false>::ReverseSeq>(&op);
    } else if (TILING_KEY_IS(BATCH_DIM_1_C_SMALL)) {
        ReverseSequenceBatch1<DTYPE_SEQ_LENGTHS> op(&tilingData);
        op.Init(x, seq_lengths, y, workspace);
        op.Process<ReverseSequenceBatch1<DTYPE_SEQ_LENGTHS>, &ReverseSequenceBatch1<DTYPE_SEQ_LENGTHS>::ReverseSeq>(
            &op);
    } else if (TILING_KEY_IS(BATCH_DIM_1_C_BIG)) {
        ReverseSequenceBatch1<DTYPE_SEQ_LENGTHS, false> op(&tilingData);
        op.Init(x, seq_lengths, y, workspace);
        op.Process<ReverseSequenceBatch1<DTYPE_SEQ_LENGTHS, false>,
                   &ReverseSequenceBatch1<DTYPE_SEQ_LENGTHS, false>::ReverseSeq>(&op);
    }
}