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
 * \file ctc_loss_v3.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ctc_loss_v3.h"
using namespace CTCLossV3NS;

extern "C" __global__ __aicore__ void ctc_loss_v3(GM_ADDR log_probs, GM_ADDR targets,
                                                       GM_ADDR input_lengths, GM_ADDR target_lengths,
                                                       GM_ADDR neg_log_likelihood, GM_ADDR log_alpha,
                                                       GM_ADDR workspace, GM_ADDR tiling) {
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA(tilingData, tiling);
    CTCLossV3<DTYPE_LOG_PROBS, DTYPE_TARGETS> op;
    TPipe pipe;
    op.Init(&tilingData, log_probs, targets,
            input_lengths, target_lengths, neg_log_likelihood, log_alpha, workspace, &pipe);
    op.Process();
}