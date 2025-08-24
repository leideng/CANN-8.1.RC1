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
 * \file grouped_mat_mul_all_reduce.cpp
 * \brief
 */
#define K_MAX_SHAPE_DIM 0

#include "kernel_operator.h"
#include "grouped_mat_mul_all_reduce_utils.h"
#include "grouped_mat_mul_all_reduce.h"

using namespace AscendC;
using namespace matmul;
using namespace GROUPED_MAT_MUL_ALL_REDUCE;

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
    uint64_t winSize;
    uint64_t windowsIn[AC_MAX_RANK_NUM];
    uint64_t windowsOut[AC_MAX_RANK_NUM];
    char hcomId[HCCL_COMM_DOMAIN_KEY_MAX_LEN];
    HcclStreamInfo streamInfo[AC_MAX_RANK_NUM];
    HcclCombinOpSignalParam signalInfo;
    HcclConfig config;  // 配置参数
};

// for oom check
__aicore__ inline void OOMInit(__gm__ HcclCombinOpParam *context) {
#ifndef __CCE_KT_TEST__
    AscendC::OOMCheckAddrRange((__gm__ uint8_t *)(context->WorkSpace), context->WorkSpaceSize);
    AscendC::OOMCheckAddrRange((__gm__ uint8_t *)(context->windowsIn[context->rankId]), context->winSize);
#endif
}

using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X, false>;
using weightType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_WEIGHT, false>;
using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_Y>;
using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;

extern "C" __global__ __aicore__ void grouped_mat_mul_all_reduce(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                                                GM_ADDR group_list, GM_ADDR y,
                                                                GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA_MEMBER(GMMAllReduceTilingData, aicoreTiling, aicore_tiling_data, tiling);
    if (aicore_tiling_data.debugMode == static_cast<uint32_t>(DebugMode::MC2_DEBUG_ONLY_AICPU)) {
        return;
    }

    const TCubeTiling* __restrict mmTiling = &(aicore_tiling_data.mmTilingData);
    GM_ADDR user1 = GetUserWorkspace(workspace);

    TPipe tPipe;
    HcclServer hcclServer;
    __gm__ HcclCombinOpParam *context = (__gm__ HcclCombinOpParam *)(GetHcclContext<0>());
    OOMInit(context);
    __gm__ uint8_t *workspaceMsg = (__gm__ uint8_t *)(context->WorkSpace + aicore_tiling_data.notifyOff);
    hcclServer.Init(workspaceMsg, aicore_tiling_data.debugMode);

    if (TILING_KEY_IS(0)) {  // float16 and bf16
        if ASCEND_IS_AIV { return; }
        using mmType = MMImplType<xType, weightType, yType, biasType, CFG_MDL>;
        mmType::MT mm;
        mm.SetSubBlockIdx(0);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        PRELOAD(4);  // If comment this line, the program will be blocked.
#endif
        mm.Init(mmTiling, &tPipe);
        GMMCompute<mmType> computeOp(mm);
        computeOp.Init(x, weight, bias, y, user1);
        GMMProcess<decltype(computeOp)> op(computeOp);
        op.Init(&aicore_tiling_data, &tPipe);
        op.Process(hcclServer);
    }
}
