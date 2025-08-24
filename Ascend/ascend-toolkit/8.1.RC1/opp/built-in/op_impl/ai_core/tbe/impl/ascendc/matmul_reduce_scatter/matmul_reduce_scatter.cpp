/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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

/* !
 * \file matmul_reduce_scatter.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "matmul_reduce_scatter_tiling.h"
#include "matmul_reduce_scatter_full_mesh.h"

#define INVOKE_MATMUL_REDUCE_SCATTER_OP_IMPL(templateClass, ...)                         \
    do {                                                                                 \
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_DTYPE, true>; \
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;       \
        templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                    \
        op.Init(aGM, bGM, biasGM, cGM, workspaceGM, contextGM, &tilingData, &pipe, mc2InitTiling, mc2CcTiling);  \
        op.Process();                                                                    \
    } while (0)
using namespace AscendC;
namespace AscendC {
template <class T> struct BiasType {
    using type = float;
};
template <> struct BiasType<half> {
    using type = half;
};

extern "C" __global__ __aicore__ void matmul_reduce_scatter(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MatmulReduceScatterTilingData);
    auto tiling = (__gm__ MatmulReduceScatterTilingData*)tilingGM;
    __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));
    __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));
    GET_TILING_DATA(tilingData, tilingGM);

    TPipe pipe;
    GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();
    if (TILING_KEY_IS(110)) {
        // full mesh + nd2nz + no bias cast
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, B_DTYPE, true>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, typename BiasType<BIAS_DTYPE>::type>;
        INVOKE_MATMUL_REDUCE_SCATTER_OP_IMPL(MatmulReduceScatterFullMesh, true, false);
    } else if (TILING_KEY_IS(100)) {
        // full mesh + no nd2nz + no bias cast
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_DTYPE, true>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, typename BiasType<BIAS_DTYPE>::type>;
        INVOKE_MATMUL_REDUCE_SCATTER_OP_IMPL(MatmulReduceScatterFullMesh, false, false);
    } else if (TILING_KEY_IS(111)) {
        // full mesh + nd2nz + biasNeededCast
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, B_DTYPE, true>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
        INVOKE_MATMUL_REDUCE_SCATTER_OP_IMPL(MatmulReduceScatterFullMesh, true, true);
    } else if (TILING_KEY_IS(101)) {
        // full mesh + no nd2nz + biasNeededCast
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_DTYPE, true>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
        INVOKE_MATMUL_REDUCE_SCATTER_OP_IMPL(MatmulReduceScatterFullMesh, false, true);
    } 
}
}