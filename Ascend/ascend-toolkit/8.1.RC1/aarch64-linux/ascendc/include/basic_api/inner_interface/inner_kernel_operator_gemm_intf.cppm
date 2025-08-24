/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
 * \file inner_kernel_operator_gemm_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_GEMM_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_GEMM_INTERFACE_H
#include "kernel_tensor.h"
#include "impl/kernel_operator_gemm_base_impl.h"
#include "kernel_operator_data_copy_intf.h"
#include "kernel_struct_data_copy.h"

namespace AscendC {
// T should be left matrix dtype
template <typename T> 
[[deprecated("NOTICE: GetGemmTiling has been deprecated and will be removed in the next version. "
        "Please do not use it!")]]
__aicore__ inline GemmTiling GetGemmTiling(uint32_t m, uint32_t k, uint32_t n)
{
    uint32_t c0 = 0;
    uint32_t dSize = 1;
    if (IsSameType<T, uint8_t>::value || IsSameType<T, int8_t>::value) {
        c0 = 32;
        dSize = 1;
    } else {
        c0 = 16;
        dSize = 2;
    }
    GemmTiling tilling;
    tilling.c0Size = c0;
    tilling.dtypeSize = dSize;
    tilling.mNum = m;
    tilling.nNum = n;
    tilling.kNum = k;
    tilling.roundM = DivCeil(m, tilling.blockSize) * tilling.blockSize; // blockSize = 16 * 16
    tilling.roundN = DivCeil(n, tilling.blockSize) * tilling.blockSize;
    tilling.roundK = DivCeil(k, tilling.c0Size) * tilling.c0Size; // c0Size = 16 || c0Size = 32
    uint32_t k0a = TOTAL_L0A_SIZE / 2 / (tilling.roundM * dSize);
    uint32_t k0b = TOTAL_L0B_SIZE / 2 / (tilling.roundN * dSize);
    uint32_t k0 = k0a > k0b ? k0b : k0a;
    k0 = k0 > k ? k : k0;

    tilling.kTileBlock = k0 / tilling.c0Size;
    if (tilling.kTileBlock == 0) {
        tilling.kTileBlock = 1;
    }
    tilling.loopMode = LoopMode::MODE_NM;

    tilling.mBlockNum = DivCeil(m, tilling.blockSize);
    tilling.nBlockNum = DivCeil(n, tilling.blockSize);
    tilling.kBlockNum = DivCeil(k, tilling.c0Size);

    CalculateGemmTiling(tilling);

    return tilling;
}

/*
 * @ingroup Gemm
 * @brief Multiply two matrices
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input GlobalTensor
 * @param [in] src1Local input GlobalTensor
 * @param [in] m Number of rows of src0Local
 * @param [in] n Number of rows of src1Local
 * @param [in] k Number of columns of src1Local
 * @param [in] tilling.blockSize size of block
 * @param [in] tilling.mNum args of m
 * @param [in] tilling.nNum args of n
 * @param [in] tilling.kNum args of k
 * @param [in] tilling.roundM/N/K Rounding parameter
 * @param [in] tilling.c0Size The byte length of a block
 * @param [in] tilling.dtypeSize Byte length of the incoming data type
 * @param [in] tilling.m/n/kBlockNum Number of blocks of m/n/k axis
 * @param [in] tilling.m/n/kIterNum Number of traversal dimensions
 * @param [in] tilling.m/k/nTileBlock Number of M/N/K axis cutting blocks
 * @param [in] tilling.m/n/kHasTailNumber of tail blocks of M/K/N axis
 * @param [in] tilling.kHasTileEle Judge whether the tail block exists
 * @param [in] tilling.KtailEle K-axis tail block element
 * @param [in] tilling.kThreadNum K-axis passes
 * @param [in] partialsum judge whether the calculation result is moved out
 * @param [in] initValue Initialization parameters
 */
template <typename dst_T, typename src0_T, typename src1_T>
[[deprecated("NOTICE: Gemm has been deprecated and will be removed in the next version. "
        "Please do not use it!")]]
__aicore__ inline __inout_pipe__(V) void Gemm(const LocalTensor<dst_T>& dstLocal, const LocalTensor<src0_T>& src0Local,
    const LocalTensor<src1_T>& src1Local, const uint32_t m, const uint32_t k, const uint32_t n, GemmTiling tilling,
    bool partialsum, int32_t initValue)
{
#if ASCENDC_CPU_DEBUG
    bool flag = CheckParams(dstLocal, src0Local, src1Local, m, k, n, tilling);
    if (!flag) {
        return;
    }
#endif

#if __CCE_AICORE__ < 220
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    DataCopyEnhancedParams enhancedParams;
    enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;
#endif

    const Hardware dstScope = GetPhyType((TPosition)dstLocal.GetPosition());
    LocalTensor<dst_T> L0c;
    if (dstScope == Hardware::L0C) {
        L0c = dstLocal[0];
    } else {
#if __CCE_AICORE__ < 220
        TBuffAddr tbufc;
        tbufc.logicPos = (uint8_t)TPosition::C2;
        L0c.SetAddr(tbufc);
        L0c.InitBuffer(0, TOTAL_L0C_SIZE / sizeof(PrimT<dst_T>));

        dataCopyParams.blockLen = dstLocal.GetSize() * sizeof(PrimT<dst_T>) / 1024;
        DataCopy(L0c, dstLocal, dataCopyParams, enhancedParams);
#endif
    }

    if (tilling.loopMode == LoopMode::MODE_NM) {
        GemmExecNm(L0c, src0Local, src1Local, tilling, initValue);
    } else if (tilling.loopMode == LoopMode::MODE_MN) {
        GemmExecMn(L0c, src0Local, src1Local, tilling, initValue);
    } else {
        // other mode are not supported
    }

#if __CCE_AICORE__ < 220
    if (dstScope == Hardware::UB) {
        pipe_barrier(PIPE_ALL);
        dataCopyParams.blockLen = tilling.roundM * tilling.roundN * sizeof(PrimT<dst_T>) / 1024;
        DataCopy(dstLocal, L0c, dataCopyParams, enhancedParams);
    }
#endif
}
} // namespace AscendC
#endif // ASCENDC_MODULE_INNER_OPERATOR_GEMM_INTERFACE_H