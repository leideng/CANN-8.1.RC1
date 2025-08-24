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
 * \file kernel_operator_gemm_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_GEMM_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_GEMM_INTERFACE_H
#include "kernel_tensor.h"
#include "impl/kernel_operator_gemm_base_impl.h"
#include "kernel_operator_data_copy_intf.h"
namespace AscendC {
// T should be left matrix dtype
template <typename T> __aicore__ inline GemmTiling GetGemmTiling(uint32_t m, uint32_t k, uint32_t n);

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
__aicore__ inline __inout_pipe__(V) void Gemm(const LocalTensor<dst_T>& dstLocal, const LocalTensor<src0_T>& src0Local,
    const LocalTensor<src1_T>& src1Local, const uint32_t m, const uint32_t k, const uint32_t n, GemmTiling tilling,
    bool partialsum = true, int32_t initValue = 0);
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_GEMM_INTERFACE_H