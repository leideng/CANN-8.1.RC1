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
 * \file kernel_operator_vec_transpose_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_TRANSPOSE_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_TRANSPOSE_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_transpose.h"

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif
#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_transpose_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_transpose_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_transpose_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_transpose_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_vec_transpose_impl.h"
#endif

namespace AscendC {
#pragma begin_pipe(V)
/* **************************************************************************************************
 * Transpose                                            *
 * ************************************************************************************************* */
/*
 * @ingroup Transpose
 * @brief dst[i][j] = src[j][i]
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 */
template <typename T> __aicore__ inline void Transpose(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal);

/* **************************************************************************************************
 * TransDataTo5HD                                            *
 * ************************************************************************************************* */
/*
 * @ingroup Nchwconv
 * @brief NCHW to NC1HWC0 format
 * @param [out] dstLocalList output LocalTensor list
 * @param [in] srcLocalList input LocalTensor list
 * @param [in] nchwconvParams.dstHighHalf Specify dst data is stored in the upper half or lower half of the block
 * @param [in] nchwconvParams.srcHighHalf Specify src data is stored in the upper half or lower half of the block
 * @param [in] nchwconvParams.repeatTimes repeat times
 * @param [in] nchwconvParams.dstRepStride dst repeat stride
 * @param [in] nchwconvParams.srcRepStride src repeat stride
 */
template <typename T>
__aicore__ inline __check_sync_alias__ void TransDataTo5HD(const LocalTensor<T> (&dstLocalList)[NCHW_CONV_ADDR_LIST_SIZE],
    const LocalTensor<T> (&srcLocalList)[NCHW_CONV_ADDR_LIST_SIZE], const TransDataTo5HDParams& nchwconvParams);

template <typename T>
__aicore__ inline __check_sync_alias__ void TransDataTo5HD(uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE],
    uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE], const TransDataTo5HDParams& nchwconvParams);

template <typename T>
__aicore__ inline void Transpose(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const TransposeParamsExt &transposeParams);
#pragma end_pipe
template <typename T>
__aicore__ inline __check_sync_alias__ __in_pipe__(S) __out_pipe__(V) void TransDataTo5HD(const LocalTensor<uint64_t>& dstLocal, const LocalTensor<uint64_t>& srcLocal,
    const TransDataTo5HDParams& nchwconvParams);
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_TRANSPOSE_INTERFACE_H
