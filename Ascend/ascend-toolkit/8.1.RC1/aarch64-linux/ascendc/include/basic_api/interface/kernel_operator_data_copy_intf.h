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
 * \file kernel_operator_data_copy_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_DATA_COPY_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_DATA_COPY_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_process_lock.h"

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

#include "impl/kernel_operator_data_copy_base_impl.h"

namespace AscendC {
/* **************************************************************************************************
 * DataCopy                                             *
 * ************************************************************************************************* */
/*
 * @ingroup DataCopy Level 0
 * @brief datacopy from src to dst, applicable to vector data
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcGlobal input GlobalTensor
 * @param [in] repeatParams.blockCount number of blocks
 * @param [in] repeatParams.blockLen Length of blocks
 * @param [in] repeatParams.srcStride src block stride
 * @param [in] repeatParams.dstStride dst block stride
 */
template <typename T>
__aicore__ inline void __inout_pipe__(MTE2)
    DataCopy(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal, const DataCopyParams& repeatParams);

/*
 * @ingroup DataCopy Level 0
 * @brief format transform(such as nd2nz) during data load from OUT to L1
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcGlobal input GlobalTensor
 * @param [in] intriParams.ndNum nd number of data to be moved
 * @param [in] intriParams.nValue n value
 * @param [in] intriParams.dValue d value in unit of element
 * @param [in] intriParams.srcNdMatrixStride stride between nd matrixs at source ND matrix in unit of element
 * @param [in] intriParams.srcDValue SRC_D value in unit of element
 * @param [in] intriParams.dstNzC0Stride stride of nz between 2 C0 in L1 in unit of C0_size
 * @param [in] intriParams.dstNzNStride stride of n between 2 C0 in L1
 * @param [in] intriParams.dstNzMatrixStride DST_nz_matrix_stride in L1 in unit of element
 */
template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void DataCopy(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
                                                     const Nd2NzParams& intriParams);

/*
 * @ingroup DataCopy Level 0
 * @brief format transform(such as nd2nz) during data load from UB to L1(Only TSCM)
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcGlobal input LocalTensor
 * @param [in] intriParams.ndNum nd number of data to be moved, onlyc can be 1
 * @param [in] intriParams.nValue n value
 * @param [in] intriParams.dValue d value in unit of element
 * @param [in] intriParams.srcNdMatrixStride stride between nd matrixs at source ND matrix in unit of element
 * @param [in] intriParams.srcDValue SRC_D value in unit of element
 * @param [in] intriParams.dstNzC0Stride stride of nz between 2 C0 in L1 in unit of C0_size
 * @param [in] intriParams.dstNzNStride stride of n between 2 C0 in L1
 * @param [in] intriParams.dstNzMatrixStride DST_nz_matrix_stride in L1 in unit of element
 */
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcGlobal,
                                     const Nd2NzParams& intriParams);

/*
 * @ingroup DataCopy Level 0
 * @brief datacopy from src to dst, applicable to vector data
 * @param [out] dstGlobal output GlobalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] repeatParams.blockCount number of blocks
 * @param [in] repeatParams.blockLen Length of blocks
 * @param [in] repeatParams.srcStride src block stride
 * @param [in] repeatParams.dstStride dst block stride
 */
template <typename T>
__aicore__ inline __inout_pipe__(MTE3) void DataCopy(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
                                                     const DataCopyParams& repeatParams);

/*
 * @ingroup DataCopy Level 0
 * @brief datacopy from src to dst, applicable to vector data
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] repeatParams.blockCount number of blocks
 * @param [in] repeatParams.blockLen Length of blocks
 * @param [in] repeatParams.srcStride src block stride
 * @param [in] repeatParams.dstStride dst block stride
 */
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                const DataCopyParams& repeatParams);

/*
 * @ingroup DataCopy Level 0
 * @brief datacopy from L1 to bt, applicable to vector data
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] repeatParams.blockCount number of blocks
 * @param [in] repeatParams.blockLen Length of blocks
 * @param [in] repeatParams.srcStride src block stride
 * @param [in] repeatParams.dstStride dst block stride
 */
template <typename dst_T, typename src_T>
__aicore__ inline void DataCopy(const LocalTensor<dst_T>& dstLocal, const LocalTensor<src_T>& srcLocal,
                                const DataCopyParams& repeatParams);

/*
 * @ingroup Copy Level 0
 * @brief datacopy from src to dst, applicable to vector data
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstStride dst block stride
 * @param [in] repeatParams.srcStride src block stride
 * @param [in] repeatParams.dstRepeatSize dst repeat stride
 * @param [in] repeatParams.srcRepeatSize src repeat stride
 */
// Copy::Level 0 - mask bit mode
template <typename T, bool isSetMask = true>
__aicore__ inline __inout_pipe__(V) void Copy(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                              const uint64_t mask[], const uint8_t repeatTimes,
                                              const CopyRepeatParams& repeatParams);

// Copy::Level 0 - mask count mode
template <typename T, bool isSetMask = true>
__aicore__ inline __inout_pipe__(V) void Copy(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                              const uint64_t mask, const uint8_t repeatTimes,
                                              const CopyRepeatParams& repeatParams);

/*
 * @ingroup DataCopy Level 1
 * @brief datacopy from src to dst, applicable to vector data
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcGlobal input GlobalTensor
 * @param [in] SliceInfo dstSliceInfo[] ub
 * @param [in] SliceInfo srcSliceInfo[] gm
 * @param [in] dimValue dim value also for length for dstSliceInfo[] and srcSliceInfo[]
 */
template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void DataCopy(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
                                                     const SliceInfo dstSliceInfo[], const SliceInfo srcSliceInfo[],
                                                     const uint32_t dimValue = 1);

/*
 * @ingroup DataCopy Level 1
 * @brief datacopy from src to dst, applicable to vector data
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcGlobal input GlobalTensor
 * @param [in] SliceInfo dstSliceInfo[] gm
 * @param [in] SliceInfo srcSliceInfo[] ub
 * @param [in] dimValue dim value also for length for dstSliceInfo[] and srcSliceInfo[]
 */
template <typename T>
__aicore__ inline __inout_pipe__(MTE3) void DataCopy(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
                                                     const SliceInfo dstSliceInfo[], const SliceInfo srcSliceInfo[],
                                                     const uint32_t dimValue = 1);

/*
 * @ingroup DataCopy Level 2
 * @brief datacopy from src to dst, applicable to vector data
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcGlobal input GlobalTensor
 * @param [in] calCount Number of operands
 */
template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void DataCopy(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
                                                     const uint32_t calCount);

/*
 * @ingroup DataCopy Level 2
 * @brief datacopy from src to dst, applicable to vector data
 * @param [out] dstGlobal output GlobalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calCount Number of operands
 */
template <typename T>
__aicore__ inline __inout_pipe__(MTE3) void DataCopy(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
                                                     const uint32_t calCount);

/*
 * @ingroup DataCopy Level 2
 * @brief datacopy from src to dst, applicable to vector data
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calCount Number of operands
 */
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                const uint32_t calCount);

/*
 * @ingroup DataCopy Level 2
 * @brief datacopy from src to dst, nz2nd, applicable to simulated cube data(such as data from l0c, 16*16)
 * @param [out] dstGlobal output GlobalTensor
 * @param [in] srcLocal input LocalTensor
 */
template <typename T>
__aicore__ inline __inout_pipe__(MTE3) void DataCopy(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
                                                     const Nz2NdParamsFull& intriParams);

/* **************************************************************************************************
 * DataCopy Enhanced                                             *
 * ************************************************************************************************* */
/*
 * @ingroup DataCopy
 * @brief datacopy from src to dst, applicable to cube data
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcGlobal input GlobalTensor
 * @param [in] intriParams.blockCount number of blocks
 * @param [in] intriParams.blockLen Length of blocks
 * @param [in] intriParams.srcStride src block stride
 * @param [in] intriParams.dstStride dst block stride
 * @param [in] enhancedParams.blockMode Basic fractal of data movement
 * @param [in] enhancedParams.deqScale Auxiliary parameters for path accuracy conversion
 * @param [in] enhancedParams.deqValue size of convert with path precision
 * @param [in] enhancedParams.sidStoreMode Multiplex input
 * @param [in] enhancedParams.isRelu Configure whether Relu can be performed along the circuit
 */
template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void DataCopy(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
                                                     const DataCopyParams& intriParams,
                                                     const DataCopyEnhancedParams& enhancedParams);

template <typename T>
__aicore__ inline __inout_pipe__(MTE3) void DataCopy(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
                                                     const DataCopyParams& intriParams,
                                                     const DataCopyEnhancedParams& enhancedParams);

template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                const DataCopyParams& intriParams, const DataCopyEnhancedParams& enhancedParams);

template <typename T, typename U>
__aicore__ inline void DataCopy(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal,
                                const DataCopyCO12DstParams& intriParams);

template <typename T, typename U>
__aicore__ inline void DataCopy(const GlobalTensor<T>& dstGlobal, const LocalTensor<U>& srcLocal,
                                const DataCopyCO12DstParams& intriParams);

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
// float to bfloat16_t
template <typename T, typename U,
          typename std::enable_if<IsSameType<PrimT<T>, bfloat16_t>::value && IsSameType<PrimT<U>, float>::value,
                                  bool>::type = true>
__aicore__ inline void DataCopy(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal,
                                const DataCopyParams& intriParams, const DataCopyEnhancedParams& enhancedParams);
#endif

// float to half
template <
    typename T, typename U,
    typename std::enable_if<IsSameType<PrimT<T>, half>::value && IsSameType<PrimT<U>, float>::value, bool>::type = true>
__aicore__ inline void DataCopy(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal,
                                const DataCopyParams& intriParams, const DataCopyEnhancedParams& enhancedParams);

// int32_t to half
template <typename T, typename U,
          typename std::enable_if<IsSameType<PrimT<T>, half>::value && IsSameType<PrimT<U>, int32_t>::value,
                                  bool>::type = true>
__aicore__ inline __inout_pipe__(V) void DataCopy(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal,
                                                  const DataCopyParams& intriParams,
                                                  const DataCopyEnhancedParams& enhancedParams);

// int32_t to int16_t
template <typename T, typename U,
          typename std::enable_if<IsSameType<PrimT<T>, int16_t>::value && IsSameType<PrimT<U>, int32_t>::value,
                                  bool>::type = true>
__aicore__ inline __inout_pipe__(V) void DataCopy(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal,
                                                  const DataCopyParams& intriParams,
                                                  const DataCopyEnhancedParams& enhancedParams);

// int32_t to int8_t
template <typename T, typename U,
          typename std::enable_if<IsSameType<PrimT<T>, int8_t>::value && IsSameType<PrimT<U>, int32_t>::value,
                                  bool>::type = true>
__aicore__ inline __inout_pipe__(V) void DataCopy(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal,
                                                  const DataCopyParams& intriParams,
                                                  const DataCopyEnhancedParams& enhancedParams);

// int32_t to uint8_t
template <typename T, typename U,
          typename std::enable_if<IsSameType<PrimT<T>, uint8_t>::value && IsSameType<PrimT<U>, int32_t>::value,
                                  bool>::type = true>
__aicore__ inline __inout_pipe__(V) void DataCopy(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal,
                                                  const DataCopyParams& intriParams,
                                                  const DataCopyEnhancedParams& enhancedParams);

// half to float
template <
    typename T, typename U,
    typename std::enable_if<IsSameType<PrimT<T>, float>::value && IsSameType<PrimT<U>, half>::value, bool>::type = true>
__aicore__ inline __inout_pipe__(V) void DataCopy(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal,
                                                  const DataCopyParams& intriParams,
                                                  const DataCopyEnhancedParams& enhancedParams);

template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void DataCopyPad(const LocalTensor<T>& dstLocal,
                                                        const GlobalTensor<T>& srcGlobal,
                                                        const DataCopyParams& dataCopyParams,
                                                        const DataCopyPadParams& padParams);

template <typename T>
__aicore__ inline __inout_pipe__(MTE3) void DataCopyPad(const GlobalTensor<T>& dstGlobal,
                                                        const LocalTensor<T>& srcLocal,
                                                        const DataCopyParams& dataCopyParams);

template <typename T>
__aicore__ inline void DataCopyPad(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                        const DataCopyParams& dataCopyParams, const Nd2NzParams& nd2nzParams);

// override DataCopyPad, use new param DataCopyExtParams
template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void DataCopyPad(const LocalTensor<T>& dstLocal,
                                                        const GlobalTensor<T>& srcGlobal,
                                                        const DataCopyExtParams& dataCopyParams,
                                                        const DataCopyPadExtParams<T>& padParams);

// override DataCopyPad, use new param DataCopyExtParams
// T use TensorTrait while U is primitive type
template <typename T, typename U,
          typename std::enable_if<IsSameType<PrimT<T>, U>::value && (!IsSameType<T, U>::value), bool>::type = true>
__aicore__ inline __inout_pipe__(MTE2) void DataCopyPad(const LocalTensor<T>& dstLocal,
                                                        const GlobalTensor<T>& srcGlobal,
                                                        const DataCopyExtParams& dataCopyParams,
                                                        const DataCopyPadExtParams<U>& padParams);

template <typename T>
__aicore__ inline __inout_pipe__(MTE3) void DataCopyPad(const GlobalTensor<T>& dstGlobal,
                                                        const LocalTensor<T>& srcLocal,
                                                        const DataCopyExtParams& dataCopyParams);

template <typename T>
__aicore__ inline void DataCopyPad(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                        const DataCopyExtParams& dataCopyParams, const Nd2NzParams& nd2nzParams);

template <typename T, TPosition pos = TPosition::MAX>
__aicore__ inline void SetPadValue(T paddingValue);
}  // namespace AscendC
#endif  // ASCENDC_MODULE_OPERATOR_VEC_VCONV_INTERFACE_H