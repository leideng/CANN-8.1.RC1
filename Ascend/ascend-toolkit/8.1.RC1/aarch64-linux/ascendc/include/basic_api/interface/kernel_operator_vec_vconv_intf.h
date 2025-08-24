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
 * \file kernel_operator_vec_vconv_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_VCONV_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_VCONV_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_binary.h"
#include "kernel_struct_unary.h"
#include "kernel_struct_vdeq.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_vconv_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_vconv_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_vconv_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_vconv_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_vec_vconv_impl.h"
#endif

namespace AscendC {
#pragma begin_pipe(V)
/* **************************************************************************************************
 * Cast                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Cast Level 0
 * @brief Precision conversion
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] round_mode round mode
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
// Cast::Level 0 - mask bit mode
template <typename T1, typename T2, bool isSetMask = true>
__aicore__ inline void Cast(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& srcLocal,
    const RoundMode& round_mode, const uint64_t mask[], const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams);

// Cast::Level 0 - mask count mode
template <typename T1, typename T2, bool isSetMask = true>
__aicore__ inline void Cast(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& srcLocal,
    const RoundMode& round_mode, const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Cast Level 2
 * @brief dst[i] = Precision conversion
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] round_mode round mode
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T1, typename T2>
__aicore__ inline void Cast(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& srcLocal,
    const RoundMode& round_mode, const uint32_t calCount);

/*
 * @ingroup CastDeq Level 0
 * @brief Dequant from int16_t to uint8_t/int8_t
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.srcRepStride src repeat stride
 */
template <typename T1, typename T2, bool isSetMask = true, bool isVecDeq = true, bool halfBlock = true>
__aicore__ inline void CastDeq(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& srcLocal,
    const uint64_t mask[], uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

template <typename T1, typename T2, bool isSetMask = true, bool isVecDeq = true, bool halfBlock = true>
__aicore__ inline void CastDeq(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& srcLocal,
    const int32_t mask, uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup CastDeq Level 2
 * @brief Dequant from int16_t to uint8_t/int8_t
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T1, typename T2, bool isVecDeq = true, bool halfBlock = true>
__aicore__ inline void CastDeq(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& srcLocal,
    const uint32_t calCount);

/* **************************************************************************************************
 * AddReluCast                                             *
 * ************************************************************************************************* */
/*
 * @ingroup AddReluCast Level 0
 * @brief Precision conversion
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src block stride
 * @param [in] repeatParams.src1BlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 * @param [in] repeatParams.src1RepStride src repeat stride
 */
// AddReluCast::Level 0 - mask count mode
template <typename T1, typename T2, bool isSetMask = true>
__aicore__ inline void AddReluCast(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& src0Local,
    const LocalTensor<T2>& src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams);

// AddReluCast::Level 0 - mask bit mode
template <typename T1, typename T2, bool isSetMask = true>
__aicore__ inline void AddReluCast(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& src0Local,
    const LocalTensor<T2>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams);

/*
 * @ingroup AddReluCast Level 2
 * @brief dst[i] = Precision conversion
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T1, typename T2>
__aicore__ inline void AddReluCast(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& src0Local,
    const LocalTensor<T2>& src1Local, const uint32_t calCount);

/* **************************************************************************************************
 * SubReluCast                                             *
 * ************************************************************************************************* */
/*
 * @ingroup SubReluCast Level 0
 * @brief Precision conversion
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src block stride
 * @param [in] repeatParams.src1BlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 * @param [in] repeatParams.src1RepStride src repeat stride
 */
// SubReluCast::Level 0 - mask count mode
template <typename T1, typename T2, bool isSetMask = true>
__aicore__ inline void SubReluCast(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& src0Local,
    const LocalTensor<T2>& src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams);

// SubReluCast::Level 0 - mask bit mode
template <typename T1, typename T2, bool isSetMask = true>
__aicore__ inline void SubReluCast(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& src0Local,
    const LocalTensor<T2>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams);

/*
 * @ingroup SubReluCast Level 2
 * @brief dst[i] = Precision conversion
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T1, typename T2>
__aicore__ inline void SubReluCast(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& src0Local,
    const LocalTensor<T2>& src1Local, const uint32_t calCount);

#pragma end_pipe
__aicore__ inline void SetDeqScale(half scale);

__aicore__ inline void SetDeqScale(float scale, int16_t offset, bool signMode);

template <typename T>
__aicore__ inline void SetDeqScale(const LocalTensor<T>& vdeqTensor, const VdeqInfo& vdeqInfo);
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_VCONV_INTERFACE_H
