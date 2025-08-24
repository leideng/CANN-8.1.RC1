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
 * \file inner_kernel_operator_vec_transpose_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_VEC_TRANSPOSE_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_VEC_TRANSPOSE_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_check.h"
#include "kernel_struct_transpose.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_transpose_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_transpose_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_transpose_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_transpose_impl.h"
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
template <typename T> __aicore__ inline void Transpose(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal)
{
    ASCENDC_ASSERT((SupportType<T, int16_t, uint16_t, half>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Transpose, current api support dtype combination is "
        "src and dst both: int16_t, uint16_t, half");});
#if ASCENDC_CPU_DEBUG
    if (!CheckFunTranspose(dstLocal, srcLocal, "Transpose")) {
        ASCENDC_REPORT_CHECK_ERROR("Transpose", KernelFuncType::NONE_MODE);
    }
#endif
    TransposeImpl((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr());
}

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
__aicore__ inline void TransDataTo5HD(const LocalTensor<T> (&dstLocalList)[NCHW_CONV_ADDR_LIST_SIZE],
    const LocalTensor<T> (&srcLocalList)[NCHW_CONV_ADDR_LIST_SIZE], const TransDataTo5HDParams& nchwconvParams)
{
    ASCENDC_ASSERT((SupportType<T, int8_t, uint8_t, int16_t, uint16_t, half, float, int32_t, uint32_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in TransDataTo5HD, current api support dtype combination is "
        "src and dst both: int8_t, uint8_t, int16_t, uint16_t, half, float, int32_t, uint32_t");});
#if ASCENDC_CPU_DEBUG
    if (!CheckFunTransDataTo5HD(dstLocalList, srcLocalList, nchwconvParams, "TransDataTo5HD")) {
        ASCENDC_REPORT_CHECK_ERROR("TransDataTo5HD", KernelFuncType::NONE_MODE);
    }
#endif
    __ubuf__ T* dstList[NCHW_CONV_ADDR_LIST_SIZE];
    __ubuf__ T* srcList[NCHW_CONV_ADDR_LIST_SIZE];

    for (int32_t i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        dstList[i] = (__ubuf__ T*)dstLocalList[i].GetPhyAddr();
        srcList[i] = (__ubuf__ T*)srcLocalList[i].GetPhyAddr();
    }

    TransDataTo5HDImpl(dstList, srcList, nchwconvParams);
}

template <typename T>
__aicore__ inline void TransDataTo5HD(uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE],
    uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE], const TransDataTo5HDParams& nchwconvParams)
{
    ASCENDC_ASSERT((SupportType<T, int8_t, uint8_t, int16_t, uint16_t, half, float, int32_t, uint32_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in TransDataTo5HD, current api support dtype combination "
        "is src and dst both: int8_t, uint8_t, int16_t, uint16_t, half, float, int32_t, uint32_t");});
#if ASCENDC_CPU_DEBUG
    for (int8_t i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        uint64_t dstAddr = (uint8_t *)dstList[i] -
                           (uint8_t*)(GetTPipePtr()->GetBaseAddr(int8_t(AscendC::TPosition(TPosition::VECIN))));
        uint64_t srcAddr = (uint8_t *)srcList[i] -
                           (uint8_t*)(GetTPipePtr()->GetBaseAddr(int8_t(AscendC::TPosition(TPosition::VECIN))));
        ASCENDC_ASSERT((dstAddr % ONE_BLK_SIZE == 0),
            {KERNEL_LOG(KERNEL_ERROR, "Failed to check dst tensor address list alignment in TransDataTo5HD, "
            "it should be 32B aligned");});
        ASCENDC_ASSERT((srcAddr % ONE_BLK_SIZE == 0),
            {KERNEL_LOG(KERNEL_ERROR, "Failed to check src tensor address list alignment in TransDataTo5HD, "
            "it should be 32B aligned");});
    }
#endif
    TransDataTo5HDImpl<T>(dstList, srcList, nchwconvParams);
}

template <typename T>
__aicore__ inline void Transpose(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const TransposeParamsExt &transposeParams)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckFunTranspose(dstLocal, srcLocal, sharedTmpBuffer, transposeParams, "Transpose")) {
        ASCENDC_REPORT_CHECK_ERROR("Transpose", KernelFuncType::NONE_MODE);
    }
#endif
    if ((transposeParams.transposeType == TransposeType::TRANSPOSE_ND2ND_B16) &&
        (transposeParams.hSize == NCHW_CONV_ADDR_LIST_SIZE) && (transposeParams.wSize == NCHW_CONV_ADDR_LIST_SIZE)) {
        ASCENDC_ASSERT((SupportType<T, uint16_t>()),
            {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Transpose when transposeType is TRANSPOSE_ND2ND_B16, "
            "current api support dtype combination is src and dst both: uint16_t");});
        TransposeImpl((__ubuf__ T *)dstLocal.GetPhyAddr(), (__ubuf__ T *)srcLocal.GetPhyAddr());
    } else if (transposeParams.transposeType == TransposeType::TRANSPOSE_NCHW2NHWC ||
        transposeParams.transposeType == TransposeType::TRANSPOSE_NHWC2NCHW) {
        ASCENDC_ASSERT((SupportType<T, int8_t, uint8_t, int16_t, uint16_t, half, int32_t, uint32_t, float>()),
            {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Transpose when transposeType is TRANSPOSE_NCHW2NHWC / "
            "TRANSPOSE_NHWC2NCHW, current api support dtype combination is src and dst both: int8_t / uint8_t / int16_t"
            " / uint16_t / half / int32_t / uint32_t / float");});
        if (transposeParams.cSize == 1) {
            struct DataCopyParams repeatParams;
            repeatParams.blockLen = transposeParams.nSize * transposeParams.cSize * transposeParams.hSize *
                transposeParams.wSize / AscendCUtils::GetC0Count(sizeof(T));
            DataCopyUB2UBImpl((__ubuf__ T *)dstLocal.GetPhyAddr(), (__ubuf__ T *)srcLocal.GetPhyAddr(), repeatParams);
        } else {
#if ASCENDC_CPU_DEBUG
            uint32_t imageSize = transposeParams.hSize * transposeParams.wSize;  // uint16 * uint16
            ASCENDC_CHECK_VALUE_RANGE(transposeParams.cSize, 0, UINT12_MAX, "cSize", "Transpose");
            ASCENDC_CHECK_VALUE_RANGE(imageSize, 0, UINT12_MAX, "hSize * wSize", "Transpose");
            ASCENDC_ASSERT(((imageSize * sizeof(T)) % ONE_BLK_SIZE == 0), {KERNEL_LOG(KERNEL_ERROR, "Failed to check "
                "hSize, wSize value in Transpose when transposeType is TRANSPOSE_NCHW2NHWC / TRANSPOSE_NHWC2NCHW, "
                "hSize * wSize * sizeof(T) should be 32B aligned, current value is %u.", imageSize * sizeof(T));});
#endif
            Transpose4DImpl(dstLocal, srcLocal, sharedTmpBuffer, transposeParams);
        }
    }
}
#pragma end_pipe
template <typename T>
__aicore__ inline __in_pipe__(S) __out_pipe__(V) void TransDataTo5HD(const LocalTensor<uint64_t> &dstLocal,
    const LocalTensor<uint64_t> &srcLocal, const TransDataTo5HDParams &nchwconvParams)
{
    ASCENDC_ASSERT((SupportType<T, int8_t, uint8_t, int16_t, uint16_t, half, float, int32_t, uint32_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in TransDataTo5HD, current api support dtype combination is "
        "src and dst both: int8_t, uint8_t, int16_t, uint16_t, half, float, int32_t, uint32_t");});
#if ASCENDC_CPU_DEBUG
    if (!CheckFunTransDataTo5HD<T, uint64_t>(dstLocal, srcLocal, nchwconvParams, "TransDataTo5HD")) {
        ASCENDC_REPORT_CHECK_ERROR("TransDataTo5HD", KernelFuncType::NONE_MODE);
    }
    TransDataTo5HDVldVaRegImpl<T>(
        (__ubuf__ uint64_t*)dstLocal.GetPhyAddr(), (__ubuf__ uint64_t*)srcLocal.GetPhyAddr(), nchwconvParams);
#else
    constexpr uint32_t vaRegSize = VA_REG_ARRAY_LEN / HALF_FACTOR;
    constexpr uint32_t vaOne = 1;
    constexpr uint32_t vaTwo = 2;
    constexpr uint32_t vaThree = 3;
    constexpr uint64_t vaAddr = 5;
    constexpr uint64_t vaMask = 0x1fff;
    constexpr uint64_t vaBit1 = 16;
    constexpr uint64_t vaBit2 = 32;
    constexpr uint64_t vaBit3 = 48;

    for (uint32_t i = 0; i < vaRegSize; i++)
    {
        uint64_t dstAddrConfig = (((dstLocal.GetValue(vaRegSize * i) >> vaAddr) & vaMask) |
                                  (((dstLocal.GetValue(vaRegSize * i + vaOne) >> vaAddr) & vaMask) << vaBit1) |
                                  (((dstLocal.GetValue(vaRegSize * i + vaTwo) >> vaAddr) & vaMask) << vaBit2) |
                                  (((dstLocal.GetValue(vaRegSize * i + vaThree) >> vaAddr) & vaMask) << vaBit3));
        dstLocal.SetValue(i, dstAddrConfig);

        uint64_t srcAddrConfig = (((srcLocal.GetValue(vaRegSize * i) >> vaAddr) & vaMask) |
                                  (((srcLocal.GetValue(vaRegSize * i + vaOne) >> vaAddr) & vaMask) << vaBit1) |
                                  (((srcLocal.GetValue(vaRegSize * i + vaTwo) >> vaAddr) & vaMask) << vaBit2) |
                                  (((srcLocal.GetValue(vaRegSize * i + vaThree) >> vaAddr) & vaMask) << vaBit3));
        srcLocal.SetValue(i, srcAddrConfig);
    }

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    TransDataTo5HDVldVaRegImpl<T>(
        (__ubuf__ uint64_t*)dstLocal.GetPhyAddr(), (__ubuf__ uint64_t*)srcLocal.GetPhyAddr(), nchwconvParams);
#endif
}
} // namespace AscendC
#endif // ASCENDC_MODULE_INNER_OPERATOR_VEC_TRANSPOSE_INTERFACE_H
