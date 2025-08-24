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
 * \file kernel_operator_mm_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_MM_IMPL_H
#define ASCENDC_MODULE_OPERATOR_MM_IMPL_H
#include "kernel_struct_mm.h"

namespace AscendC {
/* **************************************************************************************************
 * LoadData 2dv2                                             *
 * ************************************************************************************************* */

template <typename T>
__aicore__ inline void CheckLoadData2DType()
{
    ASCENDC_ASSERT((SupportType<T, uint8_t, int8_t, uint16_t, int16_t, half>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to "
        "check dtype in LoadData with LoadData2DParams, current api support dtype combination is src and dst both: "
        "uint8_t, int8_t, uint16_t, int16_t, half.");});
}

template <typename T>
__aicore__ inline void LoadData2DL12L0ACal(__ca__ T* dst, __cbuf__ T* src, const LoadData2DParams& loadDataParam)
{
#if defined(__DAV_M200_VEC__)
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData from A1 to A2");
#else
    CheckLoadData2DType<T>();
    if (loadDataParam.ifTranspose) {
        load_cbuf_to_ca(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes, loadDataParam.srcStride,
            loadDataParam.dstGap, loadDataParam.sid, 1, inc);
    } else {
        load_cbuf_to_ca(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes, loadDataParam.srcStride,
            loadDataParam.dstGap, loadDataParam.sid, 0, inc);
    }
#endif
}

template <typename T>
__aicore__ inline void LoadData2DL12L0BCal(__cb__ T* dst, __cbuf__ T* src, const LoadData2DParams& loadDataParam)
{
#if defined(__DAV_M200_VEC__)
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData from B1 to B2");
#else
    CheckLoadData2DType<T>();
    if (loadDataParam.ifTranspose) {
        load_cbuf_to_cb(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes, loadDataParam.srcStride,
            loadDataParam.dstGap, loadDataParam.sid, 1, inc);
    } else {
        load_cbuf_to_cb(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes, loadDataParam.srcStride,
            loadDataParam.dstGap, loadDataParam.sid, 0, inc);
    }
#endif
}

template <typename T>
__aicore__ inline void LoadData2DGM2L0ACal(__ca__ T* dst, __gm__ T* src, const LoadData2DParams& loadDataParam)
{
#if defined(__DAV_M200_VEC__)
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData from GM to A2");
#else
    CheckLoadData2DType<T>();
    load_gm_to_ca(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes, loadDataParam.srcStride,
        loadDataParam.dstGap, loadDataParam.sid, (addr_cal_mode_t)0);
#endif
}

template <typename T>
__aicore__ inline void LoadData2DGM2L0BCal(__cb__ T* dst, __gm__ T* src, const LoadData2DParams& loadDataParam)
{
#if defined(__DAV_M200_VEC__)
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData from GM to B2");
#else
    CheckLoadData2DType<T>();
    load_gm_to_cb(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes, loadDataParam.srcStride,
        loadDataParam.dstGap, loadDataParam.sid, (addr_cal_mode_t)0);
#endif
}

template <typename T>
__aicore__ inline void LoadData2DGM2L1Cal(__cbuf__ T* dst, __gm__ T* src, const LoadData2DParams& loadDataParam)
{
#if defined(__DAV_M200_VEC__)
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData from GM to A1 / B1");
#else
    CheckLoadData2DType<T>();
    if (loadDataParam.addrMode == 0) {
        load_gm_to_cbuf(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes, loadDataParam.srcStride,
            loadDataParam.dstGap, loadDataParam.sid, inc);
    } else {
        load_gm_to_cbuf(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes, loadDataParam.srcStride,
            loadDataParam.dstGap, loadDataParam.sid, dec);
    }
#endif
}

template <typename T>
__aicore__ inline void LoadData2DL12L0ACal(__ca__ T *dst, __cbuf__ T *src, const LoadData2DParamsV2 &loadDataParam)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData2DParamsV2");
}

template <typename T>
__aicore__ inline void LoadData2DL12L0BCal(__cb__ T *dst, __cbuf__ T *src, const LoadData2DParamsV2 &loadDataParam)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData2DParamsV2");
}

template <typename T>
__aicore__ inline void LoadData2DGM2L0ACal(__ca__ T *dst, __gm__ T *src, const LoadData2DParamsV2 &loadDataParam)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData2DParamsV2");
}

template <typename T>
__aicore__ inline void LoadData2DGM2L0BCal(__cb__ T *dst, __gm__ T *src, const LoadData2DParamsV2 &loadDataParam)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData2DParamsV2");
}

template <typename T>
__aicore__ inline void LoadData2DGM2L1Cal(__cbuf__ T *dst, __gm__ T *src, const LoadData2DParamsV2 &loadDataParam)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData2DParamsV2");
}

template <typename T>
__aicore__ inline void LoadData2DL12L0ATransposeCal(__ca__ T *dst, __cbuf__ T *src,
    const LoadData2dTransposeParams &loadDataParam)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadDataWithTranspose");
}

template <typename T>
__aicore__ inline void LoadData2DL12L0BTransposeCal(__cb__ T *dst, __cbuf__ T *src,
    const LoadData2dTransposeParams &loadDataParam)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadDataWithTranspose");
}

template <typename T>
__aicore__ inline void LoadData2DL12L0BTransposeCal(__cb__ T *dst, __cbuf__ T *src,
    const LoadData2dTransposeParamsV2 &loadDataParam)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadDataWithTranspose");
}

/* **************************************************************************************************
 * LoadData 3dv1                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadData3DV1L12L0ACal(__ca__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV1<T>& loadDataParams)
{
    if (loadDataParams.cSize == 0) {
        img2col_cbuf_to_ca(dst, src, loadDataParams.fetchFilterW, loadDataParams.fetchFilterH, loadDataParams.leftTopW,
            loadDataParams.leftTopH, loadDataParams.c1Index, loadDataParams.strideW, loadDataParams.strideH,
            loadDataParams.filterW, loadDataParams.filterH, loadDataParams.dilationFilterW,
            loadDataParams.dilationFilterH, loadDataParams.jumpStride, loadDataParams.repeatMode,
            loadDataParams.repeatTime, CSIZE0);
    } else {
        img2col_cbuf_to_ca(dst, src, loadDataParams.fetchFilterW, loadDataParams.fetchFilterH, loadDataParams.leftTopW,
            loadDataParams.leftTopH, loadDataParams.c1Index, loadDataParams.strideW, loadDataParams.strideH,
            loadDataParams.filterW, loadDataParams.filterH, loadDataParams.dilationFilterW,
            loadDataParams.dilationFilterH, loadDataParams.jumpStride, loadDataParams.repeatMode,
            loadDataParams.repeatTime, CSIZE1);
    }
}

template <typename T>
__aicore__ inline void LoadData3DV1L12L0BCal(__cb__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV1<T>& loadDataParams)
{
    if (loadDataParams.cSize == 0) {
        img2col_cbuf_to_cb(dst, src, loadDataParams.fetchFilterW, loadDataParams.fetchFilterH, loadDataParams.leftTopW,
            loadDataParams.leftTopH, loadDataParams.c1Index, loadDataParams.strideW, loadDataParams.strideH,
            loadDataParams.filterW, loadDataParams.filterH, loadDataParams.dilationFilterW,
            loadDataParams.dilationFilterH, loadDataParams.jumpStride, loadDataParams.repeatMode,
            loadDataParams.repeatTime, CSIZE0);
    } else {
        img2col_cbuf_to_cb(dst, src, loadDataParams.fetchFilterW, loadDataParams.fetchFilterH, loadDataParams.leftTopW,
            loadDataParams.leftTopH, loadDataParams.c1Index, loadDataParams.strideW, loadDataParams.strideH,
            loadDataParams.filterW, loadDataParams.filterH, loadDataParams.dilationFilterW,
            loadDataParams.dilationFilterH, loadDataParams.jumpStride, loadDataParams.repeatMode,
            loadDataParams.repeatTime, CSIZE1);
    }
}

template <typename T>
__aicore__ inline void LoadData3DV1L12UBCal(__ubuf__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV1<T>& loadDataParams)
{
    if (loadDataParams.cSize == 0) {
        img2col_cbuf_to_ub(dst, src, loadDataParams.fetchFilterW, loadDataParams.fetchFilterH, loadDataParams.leftTopW,
            loadDataParams.leftTopH, loadDataParams.c1Index, loadDataParams.strideW, loadDataParams.strideH,
            loadDataParams.filterW, loadDataParams.filterH, loadDataParams.dilationFilterW,
            loadDataParams.dilationFilterH, loadDataParams.jumpStride, loadDataParams.repeatMode,
            loadDataParams.repeatTime, CSIZE0);
    } else {
        img2col_cbuf_to_ub(dst, src, loadDataParams.fetchFilterW, loadDataParams.fetchFilterH, loadDataParams.leftTopW,
            loadDataParams.leftTopH, loadDataParams.c1Index, loadDataParams.strideW, loadDataParams.strideH,
            loadDataParams.filterW, loadDataParams.filterH, loadDataParams.dilationFilterW,
            loadDataParams.dilationFilterH, loadDataParams.jumpStride, loadDataParams.repeatMode,
            loadDataParams.repeatTime, CSIZE1);
    }
}

/* **************************************************************************************************
 * LoadData 3dv2                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadData3DV2L12L0ACal(__ca__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV2<T>& loadDataParams)
{
#if defined(__DAV_M200_VEC__)
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData3DParamsV2 from A1 to A2");
#else
    img2colv2_cbuf_to_ca(dst, src, loadDataParams.kExtension, loadDataParams.mExtension, loadDataParams.kStartPt,
        loadDataParams.mStartPt, loadDataParams.strideW, loadDataParams.strideH, loadDataParams.filterW,
        loadDataParams.filterH, loadDataParams.dilationFilterW, loadDataParams.dilationFilterH,
        loadDataParams.enTranspose, loadDataParams.channelSize);
#endif
}

template <typename T>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV2<T>& loadDataParams)
{
#if defined(__DAV_M200_VEC__)
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData3DParamsV2 from B1 to B2");
#else
    img2colv2_cbuf_to_cb(dst, src, loadDataParams.kExtension, loadDataParams.mExtension, loadDataParams.kStartPt,
        loadDataParams.mStartPt, loadDataParams.strideW, loadDataParams.strideH, loadDataParams.filterW,
        loadDataParams.filterH, loadDataParams.dilationFilterW, loadDataParams.dilationFilterH,
        loadDataParams.enTranspose, loadDataParams.channelSize);
#endif
}

template <typename T>
__aicore__ inline void LoadData3DV2L12UBCal(__ubuf__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV2<T>& loadDataParams)
{
    img2colv2_cbuf_to_ub(dst, src, loadDataParams.kExtension, loadDataParams.mExtension, loadDataParams.kStartPt,
        loadDataParams.mStartPt, loadDataParams.strideW, loadDataParams.strideH, loadDataParams.filterW,
        loadDataParams.filterH, loadDataParams.dilationFilterW, loadDataParams.dilationFilterH,
        loadDataParams.enTranspose, loadDataParams.channelSize);
}

/* **************************************************************************************************
 * LoadData 3dv2Pro                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadData3DV2L12L0ACal(__ca__ T *dst, __cbuf__ T *src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
#if defined(__DAV_M200_VEC__)
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData3DParamsV2Pro from A1 to A2");
#else
    img2colv2_cbuf_to_ca(dst, src, loadDataParams.extConfig, loadDataParams.extConfig >> LOAD_M_EXTENSION,
        loadDataParams.extConfig >> LOAD_K_START_POSITION, loadDataParams.extConfig >> LOAD_M_START_POSITION,
        loadDataParams.filterConfig, loadDataParams.filterConfig >> LOAD_STRIDE_H,
        loadDataParams.filterConfig >> LOAD_FILTER_W, loadDataParams.filterConfig >> LOAD_FILTER_H,
        loadDataParams.filterConfig >> LOAD_DILATION_FILTER_W,
        loadDataParams.filterConfig >> LOAD_DILATION_FILTER_H, loadDataParams.enTranspose,
        loadDataParams.channelSize);
#endif
}

template <typename T>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ T *dst, __cbuf__ T *src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
#if defined(__DAV_M200_VEC__)
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData3DParamsV2Pro from B1 to B2");
#else
    img2colv2_cbuf_to_cb(dst, src, loadDataParams.extConfig, loadDataParams.extConfig >> LOAD_M_EXTENSION,
        loadDataParams.extConfig >> LOAD_K_START_POSITION, loadDataParams.extConfig >> LOAD_M_START_POSITION,
        loadDataParams.filterConfig, loadDataParams.filterConfig >> LOAD_STRIDE_H,
        loadDataParams.filterConfig >> LOAD_FILTER_W, loadDataParams.filterConfig >> LOAD_FILTER_H,
        loadDataParams.filterConfig >> LOAD_DILATION_FILTER_W,
        loadDataParams.filterConfig >> LOAD_DILATION_FILTER_H, loadDataParams.enTranspose,
        loadDataParams.channelSize);
#endif
}

template <typename T>
__aicore__ inline void LoadData3DV2L12UBCal(__ubuf__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData3DParamsV2Pro from L1 to UB");
}

template <>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ int8_t* dst, __cbuf__ int8_t* src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData3DParamsV2Pro from B1 to B2 with type int8_t");
}

template <>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ uint8_t* dst, __cbuf__ uint8_t* src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData3DParamsV2Pro from B1 to B2 with type uint8_t");
}

/* **************************************************************************************************
 * Mmad                                             *
 * ************************************************************************************************* */
template <typename DstT, typename Src0T, typename Src1T>
__aicore__ inline void MmadCal(__cc__ DstT* c, __ca__ Src0T* a, __cb__ Src1T* b, const MmadParams& mmadParams)
{
    ASCENDC_ASSERT(mmadParams.cmatrixSource == 0,
                   { KERNEL_LOG(KERNEL_ERROR, "the C matrix source doesn't support BT buffer on current device"); });
    bool cmatrixInitVal = mmadParams.cmatrixInitVal && (!mmadParams.isBias);
#if defined(__DAV_M200_VEC__)
    ASCENDC_REPORT_NOT_SUPPORT(false, "Mmad");
#else
    ASCENDC_ASSERT((SupportType<Tuple<DstT, Src0T, Src1T>,
        Tuple<int32_t, int8_t, int8_t>, Tuple<int32_t, uint8_t, int8_t>, Tuple<int32_t, uint8_t, uint8_t>,
        Tuple<half, half, half>, Tuple<float, half, half>>()), {KERNEL_LOG(KERNEL_ERROR,
        "Failed to check dtype in Mmad, current api support dtype combination is Dst: int32_t, src0: int8_t, "
        "src1: int8_t; Dst: int32_t, src0: uint8_t, src1: int8_t; Dst: int32_t, src0: uint8_t, src1: uint8_t; "
        "Dst: half, src0: half, src1: half; Dst: float, src0: half, src1: half");});
    mad(c, a, b, mmadParams.m, mmadParams.k, mmadParams.n, cmatrixInitVal);
#endif
}

template <typename DstT, typename Src0T, typename Src1T>
__aicore__ inline void MmadCal(__cc__ DstT* c, __ca__ Src0T* a, __cb__ Src1T* b, uint64_t bias,
    const MmadParams& mmadParams, bool cmatrixSource)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Mmad with biasLocal");
}

__aicore__ inline void MmadSpCal(__cc__ int32_t *c, __ca__ int8_t *a, __cb__ int8_t *b, const MmadParams &mmadParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "MmadWithSparse");
}

template <typename T = int8_t, typename U = uint8_t,
    typename std::enable_if<IsSameType<PrimT<T>, int8_t>::value, bool>::type = true,
    typename std::enable_if<IsSameType<PrimT<U>, uint8_t>::value, bool>::type = true>
__aicore__ inline void LoadDataWithSparseCal(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const LocalTensor<U> &idxLocal, const LoadData2dParams &loadDataParam)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadDataWithSparse");
}

template <typename T = int8_t, typename std::enable_if<IsSameType<PrimT<T>, int8_t>::value, bool>::type = true> 
__aicore__ inline void LoadUnzipIndexCal(const GlobalTensor<T> &srcTensor, uint32_t numOfIndexTabEntry)
{
#if ASCENDC_CPU_DEBUG
    const __gm__ int64_t *srcAddr = reinterpret_cast<const __gm__ int64_t *>(srcTensor.GetPhyAddr());
    constexpr uint32_t unitOfLength = 128;
    for (int i = 0; i < numOfIndexTabEntry; i++) {
        uint32_t compressedLength = *(srcAddr + i) & 0x1fff;
        SetUnzipCompressedLen(compressedLength * unitOfLength);
    }
    constexpr uint32_t maxSize = 32 * 1024;        // max is 32 * 1024  = 32KB
    ASCENDC_ASSERT((srcTensor.GetSize() % VALUE_512 == 0 && srcTensor.GetSize() <= maxSize), {KERNEL_LOG(KERNEL_ERROR,
        "Failed to check srcTensor size in LoadUnzipIndex, it must be less than 32KB and divisible by 512, current "
        "value is %u.", srcTensor.GetSize());});
    ASCENDC_CHECK_VALUE_RANGE(numOfIndexTabEntry, 1, UINT32_MAX, "numOfIndexTabEntry", "LoadUnzipIndex");
#endif
    load_unzip_index_from_gm((__gm__ void *)srcTensor.GetPhyAddr(), numOfIndexTabEntry, 0);
}

/* **************************************************************************************************
 * BroadCastVecToMM                                             *
 * ************************************************************************************************* */
template <typename T, typename U>
__aicore__ inline void BroadCastVecToMMCal(__cc__ T* dstLocal, __ubuf__ U* srcLocal, const int32_t blockCount,
    const uint8_t blockLen, const uint8_t srcGap, const uint8_t dstGap)
{
#if defined(__DAV_M200_VEC__)
    ASCENDC_REPORT_NOT_SUPPORT(false, "BroadCastVecToMM");
#else
    ASCENDC_CHECK_VALUE_RANGE(blockCount, 1, 255, "blockCount", "BroadCastVecToMM");
    ASCENDC_CHECK_VALUE_RANGE(blockLen, 1, 255, "blockLen", "BroadCastVecToMM");
    ASCENDC_ASSERT((SupportType<Tuple<T, U>, Tuple<int32_t, int32_t>, Tuple<float, float>, Tuple<half, half>>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in BroadCastVecToMM, current api support dtype combination is "
        "src and dst both: half / float / int32_t.");});
    broadcast_ub_to_cc(dstLocal, srcLocal, blockCount, blockLen, srcGap, dstGap);
#endif
}

/* **************************************************************************************************
 * LoadData 3dv1                                             *
 * ************************************************************************************************* */
__aicore__ inline void Load3DSetFMatrixCal(uint16_t l1H, uint16_t l1W, const uint8_t padList[4])
{
    uint64_t regFMatrix = 0;
    regFMatrix |= uint64_t(l1W & 0xFFFF);

    uint32_t l1HShiftBit = 16;
    regFMatrix |= uint64_t(l1H & 0xFFFF) << l1HShiftBit;

    uint32_t padNumber = 4;
    uint32_t padListShiftBit = 8;
    uint32_t padListShiftBase = 32;
    for (uint32_t i = 0; i < padNumber; i++) {
        regFMatrix |= uint64_t(padList[i] & 0xFF) << (padListShiftBase + i * padListShiftBit);
    }
    set_fmatrix(regFMatrix);
}

__aicore__ inline void Load3DSetFMatrixBCal(uint16_t l1H, uint16_t l1W, const uint8_t padList[4])
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "SetFMatrix with fmatrixMode = FMATRIX_RIGHT");
}

template <typename T>
__aicore__ inline void Load3DSetPaddingCal(const T padValue)
{
    uint16_t paddingValue = 0;
    uint16_t padValueShiftBit = 8;
    if constexpr (sizeof(T) == B16_BYTE_SIZE) {
        paddingValue = (uint16_t)GetScalarBitcodeValue((T)padValue);
    } else if constexpr (sizeof(T) == B32_BYTE_SIZE) {
        ASCENDC_ASSERT(false, {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in SetLoadDataPaddingValue, current "
            "api support dtype combination is: int8_t / uint8_t / half / int16_t / uint16_t.");});
    } else {
        paddingValue = (((uint16_t)padValue) << padValueShiftBit) | (uint16_t)padValue;
    }
    set_padding(paddingValue);
}

/* **************************************************************************************************
 * InitL1Buffer                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void InitL1BufferCal(__cbuf__ T *dst, const InitConstValueParams<T> &initConstValueParams)
{
#if defined(__DAV_M200_VEC__)
    ASCENDC_REPORT_NOT_SUPPORT(false, "InitConstValue in A1 / B1");
#else
    ASCENDC_CHECK_VALUE_RANGE(initConstValueParams.repeatTimes, 0, UINT8_MAX, "repeatTimes", "InitConstValue");
    int64_t repeatBit = initConstValueParams.repeatTimes;
    if constexpr (IsSameType<T, half>::value) {
        create_cbuf_matrix((__cbuf__ void *)dst, repeatBit, (half)initConstValueParams.initValue);
    } else if constexpr  (IsSameType<T, int16_t>::value || IsSameType<T, uint16_t>::value) {
        create_cbuf_matrix((__cbuf__ void *)dst, repeatBit,
            GetScalarBitcodeToHalf(initConstValueParams.initValue));
    } else {
        ASCENDC_ASSERT(false, {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in InitConstValue, current api "
            "support dtype combination is dst: half, int16_t, uint16_t");});
    }
#endif
}

/* **************************************************************************************************
 * InitL0ANzMatrix                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void InitL0ANzMatrixCal(__ca__ T *dst, const InitConstValueParams<T> &initConstValueParams)
{
#if defined(__DAV_M200_VEC__)
    ASCENDC_REPORT_NOT_SUPPORT(false, "InitConstValue in A2");
#else
    ASCENDC_CHECK_VALUE_RANGE(initConstValueParams.repeatTimes, 0, UINT8_MAX, "repeatTimes", "InitConstValue");
    int64_t repeatBit = initConstValueParams.repeatTimes;
    if constexpr (IsSameType<T, half>::value) {
        create_ca_matrix((__ca__ void *)dst, repeatBit, (half)initConstValueParams.initValue);
    } else if constexpr (IsSameType<T, int16_t>::value || IsSameType<T, uint16_t>::value) {
        create_ca_matrix((__ca__ void *)dst, repeatBit, GetScalarBitcodeToHalf(initConstValueParams.initValue));
    } else {
        ASCENDC_ASSERT(false, {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in InitConstValue, current api "
            "support dtype combination is dst: half, int16_t, uint16_t");});
    }
#endif
}

/* **************************************************************************************************
 * InitL0BNzMatrix                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void InitL0BNzMatrixCal(__cb__ T *dst, const InitConstValueParams<T> &initConstValueParams)
{
#if defined(__DAV_M200_VEC__)
    ASCENDC_REPORT_NOT_SUPPORT(false, "InitConstValue in B2");
#else
    ASCENDC_CHECK_VALUE_RANGE(initConstValueParams.repeatTimes, 0, UINT8_MAX, "repeatTimes", "InitConstValue");
    int64_t repeatBit = initConstValueParams.repeatTimes;
    if constexpr (IsSameType<T, half>::value) {
        create_cb_matrix((__cb__ void *)dst, repeatBit, (half)initConstValueParams.initValue);
    } else if constexpr (IsSameType<T, int16_t>::value || IsSameType<T, uint16_t>::value) {
        create_cb_matrix((__cb__ void *)dst, repeatBit, GetScalarBitcodeToHalf(initConstValueParams.initValue));
    } else {
        ASCENDC_ASSERT(false, {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in InitConstValue, current api "
            "support dtype combination is dst: half, int16_t, uint16_t");});
    }
#endif
}

/* **************************************************************************************************
 * SetLoadDataRepeat                                             *
 * ************************************************************************************************* */
__aicore__ inline void SetLoadDataRepeatCal(const LoadDataRepeatParam& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "SetLoadDataRepeat");
}

/* **************************************************************************************************
 * SetLoadDataBoundary                                             *
 * ************************************************************************************************* */
__aicore__ inline void SetLoadDataBoundaryCal(uint32_t boundaryValue)
{
    set_l1_3d_size(static_cast<uint64_t>(boundaryValue));
}

/* **************************************************************************************************
 * LoadImageToLocalCal                                            *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadImageToLocalCal(__cbuf__ T *dst, const LoadImageToLocalParams &loadDataParams)
{
    ASCENDC_ASSERT((SupportType<T, uint8_t, int8_t, half>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "LoadImageToLocal, current api support dtype combination is dst: uint8_t / int8_t / half.");});
    load_image_to_cbuf(dst, static_cast<uint16_t>(loadDataParams.horizSize - 1),
        static_cast<uint16_t>(loadDataParams.vertSize - 1), loadDataParams.horizStartPos, loadDataParams.vertStartPos,
        static_cast<uint16_t>(loadDataParams.srcHorizSize - 1), loadDataParams.topPadSize, loadDataParams.botPadSize,
        loadDataParams.leftPadSize, loadDataParams.rightPadSize, loadDataParams.sid);
}

/* **************************************************************************************************
 * LoadDataUnzip                                            *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadDataUnzipToL1Cal(__cbuf__ T *dst, __gm__ T *src)
{
    load_gm_to_cbuf_unzip((__cbuf__ void *)dst, (__gm__ void *)src);
}

template <typename T>
__aicore__ inline void LoadDataUnzipToL0BCal(__cb__ T *dst, __gm__ T *src)
{
    load_gm_to_cb_unzip((__cb__ void *)dst, (__gm__ void *)src);
}

template <typename T>
__aicore__ inline void LoadDataUnzipToL0ACal(__ca__ T *dst, __gm__ T *src)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadDataUnzip to A2");
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_MM_IMPL_H
