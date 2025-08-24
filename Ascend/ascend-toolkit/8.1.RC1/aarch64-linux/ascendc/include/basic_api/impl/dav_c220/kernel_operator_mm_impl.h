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
__aicore__ inline void LoadData2DL12L0ACal(__ca__ T* dst, __cbuf__ T* src, const LoadData2DParams& loadDataParam)
{
    if ASCEND_IS_AIC {
        if constexpr (IsSameType<T, int4b_t>::value) {
            load_cbuf_to_ca_s4((__ca__ void *)dst, (__cbuf__ void *)src, loadDataParam.startIndex,
                loadDataParam.repeatTimes, loadDataParam.srcStride, loadDataParam.dstGap, loadDataParam.sid, 0, inc);
        } else {
            if (loadDataParam.ifTranspose) {
                load_cbuf_to_ca(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes, loadDataParam.srcStride,
                    loadDataParam.dstGap, loadDataParam.sid, 1, inc);
            } else {
                load_cbuf_to_ca(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes, loadDataParam.srcStride,
                    loadDataParam.dstGap, loadDataParam.sid, 0, inc);
            }
        }
    }
}

template <typename T>
__aicore__ inline void LoadData2DL12L0BCal(__cb__ T* dst, __cbuf__ T* src, const LoadData2DParams& loadDataParam)
{
    if ASCEND_IS_AIC {
        if constexpr (IsSameType<T, int4b_t>::value) {
            load_cbuf_to_cb_s4((__cb__ void *)dst, (__cbuf__ void *)src, loadDataParam.startIndex,
                loadDataParam.repeatTimes, loadDataParam.srcStride, loadDataParam.dstGap, loadDataParam.sid, 0, inc);
        } else {
            if (loadDataParam.ifTranspose) {
                load_cbuf_to_cb(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes, loadDataParam.srcStride,
                    loadDataParam.dstGap, loadDataParam.sid, 1, inc);
            } else {
                load_cbuf_to_cb(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes, loadDataParam.srcStride,
                    loadDataParam.dstGap, loadDataParam.sid, 0, inc);
            }
        }
    }
}

template <typename T>
__aicore__ inline void LoadData2DGM2L0ACal(__ca__ T* dst, __gm__ T* src, const LoadData2DParams& loadDataParam)
{
    if ASCEND_IS_AIC {
        load_gm_to_ca(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes, loadDataParam.srcStride,
            loadDataParam.dstGap, loadDataParam.sid, (addr_cal_mode_t)0);
    }
}

template <typename T>
__aicore__ inline void LoadData2DGM2L0BCal(__cb__ T* dst, __gm__ T* src, const LoadData2DParams& loadDataParam)
{
    if ASCEND_IS_AIC {
        load_gm_to_cb(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes, loadDataParam.srcStride,
            loadDataParam.dstGap, loadDataParam.sid, (addr_cal_mode_t)0);
    }
}

template <typename T>
__aicore__ inline void LoadData2DGM2L1Cal(__cbuf__ T* dst, __gm__ T* src, const LoadData2DParams& loadDataParam)
{
    if ASCEND_IS_AIC {
        if (loadDataParam.addrMode == 0) {
            load_gm_to_cbuf(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes, loadDataParam.srcStride,
                loadDataParam.dstGap, loadDataParam.sid, inc);
        } else {
            load_gm_to_cbuf(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes, loadDataParam.srcStride,
                loadDataParam.dstGap, loadDataParam.sid, dec);
        }
    }
}

template <typename T>
__aicore__ inline void LoadData2DL12L0ACal(__ca__ T *dst, __cbuf__ T *src, const LoadData2DParamsV2 &loadDataParam)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData2DParamsV2 from A1 to A2");
}

template <typename T>
__aicore__ inline void LoadData2DL12L0BCal(__cb__ T *dst, __cbuf__ T *src, const LoadData2DParamsV2 &loadDataParam)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData2DParamsV2 from B1 to B2");
}

template <typename T>
__aicore__ inline void LoadData2DGM2L0ACal(__ca__ T *dst, __gm__ T *src, const LoadData2DParamsV2 &loadDataParam)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData2DParamsV2 from GM to A2");
}

template <typename T>
__aicore__ inline void LoadData2DGM2L0BCal(__cb__ T *dst, __gm__ T *src, const LoadData2DParamsV2 &loadDataParam)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData2DParamsV2 from GM to B2");
}

template <typename T>
__aicore__ inline void LoadData2DGM2L1Cal(__cbuf__ T *dst, __gm__ T *src, const LoadData2DParamsV2 &loadDataParam)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData2DParamsV2 from GM to A1 / B1");
}

template <typename T>
__aicore__ inline void LoadData2DL12L0ATransposeCal(__ca__ T *dst, __cbuf__ T *src,
    const LoadData2dTransposeParams &loadDataParam)
{
    if ASCEND_IS_AIC {
        ASCENDC_ASSERT((SupportType<T, half, bfloat16_t, float, int32_t, uint32_t, uint8_t, int8_t>()),
            {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in LoadDataWithTranspose when dst position is A2, current "
            "api support dtype combination is dst: half, bfloat16_t, float, int32_t, uint32_t, uint8_t, int8_t");});
        if constexpr (!IsSameType<T, int4b_t>::value) {
            load_cbuf_to_ca_transpose(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes,
                loadDataParam.srcStride, loadDataParam.dstGap, inc, loadDataParam.dstFracGap);
        }
    }
}

template <typename T>
__aicore__ inline void LoadData2DL12L0BTransposeCal(__cb__ T *dst, __cbuf__ T *src,
    const LoadData2dTransposeParams &loadDataParam)
{
    if ASCEND_IS_AIC {
        ASCENDC_ASSERT((SupportType<T, half, bfloat16_t, float, int32_t, uint32_t, uint8_t, int8_t, int4b_t>()),
            {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in LoadDataWithTranspose when dst position is B2, current "
            "api support dtype combination is dst: half, bfloat16_t, float, int32_t, uint32_t, uint8_t, int8_t, "
            "int4b_t");});
        if constexpr (IsSameType<T, int4b_t>::value) {
            load_cbuf_to_cb_transpose_s4((__cb__ void *)dst, (__cbuf__ void *)src, loadDataParam.startIndex,
                loadDataParam.repeatTimes, loadDataParam.srcStride, loadDataParam.dstGap, inc,
                loadDataParam.dstFracGap);
        } else {
            load_cbuf_to_cb_transpose(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes,
                loadDataParam.srcStride, loadDataParam.dstGap, inc, loadDataParam.dstFracGap);
        }
    }
}

template <typename T>
__aicore__ inline void LoadData2DL12L0BTransposeCal(__cb__ T *dst, __cbuf__ T *src,
    const LoadData2dTransposeParamsV2 &loadDataParam)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadDataWithTranspose with LoadData2dTransposeParamsV2 from B1 to B2");
}

/* **************************************************************************************************
 * LoadData 3dv2                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadData3DV2L12L0ACal(__ca__ T *dst, __cbuf__ T *src,
    const LoadData3DParamsV2<T> &loadDataParams)
{
    if ASCEND_IS_AIC {
        if constexpr (IsSameType<T, int4b_t>::value) {
            img2colv2_cbuf_to_ca_s4((__ca__ void *)dst, (__cbuf__ void *)src, loadDataParams.kExtension,
                loadDataParams.mExtension, loadDataParams.kStartPt, loadDataParams.mStartPt, loadDataParams.strideW,
                loadDataParams.strideH, loadDataParams.filterW, loadDataParams.filterH, loadDataParams.dilationFilterW,
                loadDataParams.dilationFilterH, loadDataParams.filterSizeW, loadDataParams.filterSizeH,
                loadDataParams.enTranspose, loadDataParams.fMatrixCtrl, loadDataParams.channelSize);
        } else {
            img2colv2_cbuf_to_ca(dst, src, loadDataParams.kExtension, loadDataParams.mExtension,
                loadDataParams.kStartPt, loadDataParams.mStartPt, loadDataParams.strideW, loadDataParams.strideH,
                loadDataParams.filterW, loadDataParams.filterH, loadDataParams.dilationFilterW,
                loadDataParams.dilationFilterH, loadDataParams.filterSizeW, loadDataParams.filterSizeH,
                loadDataParams.enTranspose, loadDataParams.fMatrixCtrl, loadDataParams.channelSize);
        }
    }
}

template <typename T>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ T *dst, __cbuf__ T *src,
    const LoadData3DParamsV2<T> &loadDataParams)
{
    if ASCEND_IS_AIC {
        if constexpr (!IsSameType<T, int4b_t>::value) {
            img2colv2_cbuf_to_cb(dst, src, loadDataParams.kExtension, loadDataParams.mExtension,
                loadDataParams.kStartPt, loadDataParams.mStartPt, loadDataParams.strideW, loadDataParams.strideH,
                loadDataParams.filterW, loadDataParams.filterH, loadDataParams.dilationFilterW,
                loadDataParams.dilationFilterH, loadDataParams.filterSizeW, loadDataParams.filterSizeH,
                loadDataParams.enTranspose, loadDataParams.fMatrixCtrl, loadDataParams.channelSize);
        }
    }
}

template <>
__aicore__ inline void LoadData3DV2L12L0ACal(__ca__ bfloat16_t* dst, __cbuf__ bfloat16_t* src,
    const LoadData3DParamsV2<bfloat16_t>& loadDataParams)
{
    if ASCEND_IS_AIC {
        img2colv2_cbuf_to_ca(reinterpret_cast<__ca__ half*>(dst),
            reinterpret_cast<__cbuf__ half*>(src),
            loadDataParams.kExtension, loadDataParams.mExtension, loadDataParams.kStartPt,
            loadDataParams.mStartPt, loadDataParams.strideW, loadDataParams.strideH, loadDataParams.filterW,
            loadDataParams.filterH, loadDataParams.dilationFilterW, loadDataParams.dilationFilterH,
            loadDataParams.filterSizeW, loadDataParams.filterSizeH, loadDataParams.enTranspose,
            loadDataParams.fMatrixCtrl, loadDataParams.channelSize);
    }
}

template <>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ bfloat16_t* dst, __cbuf__ bfloat16_t* src,
    const LoadData3DParamsV2<bfloat16_t>& loadDataParams)
{
    if ASCEND_IS_AIC {
        img2colv2_cbuf_to_cb(reinterpret_cast<__cb__ half*>(dst),
            reinterpret_cast<__cbuf__ half*>(src),
            loadDataParams.kExtension, loadDataParams.mExtension, loadDataParams.kStartPt,
            loadDataParams.mStartPt, loadDataParams.strideW, loadDataParams.strideH, loadDataParams.filterW,
            loadDataParams.filterH, loadDataParams.dilationFilterW, loadDataParams.dilationFilterH,
            loadDataParams.filterSizeW, loadDataParams.filterSizeH, loadDataParams.enTranspose,
            loadDataParams.fMatrixCtrl, loadDataParams.channelSize);
    }
}

/* **************************************************************************************************
 * LoadData 3dv2Pro                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadData3DV2L12L0ACal(__ca__ T *dst, __cbuf__ T *src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    if ASCEND_IS_AIC {
        if constexpr (IsSameType<T, int4b_t>::value) {
            img2colv2_cbuf_to_ca_s4((__ca__ void *)dst, (__cbuf__ void *)src, loadDataParams.extConfig,
                loadDataParams.extConfig >> LOAD_M_EXTENSION, loadDataParams.extConfig >> LOAD_K_START_POSITION,
                loadDataParams.extConfig >> LOAD_M_START_POSITION, loadDataParams.filterConfig,
                loadDataParams.filterConfig >> LOAD_STRIDE_H, loadDataParams.filterConfig >> LOAD_FILTER_W,
                loadDataParams.filterConfig >> LOAD_FILTER_H, loadDataParams.filterConfig >> LOAD_DILATION_FILTER_W,
                loadDataParams.filterConfig >> LOAD_DILATION_FILTER_H, loadDataParams.filterSizeW,
                loadDataParams.filterSizeH, loadDataParams.enTranspose, loadDataParams.fMatrixCtrl,
                loadDataParams.channelSize);
        } else {
            img2colv2_cbuf_to_ca(dst, src, loadDataParams.extConfig, loadDataParams.extConfig >> LOAD_M_EXTENSION,
                loadDataParams.extConfig >> LOAD_K_START_POSITION, loadDataParams.extConfig >> LOAD_M_START_POSITION,
                loadDataParams.filterConfig, loadDataParams.filterConfig >> LOAD_STRIDE_H,
                loadDataParams.filterConfig >> LOAD_FILTER_W, loadDataParams.filterConfig >> LOAD_FILTER_H,
                loadDataParams.filterConfig >> LOAD_DILATION_FILTER_W,
                loadDataParams.filterConfig >> LOAD_DILATION_FILTER_H, loadDataParams.filterSizeW,
                loadDataParams.filterSizeH, loadDataParams.enTranspose, loadDataParams.fMatrixCtrl,
                loadDataParams.channelSize);
        }
    }
}

template <>
__aicore__ inline void LoadData3DV2L12L0ACal(__ca__ bfloat16_t* dst, __cbuf__ bfloat16_t* src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    if ASCEND_IS_AIC {
        img2colv2_cbuf_to_ca((__ca__ half*)dst, (__cbuf__ half*)src,
            loadDataParams.extConfig, loadDataParams.extConfig >> LOAD_M_EXTENSION,
            loadDataParams.extConfig >> LOAD_K_START_POSITION, loadDataParams.extConfig >> LOAD_M_START_POSITION,
            loadDataParams.filterConfig, loadDataParams.filterConfig >> LOAD_STRIDE_H,
            loadDataParams.filterConfig >> LOAD_FILTER_W, loadDataParams.filterConfig >> LOAD_FILTER_H,
            loadDataParams.filterConfig >> LOAD_DILATION_FILTER_W,
            loadDataParams.filterConfig >> LOAD_DILATION_FILTER_H, loadDataParams.filterSizeW,
            loadDataParams.filterSizeH, loadDataParams.enTranspose, loadDataParams.fMatrixCtrl,
            loadDataParams.channelSize);
    }
}

template <typename T>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ T *dst, __cbuf__ T *src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    if ASCEND_IS_AIC {
        if constexpr (!IsSameType<T, int4b_t>::value) {
            img2colv2_cbuf_to_cb(dst, src, loadDataParams.extConfig, loadDataParams.extConfig >> LOAD_M_EXTENSION,
                loadDataParams.extConfig >> LOAD_K_START_POSITION, loadDataParams.extConfig >> LOAD_M_START_POSITION,
                loadDataParams.filterConfig, loadDataParams.filterConfig >> LOAD_STRIDE_H,
                loadDataParams.filterConfig >> LOAD_FILTER_W, loadDataParams.filterConfig >> LOAD_FILTER_H,
                loadDataParams.filterConfig >> LOAD_DILATION_FILTER_W,
                loadDataParams.filterConfig >> LOAD_DILATION_FILTER_H, loadDataParams.filterSizeW,
                loadDataParams.filterSizeH, loadDataParams.enTranspose, loadDataParams.fMatrixCtrl,
                loadDataParams.channelSize);
        }
    }
}

template <>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ bfloat16_t* dst, __cbuf__ bfloat16_t* src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    if ASCEND_IS_AIC {
        img2colv2_cbuf_to_cb((__cb__ half*)dst, (__cbuf__ half*)src,
            loadDataParams.extConfig, loadDataParams.extConfig >> LOAD_M_EXTENSION,
            loadDataParams.extConfig >> LOAD_K_START_POSITION, loadDataParams.extConfig >> LOAD_M_START_POSITION,
            loadDataParams.filterConfig, loadDataParams.filterConfig >> LOAD_STRIDE_H,
            loadDataParams.filterConfig >> LOAD_FILTER_W, loadDataParams.filterConfig >> LOAD_FILTER_H,
            loadDataParams.filterConfig >> LOAD_DILATION_FILTER_W,
            loadDataParams.filterConfig >> LOAD_DILATION_FILTER_H, loadDataParams.filterSizeW,
            loadDataParams.filterSizeH, loadDataParams.enTranspose, loadDataParams.fMatrixCtrl,
            loadDataParams.channelSize);
    }
}
/* **************************************************************************************************
 * Mmad                                             *
 * ************************************************************************************************* */
template <typename DstT, typename Src0T, typename Src1T>
__aicore__ inline void MmadCal(__cc__ DstT* c, __ca__ Src0T* a, __cb__ Src1T* b, const MmadParams& mmadParams)
{
    if ASCEND_IS_AIC {
        ASCENDC_ASSERT((SupportType<Tuple<DstT, Src0T, Src1T>, Tuple<int32_t, int8_t, int8_t>,
            Tuple<float, half, half>, Tuple<float, float, float>, Tuple<float, bfloat16_t, bfloat16_t>,
            Tuple<int32_t, int4b_t, int4b_t>>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Mmad, current "
            "api support dtype combination is dst: int32_t, src0: int8_t, src1: int8_t; dst: float, src0: half, "
            "src1: half; dst: float, src0: float, src1: float; dst: float, src0: bfloat16_t, src1: bfloat16_t; "
            "dst: int32_t, src0: int4b_t, src1: int4b_t");});
        bool cmatrixInitVal = mmadParams.cmatrixInitVal && (!mmadParams.isBias);
        if constexpr ((IsSameType<Src0T, int4b_t>::value) && (IsSameType<Src1T, int4b_t>::value)) {
            mad_s4(c, (__ca__ void *)a, (__cb__ void *)b, mmadParams.m, mmadParams.k, mmadParams.n, mmadParams.unitFlag,
                mmadParams.kDirectionAlign, mmadParams.cmatrixSource, cmatrixInitVal);
        } else {
            mad(c, a, b, mmadParams.m, mmadParams.k, mmadParams.n, mmadParams.unitFlag, mmadParams.kDirectionAlign,
                mmadParams.cmatrixSource, cmatrixInitVal);
        }
    }
}

template <typename DstT, typename Src0T, typename Src1T>
__aicore__ inline void MmadCal(__cc__ DstT* c, __ca__ Src0T* a, __cb__ Src1T* b, uint64_t bias,
    const MmadParams& mmadParams, bool cmatrixSource)
{
    if ASCEND_IS_AIC {
        if constexpr ((IsSameType<Src0T, int4b_t>::value) && (IsSameType<Src1T, int4b_t>::value)) {
            mad_s4(c, (__ca__ void *)a, (__cb__ void *)b, mmadParams.m, mmadParams.k, mmadParams.n,
                mmadParams.unitFlag, mmadParams.kDirectionAlign, cmatrixSource,
                mmadParams.cmatrixInitVal);
        } else {
            mad(c, a, b, bias, mmadParams.m, mmadParams.k, mmadParams.n, mmadParams.unitFlag,
                mmadParams.kDirectionAlign, cmatrixSource, mmadParams.cmatrixInitVal);
        }
    }
}

__aicore__ inline void MmadSpCal(__cc__ int32_t *c, __ca__ int8_t *a, __cb__ int8_t *b, const MmadParams &mmadParams)
{
    if ASCEND_IS_AIC {
        mad_sp(c, a, b, mmadParams.m, mmadParams.k, mmadParams.n, mmadParams.unitFlag, mmadParams.cmatrixSource,
            mmadParams.cmatrixInitVal);
    }
}

template <typename T = int8_t, typename U = uint8_t,
    typename std::enable_if<IsSameType<PrimT<T>, int8_t>::value, bool>::type = true,
    typename std::enable_if<IsSameType<PrimT<U>, uint8_t>::value, bool>::type = true>
__aicore__ inline void LoadDataWithSparseCal(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const LocalTensor<U> &idxLocal, const LoadData2dParams &loadDataParam)
{
    if ASCEND_IS_AIC {
#ifdef ASCENDC_CPU_DEBUG
        SetIndexMatrix((void *)idxLocal.GetPhyAddr());
        __cbuf__ int8_t *src = (__cbuf__ int8_t *)srcLocal.GetPhyAddr();
#else
        uint64_t src0tmp = reinterpret_cast<uint64_t>(srcLocal.GetPhyAddr());
        uint64_t src1tmp = reinterpret_cast<uint64_t>(idxLocal.GetPhyAddr());
        // src[31:0], dense weight matrix addr.
        // src[63:32], index matrix addr.
        uint64_t srctmp = (src0tmp & 0xffffffff) | ((src1tmp & 0xffffffff) << 32);
        __cbuf__ int8_t *src = reinterpret_cast<__cbuf__ int8_t *>(srctmp);
#endif
        load_cbuf_to_cb_sp((__cb__ int8_t *)dstLocal.GetPhyAddr(), src, loadDataParam.startIndex,
            loadDataParam.repeatTimes);
    }
}

template <typename T = int8_t, typename std::enable_if<IsSameType<PrimT<T>, int8_t>::value, bool>::type = true> 
__aicore__ inline void LoadUnzipIndexCal(const GlobalTensor<T>& srcTensor, uint32_t numOfIndexTabEntry)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadUnzipIndex");
}

/* **************************************************************************************************
 * LoadData 3dv1                                             *
 * ************************************************************************************************* */
__aicore__ inline void Load3DSetFMatrixCal(uint16_t l1H, uint16_t l1W, const uint8_t padList[4])
{
    if ASCEND_IS_AIC {
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
}

__aicore__ inline void Load3DSetFMatrixBCal(uint16_t l1H, uint16_t l1W, const uint8_t padList[4])
{
    if ASCEND_IS_AIC {
        uint64_t regFMatrix = 0;
        regFMatrix |= (uint64_t)l1W;

        uint32_t l1HShiftBit = 16;
        regFMatrix |= (uint64_t)l1H << l1HShiftBit;

        uint32_t padNumber = 4;
        uint32_t padListShiftBit = 8;
        uint32_t padListShiftBase = 32;
        for (uint32_t i = 0; i < padNumber; i++) {
            regFMatrix |= uint64_t(padList[i] & 0xFF) << (padListShiftBase + i * padListShiftBit);
        }
        set_fmatrix_b(regFMatrix);
    }
}

template <typename T> __aicore__ inline void Load3DSetPaddingCal(const T padValue)
{
    if ASCEND_IS_AIC {
        uint64_t paddingValue = 0;
        uint64_t padValueShiftBit = 8;
        if constexpr (sizeof(T) == B16_BYTE_SIZE || sizeof(T) == B32_BYTE_SIZE) {
            paddingValue = (uint64_t)GetScalarBitcodeValue((T)padValue);
        } else {
            paddingValue = (((uint64_t)padValue) << padValueShiftBit) | ((uint64_t)padValue & 0xFF);
        }
        set_padding(paddingValue);
    }
}

/* **************************************************************************************************
 * LoadData 3dv1                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadData3DV1L12L0ACal(__ca__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV1<T>& loadDataParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData3DParamsV1 from A1 to A2");
}

template <typename T>
__aicore__ inline void LoadData3DV1L12L0BCal(__cb__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV1<T>& loadDataParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData3DParamsV1 from B1 to B2");
}

template <typename T>
__aicore__ inline void LoadData3DV1L12UBCal(__ubuf__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV1<T>& loadDataParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData3DParamsV1 from L1 to UB");
}

/* **************************************************************************************************
 * LoadData 3dv2                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadData3DV2L12UBCal(__ubuf__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV2<T>& loadDataParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData3DParamsV2 from L1 to UB");
}

template <>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ int8_t* dst, __cbuf__ int8_t* src,
    const LoadData3DParamsV2<int8_t>& loadDataParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData3DParamsV2 from B1 to B2 with type int8_t");
}

template <>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ uint8_t* dst, __cbuf__ uint8_t* src,
    const LoadData3DParamsV2<uint8_t>& loadDataParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadData with LoadData3DParamsV2 from B1 to B2 with type uint8_t");
}

/* **************************************************************************************************
 * LoadData 3dv2Pro                                             *
 * ************************************************************************************************* */
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
 * BroadCastVecToMM                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void BroadCastVecToMMCal(__cc__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t blockCount,
    const uint8_t blockLen, const uint8_t srcGap, const uint8_t dstGap)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "BroadCastVecToMM");
}

/* **************************************************************************************************
 * InitL1Buffer                                             *
 * ************************************************************************************************* */
__aicore__ inline void CheckInitConstValueParams(uint16_t repeatTimes, uint16_t blockNum, uint16_t dstGap)
{
    ASCENDC_CHECK_VALUE_RANGE(repeatTimes, 0, UINT15_MAX, "repeatTimes", "InitConstValue");
    ASCENDC_CHECK_VALUE_RANGE(blockNum, 0, UINT15_MAX, "blockNum", "InitConstValue");
    ASCENDC_CHECK_VALUE_RANGE(dstGap, 0, UINT15_MAX, "dstGap", "InitConstValue");
}

template <typename T>
__aicore__ inline void InitL1BufferCal(__cbuf__ T *dst, const InitConstValueParams<T> &initConstValueParams)
{
    if ASCEND_IS_AIC {
        CheckInitConstValueParams(initConstValueParams.repeatTimes, initConstValueParams.blockNum,
            initConstValueParams.dstGap);
        int64_t repeatBit = ((uint64_t)initConstValueParams.blockNum << 16) |
            ((uint64_t)initConstValueParams.dstGap << 32) | initConstValueParams.repeatTimes;
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            create_cbuf_matrix_bf16(dst, repeatBit, initConstValueParams.initValue);
        } else if constexpr (IsSameType<T, uint32_t>::value || IsSameType<T, half>::value) {
            create_cbuf_matrix(dst, repeatBit, (T)initConstValueParams.initValue);
        } else if constexpr (IsSameType<T, int16_t>::value || IsSameType<T, uint16_t>::value) {
            create_cbuf_matrix(dst, repeatBit, GetScalarBitcodeToHalf(initConstValueParams.initValue));
        } else if constexpr (IsSameType<T, float>::value || IsSameType<T, int32_t>::value) {
            create_cbuf_matrix(dst, repeatBit, (uint32_t)GetScalarBitcodeValue(initConstValueParams.initValue));
        } else {
            ASCENDC_ASSERT(false, {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in InitConstValue, current api "
                "support dtype combination is dst: half, int16_t, uint16_t, bfloat16_t, int32_t, uint32_t, float");});
        }
    }
}

/* **************************************************************************************************
 * InitL0ANzMatrix                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void InitL0ANzMatrixCal(__ca__ T *dst, const InitConstValueParams<T> &initConstValueParams)
{
    if ASCEND_IS_AIC {
        CheckInitConstValueParams(initConstValueParams.repeatTimes, initConstValueParams.blockNum,
            initConstValueParams.dstGap);
        int64_t repeatBit = ((uint64_t)initConstValueParams.blockNum << 16) |
            ((uint64_t)initConstValueParams.dstGap << 32) | initConstValueParams.repeatTimes;
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            create_ca_matrix_bf16(dst, repeatBit, initConstValueParams.initValue);
        } else if constexpr (IsSameType<T, uint32_t>::value || IsSameType<T, half>::value) {
            create_ca_matrix(dst, repeatBit, (T)initConstValueParams.initValue);
        } else if constexpr (IsSameType<T, int16_t>::value || IsSameType<T, uint16_t>::value) {
            create_ca_matrix(dst, repeatBit, GetScalarBitcodeToHalf(initConstValueParams.initValue));
        } else if constexpr (IsSameType<T, float>::value || IsSameType<T, int32_t>::value) {
            create_ca_matrix(dst, repeatBit, (uint32_t)GetScalarBitcodeValue(initConstValueParams.initValue));
        } else {
            ASCENDC_ASSERT(false, {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in InitConstValue, current api "
                "support dtype combination is dst: half, int16_t, uint16_t, bfloat16_t, int32_t, uint32_t, float");});
        }
    }
}

/* **************************************************************************************************
 * InitL0BNzMatrix                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void InitL0BNzMatrixCal(__cb__ T *dst, const InitConstValueParams<T> &initConstValueParams)
{
    if ASCEND_IS_AIC {
        CheckInitConstValueParams(initConstValueParams.repeatTimes, initConstValueParams.blockNum,
            initConstValueParams.dstGap);
        int64_t repeatBit = ((uint64_t)initConstValueParams.blockNum << 16) |
            ((uint64_t)initConstValueParams.dstGap << 32) | initConstValueParams.repeatTimes;
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            create_cb_matrix_bf16(dst, repeatBit, initConstValueParams.initValue);
        } else if constexpr (IsSameType<T, uint32_t>::value || IsSameType<T, half>::value) {
            create_cb_matrix(dst, repeatBit, (T)initConstValueParams.initValue);
        } else if constexpr (IsSameType<T, int16_t>::value || IsSameType<T, uint16_t>::value) {
            create_cb_matrix(dst, repeatBit, GetScalarBitcodeToHalf(initConstValueParams.initValue));
        } else if constexpr (IsSameType<T, float>::value || IsSameType<T, int32_t>::value) {
            create_cb_matrix(dst, repeatBit, (uint32_t)GetScalarBitcodeValue(initConstValueParams.initValue));
        } else {
            ASCENDC_ASSERT(false, {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in InitConstValue, current api "
                "support dtype combination is dst: half, int16_t, uint16_t, bfloat16_t, int32_t, uint32_t, float");});
        }
    }
}
/* **************************************************************************************************
 * SetLoadDataRepeat                                             *
 * ************************************************************************************************* */
__aicore__ inline void SetLoadDataRepeatCal(const LoadDataRepeatParam& repeatParams)
{
    if ASCEND_IS_AIC {
        uint64_t rptConfig = (uint64_t)repeatParams.repeatStride | ((uint64_t)repeatParams.repeatTime << 16) |
            ((uint64_t)repeatParams.repeatMode << 24);
        set_l3d_rpt(rptConfig);
    }
}

/* **************************************************************************************************
 * SetLoadDataBoundary                                             *
 * ************************************************************************************************* */
__aicore__ inline void SetLoadDataBoundaryCal(uint32_t boundaryValue)
{
    if ASCEND_IS_AIC {
        set_l1_3d_size(static_cast<uint64_t>(boundaryValue));
    }
}

/* **************************************************************************************************
 * LoadImageToLocalCal                                            *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadImageToLocalCal(__cbuf__ T *dst, const LoadImageToLocalParams &loadDataParams)
{
    if ASCEND_IS_AIC {
        ASCENDC_ASSERT((SupportType<T, int8_t, half>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
            "LoadImageToLocal, current api support dtype combination is dst: int8_t / half.");});
        load_image_to_cbuf(dst, static_cast<uint16_t>(loadDataParams.horizSize - 1),
            static_cast<uint16_t>(loadDataParams.vertSize - 1), loadDataParams.horizStartPos,
            loadDataParams.vertStartPos, static_cast<uint16_t>(loadDataParams.srcHorizSize - 1),
            loadDataParams.topPadSize, loadDataParams.botPadSize, loadDataParams.leftPadSize,
            loadDataParams.rightPadSize, loadDataParams.sid);
    }
}

/* **************************************************************************************************
 * LoadDataUnzip                                            *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadDataUnzipToL1Cal(__cbuf__ T *dst, __gm__ T *src)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadDataUnzip");
}

template <typename T>
__aicore__ inline void LoadDataUnzipToL0BCal(__cb__ T *dst, __gm__ T *src)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadDataUnzip");
}

template <typename T>
__aicore__ inline void LoadDataUnzipToL0ACal(__ca__ T *dst, __gm__ T *src)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LoadDataUnzip");
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_MM_IMPL_H
