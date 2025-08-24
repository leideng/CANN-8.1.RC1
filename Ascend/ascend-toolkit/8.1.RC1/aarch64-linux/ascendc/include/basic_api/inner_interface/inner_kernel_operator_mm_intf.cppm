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
 * \file inner_kernel_operator_mm_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_MM_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_MM_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_check.h"
#include "impl/kernel_operator_mm_base_impl.h"
#include "kernel_struct_mm.h"

namespace AscendC {
/* **************************************************************************************************
 * LoadData 2d                                             *
 * ************************************************************************************************* */
/*
 * @ingroup DataLoad
 * @brief Cube data loading
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] loadDataParams.startIndex Fractal matrix ID
 * @param [in] loadDataParams.repeatTimes repeat times
 * @param [in] loadDataParams.srcStride src block stride
 * @param [in] loadDataParams.sid SMMU SID
 * @param [in] loadDataParams.dstGap interval between the previous tail and the next fractal head
 * @param [in] loadDataParams.ifTranspose enable parameters of transpose function
 */
template <typename T>
__aicore__ inline void LoadData(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LoadData2DParams& loadDataParams)
{
    CheckLoadData2dDatatype<T>();
    LoadDataImpl(dstLocal, srcLocal, loadDataParams);
}

template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void LoadData(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcLocal,
    const LoadData2DParams& loadDataParams)
{
    CheckLoadData2dDatatype<T>();
    LoadDataImpl(dstLocal, srcLocal, loadDataParams);
}

/* **************************************************************************************************
 * LoadData 2dV2                                             *
 * ************************************************************************************************* */
/*
 * @ingroup DataLoad
 * @brief Cube data loading
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor/GlobalTensor
 * @param [in] loadDataParams.mStartPosition m start position
 * @param [in] loadDataParams.kStartPosition k start position
 * @param [in] loadDataParams.srcStride src block stride
 * @param [in] loadDataParams.dstStride dst block stride
 * @param [in] loadDataParams.mStep m step
 * @param [in] loadDataParams.kStep k step
 * @param [in] loadDataParams.sid SMMU SID
 * @param [in] loadDataParams.ifTranspose enable parameters of transpose function
 */
template <typename T>
__aicore__ inline void LoadData(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LoadData2DParamsV2& loadDataParams)
{
    CheckLoadData2dDatatype<T>();
    LoadDataImpl(dstLocal, srcLocal, loadDataParams);
}

template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void LoadData(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcLocal,
    const LoadData2DParamsV2& loadDataParams)
{
    CheckLoadData2dDatatype<T>();
    LoadDataImpl(dstLocal, srcLocal, loadDataParams);
}

/* **************************************************************************************************
 * LoadData 3dv1                                             *
 * ************************************************************************************************* */
/*
 * @ingroup DataLoad
 * @brief Cube data loading
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] loadDataParams.padList padding list
 * @param [in] loadDataParams.l1H operand height
 * @param [in] loadDataParams.l1W operand width
 * @param [in] loadDataParams.c1Inde The starting point of the tensor C1 dimension
 * @param [in] loadDataParams.fetchFilterW The starting position of the w dimension on the convolution kernel
 * @param [in] loadDataParams.fetchFilterH The starting position of the H dimension on the convolution kernel
 * @param [in] loadDataParams.leftTopW Start point of the W dimension on the source operand
 * @param [in] loadDataParams.leftTopH Start point of the H dimension on the source operand
 * @param [in] loadDataParams.strideW W dimension stride
 * @param [in] loadDataParams.strideH H dimension stride
 * @param [in] loadDataParams.filterW Convolution kernel width
 * @param [in] loadDataParams.filterH Convolution kernel height
 * @param [in] loadDataParams.dilationFilterW Convolution kernel width expansion coefficient
 * @param [in] loadDataParams.dilationFilterH Convolution kernel height expansion coefficient
 * @param [in] loadDataParams.jumpStride repeat stride
 * @param [in] loadDataParams.repeatMode repeat mode
 * @param [in] loadDataParams.repeatTime repeat times
 * @param [in] loadDataParams.cSize judge whether to turn on optimization
 * @param [in] loadDataParams.padValue Value of Pad filling value
 */
template <typename T, const IsResetLoad3dConfig &defaultConfig,
    typename U, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type>
__aicore__ inline void LoadData(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LoadData3DParamsV1<U>& loadDataParams)
{
    CheckLoadData3dParams(loadDataParams.l1H, loadDataParams.l1W, loadDataParams.strideW, loadDataParams.strideH);
    ASCENDC_CHECK_VALUE_RANGE(loadDataParams.c1Index, MIN_LOAD3D_C1_IDX, MAX_LOAD3D_C1_IDX, "c1Index",
        "LoadData with LoadData3DParamsV1");
    ASCENDC_CHECK_VALUE_RANGE(loadDataParams.fetchFilterW, MIN_LOAD3D_FETCH_FILTER, MAX_LOAD3D_FETCH_FILTER,
        "fetchFilterW", "LoadData with LoadData3DParamsV1");
    ASCENDC_CHECK_VALUE_RANGE(loadDataParams.fetchFilterH, MIN_LOAD3D_FETCH_FILTER, MAX_LOAD3D_FETCH_FILTER,
        "fetchFilterH", "LoadData with LoadData3DParamsV1");
    ASCENDC_CHECK_VALUE_RANGE(loadDataParams.leftTopW, MIN_LOAD3D_LEFT_TOP, MAX_LOAD3D_LEFT_TOP, "leftTopW",
        "LoadData with LoadData3DParamsV1");
    ASCENDC_CHECK_VALUE_RANGE(loadDataParams.leftTopH, MIN_LOAD3D_LEFT_TOP, MAX_LOAD3D_LEFT_TOP, "leftTopH",
        "LoadData with LoadData3DParamsV1");
    ASCENDC_CHECK_VALUE_RANGE(loadDataParams.jumpStride, MIN_LOAD3D_JUMP_STRIDE, MAX_LOAD3D_JUMP_STRIDE, "jumpStride",
        "LoadData with LoadData3DParamsV1");
    ASCENDC_CHECK_VALUE_RANGE(loadDataParams.repeatMode, 0, 1, "repeatMode", "LoadData with LoadData3DParamsV1");
    ASCENDC_CHECK_VALUE_RANGE(loadDataParams.cSize, 0, 1, "cSize", "LoadData with LoadData3DParamsV1");
    LoadDataImpl<T, defaultConfig>(dstLocal, srcLocal, loadDataParams);
}

/* **************************************************************************************************
 * LoadData 3dv2                                             *
 * enhanced from v1, suitable for aicore > 200                                             *
 * ************************************************************************************************* */
/*
 * @ingroup DataLoad
 * @brief Cube data loading
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] loadDataParams.padList padding list
 * @param [in] loadDataParams.l1H operand height
 * @param [in] loadDataParams.l1W operand width
 * @param [in] loadDataParams.channelSize number of channels
 * @param [in] loadDataParams.kExtension Transmission length of K dimension
 * @param [in] loadDataParams.mExtension Transmission length of M dimension
 * @param [in] loadDataParams.kStartPt Start point of K dimension
 * @param [in] loadDataParams.mStartPt Start point of M dimension
 * @param [in] loadDataParams.strideW W dimension stride
 * @param [in] loadDataParams.strideH H dimension stride
 * @param [in] loadDataParams.filterW Convolution kernel width
 * @param [in] loadDataParams.filterH Convolution kernel height
 * @param [in] loadDataParams.dilationFilterW Convolution kernel width expansion coefficient
 * @param [in] loadDataParams.dilationFilterH Convolution kernel height expansion coefficient
 * @param [in] loadDataParams.enTranspose judge whether to enable the transpose function
 * @param [in] loadDataParams.enSmallK Whether to enable the small k feature
 * @param [in] loadDataParams.padValue Value of Pad filling value
 */
template <typename T, const IsResetLoad3dConfig &defaultConfig,
    typename U, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type>
__aicore__ inline void LoadData(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LoadData3DParamsV2<U>& loadDataParams)
{
#if ASCENDC_CPU_DEBUG
    CheckLoadData3dv2ChannelSize<T>(loadDataParams.channelSize);
    CheckLoadData3dParams(loadDataParams.l1H, loadDataParams.l1W, loadDataParams.strideW, loadDataParams.strideH);
    CheckLoadData3dv2MatrixParams<T>(loadDataParams.kExtension, loadDataParams.mExtension, loadDataParams.kStartPt,
        loadDataParams.mStartPt);
#endif
    LoadDataImpl<T, defaultConfig>(dstLocal, srcLocal, loadDataParams);
}
/* **************************************************************************************************
 * LoadData 3dv2Pro                                             *
 * enhanced from v1, suitable for aicore > 200                                             *
 * ************************************************************************************************* */
/*
 * @ingroup DataLoad
 * @brief Cube data loading
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] loadDataParams.padList padding list
 * @param [in] loadDataParams.l1H operand height
 * @param [in] loadDataParams.l1W operand width
 * @param [in] loadDataParams.channelSize number of channels
 * @param [in] loadDataParams.kExtension Transmission length of K dimension
 * @param [in] loadDataParams.mExtension Transmission length of M dimension
 * @param [in] loadDataParams.kStartPt Start point of K dimension
 * @param [in] loadDataParams.mStartPt Start point of M dimension
 * @param [in] loadDataParams.strideW W dimension stride
 * @param [in] loadDataParams.strideH H dimension stride
 * @param [in] loadDataParams.filterW Convolution kernel width
 * @param [in] loadDataParams.filterH Convolution kernel height
 * @param [in] loadDataParams.dilationFilterW Convolution kernel width expansion coefficient
 * @param [in] loadDataParams.dilationFilterH Convolution kernel height expansion coefficient
 * @param [in] loadDataParams.enTranspose judge whether to enable the transpose function
 * @param [in] loadDataParams.enSmallK Whether to enable the small k feature
 * @param [in] loadDataParams.padValue Value of Pad filling value
 */
template <typename T>
__aicore__ inline void LoadData(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    LoadDataImpl<T>(dstLocal, srcLocal, loadDataParams);
}

/* **************************************************************************************************
 * LoadDataWithTranspose                                             *
 * ************************************************************************************************* */
/*
 * @ingroup DataLoad
 * @brief Cube data loading
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] loadDataParams.startIndex index of the first fractal in the first repeat in the source matrix
 * in unit of frac num
 * @param [in] loadDataParams.repeatTimes the repeat times
 * @param [in] loadDataParams.srcStride source stride between consequent repeat times in unit of frac num
 * @param [in] loadDataParams.dstGap destination gap between consequent repeat times in unit of 512byte
 * @param [in] loadDataParams.dstFracGap dst fractal gap in unit of one 512byte fractal
 */
template <typename T>
__aicore__ inline void LoadDataWithTranspose(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LoadData2dTransposeParams& loadDataParams)
{
    LoadDataWithTransposeImpl(dstLocal, srcLocal, loadDataParams);
}

/*
 * @ingroup DataLoad
 * @brief Cube data loading
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] loadDataParams.startIndex index of the first fractal in the first repeat in the source matrix
 * in unit of 512byte fractal
 * @param [in] loadDataParams.repeatTimes the repeat times
 * @param [in] loadDataParams.srcStride source stride between consequent repeat times in unit of 512byte
 * @param [in] loadDataParams.dstGap destination gap between consequent repeat times in unit of 512byte
 * @param [in] loadDataParams.dstFracGap dst fractal gap in unit of one 512byte fractal
 * @param [in] loadDataParams.srcFracGap dst fractal gap in unit of one 512byte fractal
 */
template <typename T>
__aicore__ inline void LoadDataWithTranspose(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LoadData2dTransposeParamsV2& loadDataParams)
{
    LoadDataWithTransposeImpl(dstLocal, srcLocal, loadDataParams);
}

/* **************************************************************************************************
 * Mmad                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Mmad
 * @brief Matrix multiplication and addition
 * @param [out] dstLocal output LocalTensor
 * @param [in] fmLocal input LocalTensor
 * @param [in] filterLocal input LocalTensor
 * @param [in] mmadParams.m Left matrix row number
 * @param [in] mmadParams.n right matrix column number
 * @param [in] mmadParams.k Left matrix column number m
 * @param [in] mmadParams.unitFlag whether enable unit flag
 * @param [in] mmadParams.kDirectionAlign is the indicator for alignment in L0A/L0B in the K direction
 * @param [in] mmadParams.cmatrixSource indicates the C matrix source, 1: the C matrix is in bias table buffer, 0: the C
 * matrix is in L0C
 * @param [in] mmadParams.cmatrixInitVal indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real
 * number in C matrix
 */

template <typename DstT, typename Src0T, typename Src1T>
__aicore__ inline void Mmad(const LocalTensor<DstT>& dstLocal, const LocalTensor<Src0T>& fmLocal,
    const LocalTensor<Src1T>& filterLocal, const MmadParams& mmadParams)
{
    MmadImpl(dstLocal, fmLocal, filterLocal, mmadParams);
}

template <typename DstT, typename Src0T, typename Src1T, typename BiasT>
__aicore__ inline void Mmad(const LocalTensor<DstT>& dstLocal, const LocalTensor<Src0T>& fmLocal,
    const LocalTensor<Src1T>& filterLocal, const LocalTensor<BiasT>& biasLocal, const MmadParams& mmadParams)
{
    MmadImpl(dstLocal, fmLocal, filterLocal, biasLocal, mmadParams);
}

#if __CCE_AICORE__ == 220
template <typename T, typename U,
    typename std::enable_if<IsSameType<PrimT<T>, int32_t>::value, bool>::type,
    typename std::enable_if<IsSameType<PrimT<U>, int8_t>::value, bool>::type>
__aicore__ inline void MmadWithSparse(const LocalTensor<T>& dstLocal, const LocalTensor<U>& fmLocal,
    const LocalTensor<U>& filterLocal, const MmadParams& mmadParams)
{
    MmadSpImpl(dstLocal, fmLocal, filterLocal, mmadParams);
}

template <typename T, typename U,
    typename std::enable_if<IsSameType<PrimT<T>, int8_t>::value, bool>::type,
    typename std::enable_if<IsSameType<PrimT<U>, uint8_t>::value, bool>::type>
__aicore__ inline void LoadDataWithSparse(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const LocalTensor<U> &idxLocal, const LoadData2dParams &loadDataParam)
{
    LoadDataWithSparseImpl(dstLocal, srcLocal, idxLocal, loadDataParam);
}
#endif

#if __CCE_AICORE__ == 200
template <typename T, typename std::enable_if<IsSameType<PrimT<T>, int8_t>::value, bool>::type> 
__aicore__ inline void LoadUnzipIndex(const GlobalTensor<T>& srcTensor, uint32_t numOfIndexTabEntry)
{
    LoadUnzipIndexImpl(srcTensor, numOfIndexTabEntry);
}
#endif

/* **************************************************************************************************
 * BroadCastVecToMM                                             *
 * ************************************************************************************************* */
template <typename T, typename U>
__aicore__ inline __inout_pipe__(V) void BroadCastVecToMM(const LocalTensor<T> &dstLocal,
    const LocalTensor<U> &srcLocal, const int32_t blockCount, const uint8_t blockLen, const uint8_t srcGap,
    const uint8_t dstGap)
{
    BroadCastVecToMMImpl(dstLocal, srcLocal, blockCount, blockLen, srcGap, dstGap);
}

/* **************************************************************************************************
 * InitConstValue                                             *
 * ************************************************************************************************* */
/*
 * @ingroup InitConstValue
 * @brief L0A/L0B/L1 value initializing
 * @param [out] dstLocal output LocalTensor
 * @param [in] initConstValueParams.repeatTimes repeat times
 * @param [in] initConstValueParams.repeatTimes blockNum block number
 * @param [in] initConstValueParams.dstGap interval between the previous tail and the next block head
 * @param [in] initConstValueParams.initValue initialize Value
 */
template <typename T, typename U, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type>
__aicore__ inline void InitConstValue(const LocalTensor<T> &dstLocal,
    const InitConstValueParams<U>& initConstValueParams)
{
#if ASCENDC_CPU_DEBUG
    uint16_t repeatTimes = initConstValueParams.repeatTimes;
#if __CCE_AICORE__ == 220
    uint16_t blockNum = initConstValueParams.blockNum;
    uint16_t dstGap = initConstValueParams.dstGap;
#else
    uint16_t blockNum = 1;
    uint16_t dstGap = 0;
#endif
    if (!CheckFuncInitConstValue(dstLocal, repeatTimes, blockNum, dstGap, "InitConstValue")) {
        ASCENDC_REPORT_CHECK_ERROR("InitConstValue", KernelFuncType::NONE_MODE);
    }
#endif
    InitConstValueImpl(dstLocal, initConstValueParams);
}
/* **************************************************************************************************
 * SetLoadDataPaddingValue                                             *
 * ************************************************************************************************* */
/*
 * @ingroup SetLoadDataPaddingValue
 * @brief setting loadData pad value
 * @param [in]padValue padding value
 */
template <typename T>
__aicore__ inline void SetLoadDataPaddingValue(const T padValue)
{
    Load3DSetPaddingImpl(padValue);
}

/* **************************************************************************************************
 * SetFmatrix                                             *
 * ************************************************************************************************* */
/*
 * @ingroup SetFmatrix
 * @brief setting fmatrix
 * @param [in]l1H operand height
 * @param [in]l1W operand width
 * @param [in]padList padding list
 */
__aicore__ inline void SetFmatrix(uint16_t l1H, uint16_t l1W, const uint8_t padList[4], const FmatrixMode& fmatrixMode)
{
    ASCENDC_CHECK_VALUE_RANGE(l1H, MIN_LOAD3D_L1, MAX_LOAD3D_L1, "l1H", "SetFmatrix");
    ASCENDC_CHECK_VALUE_RANGE(l1W, MIN_LOAD3D_L1, MAX_LOAD3D_L1, "l1W", "SetFmatrix");
    SetFmatrixImpl(l1H, l1W, padList, fmatrixMode);
}

/* **************************************************************************************************
 * SetLoadDataBoundary                                             *
 * ************************************************************************************************* */
/*
 * @ingroup SetLoadDataBoundary
 * @brief setting loaddata boundary
 * @param [in]boundaryValue
 */
__aicore__ inline void SetLoadDataBoundary(uint32_t boundaryValue)
{
    SetLoadDataBoundaryImpl(boundaryValue);
}

/* **************************************************************************************************
 * SetLoadDataRepeat                                             *
 * ************************************************************************************************* */
__aicore__ inline void SetLoadDataRepeat(const LoadDataRepeatParam& repeatParams)
{
    ASCENDC_CHECK_VALUE_RANGE(repeatParams.repeatMode, 0, 1, "repeatMode", "SetLoadDataRepeat");
    SetLoadDataRepeatImpl(repeatParams);
}

/* **************************************************************************************************
 * LoadImageToLocal                                             *
 * ************************************************************************************************* */
/*
 * @ingroup LoadImageToLocal
 * @brief loadData image from gm to L1
 * @param [out] dstLocal output LocalTensor
 * @param [in] loadImageToLocalParams.horizSize operand height
 * @param [in] loadImageToLocalParams.vertSize operand width
 * @param [in] loadImageToLocalParams.horizStartPos horizontal start position
 * @param [in] loadImageToLocalParams.vertStartPos vertical start position
 * @param [in] loadImageToLocalParams.srcHorizSize src horizontal size
 * @param [in] loadImageToLocalParams.topPadSize top padding size
 * @param [in] loadImageToLocalParams.botPadSize bottom padding size
 * @param [in] loadImageToLocalParams.leftPadSize left hblank/padding size
 * @param [in] loadImageToLocalParams.rightPadSize right hblank/padding size
 */
template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void LoadImageToLocal(const LocalTensor<T>& dstLocal,
    const LoadImageToLocalParams& loadDataParams)
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    ASCENDC_ASSERT(CheckFuncLoadImageToLocal(dstLocal, loadDataParams, "LoadImageToLocal"), {
        ASCENDC_REPORT_CHECK_ERROR("LoadImageToLocal", KernelFuncType::NONE_MODE);
    });
#endif
    ASCENDC_CHECK_VALUE_RANGE(loadDataParams.horizSize, 2, UINT12_MAX, "horizSize", "LoadImageToLocal");
    ASCENDC_CHECK_VALUE_RANGE(loadDataParams.vertSize, 2, UINT12_MAX, "vertSize", "LoadImageToLocal");
    ASCENDC_CHECK_VALUE_RANGE(loadDataParams.horizStartPos, 0, UINT12_MAX, "horizStartPos", "LoadImageToLocal");
    ASCENDC_CHECK_VALUE_RANGE(loadDataParams.vertStartPos, 0, UINT12_MAX, "vertStartPos", "LoadImageToLocal");
    ASCENDC_CHECK_VALUE_RANGE(loadDataParams.srcHorizSize, 2, UINT12_MAX, "srcHorizSize", "LoadImageToLocal");
    ASCENDC_CHECK_VALUE_RANGE(loadDataParams.topPadSize, 0, 32, "topPadSize", "LoadImageToLocal");
    ASCENDC_CHECK_VALUE_RANGE(loadDataParams.botPadSize, 0, 32, "botPadSize", "LoadImageToLocal");
    ASCENDC_CHECK_VALUE_RANGE(loadDataParams.leftPadSize, 0, 32, "leftPadSize", "LoadImageToLocal");
    ASCENDC_CHECK_VALUE_RANGE(loadDataParams.rightPadSize, 0, 32, "rightPadSize", "LoadImageToLocal");
    const Hardware dstScope = GetPhyType((TPosition)dstLocal.GetPosition());
    if (dstScope == Hardware::L1) {
        LoadImageToLocalCal((__cbuf__ PrimT<T>*)dstLocal.GetPhyAddr(), loadDataParams);
    } else {
        ASCENDC_CHECK_TPOSITION(false, "dstLocal", "A1 / B1", "LoadImageToLocal",
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
    }
}

/* **************************************************************************************************
 * LoadDataUnzip                                             *
 * ************************************************************************************************* */
/*
 * @ingroup LoadDataUnzip
 * @brief loadData and unzip
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input GlobalTensor
 */
template <typename T>
__aicore__ inline void LoadDataUnzip(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal)
{
    LoadDataUnzipImpl(dstLocal, srcGlobal);
}
} // namespace AscendC
#endif // ASCENDC_MODULE_INNER_OPERATOR_MM_INTERFACE_H
