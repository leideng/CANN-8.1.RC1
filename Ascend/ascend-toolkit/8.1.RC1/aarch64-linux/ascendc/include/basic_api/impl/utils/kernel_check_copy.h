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
 * \file kernel_check_copy.h
 * \brief
 */
#ifndef ASCENDC_MODULE_CHECK_COPY_H
#define ASCENDC_MODULE_CHECK_COPY_H

#if ASCENDC_CPU_DEBUG
#include "tikcpp_check_util.h"
#include "kernel_common.h"
#include "kernel_struct_data_copy.h"

namespace AscendC {
/* **************************************************************************************************
 * Check function for CPU debug
 * ************************************************************************************************* */
template <typename T>
bool CheckFuncCopy(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const uint64_t mask,
    const uint8_t repeatTimes, const CopyRepeatParams& repeatParams, const char* intriName)
{
    check::CopyApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstStride),
        static_cast<uint16_t>(repeatParams.srcStride),
        static_cast<uint16_t>(repeatParams.dstRepeatSize),
        static_cast<uint16_t>(repeatParams.srcRepeatSize),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()) };
    return CheckFuncCopyImpl(chkParams, mask, intriName);
}

template <typename T>
bool CheckFuncCopy(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const uint64_t mask[],
    const uint8_t repeatTimes, const CopyRepeatParams& repeatParams, const char* intriName)
{
    check::CopyApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstStride),
        static_cast<uint16_t>(repeatParams.srcStride),
        static_cast<uint16_t>(repeatParams.dstRepeatSize),
        static_cast<uint16_t>(repeatParams.srcRepeatSize),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()) };
    return CheckFuncCopyImplForMaskArray(chkParams, mask, intriName);
}

template <typename T>
bool CheckFuncDataCopy(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const DataCopyParams &repeatParams, const char *intriName)
{
    check::DataCopyApiParams chkParams{
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()),
        repeatParams.blockCount,
        repeatParams.blockLen,
        repeatParams.srcStride,
        repeatParams.dstStride};
    return CheckFuncDataCopyImpl(chkParams, intriName);
}

template <typename T>
bool CheckFuncDataCopy(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &srcGlobal,
    const DataCopyParams &repeatParams, const char *intriName)
{
    check::DataCopyApiParams chkParams{
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcGlobal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(Hardware::GM),
        repeatParams.blockCount,
        repeatParams.blockLen,
        repeatParams.srcStride,
        repeatParams.dstStride};
    return CheckFuncDataCopyImpl(chkParams, intriName);
}

template <typename T>
bool CheckFuncDataCopyPad(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &srcGlobal,
    const DataCopyParams &dataCopyParams, const DataCopyPadParams &padParams, const char *intriName)
{
    check::DataCopyPadApiParams chkParams{
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcGlobal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(Hardware::GM),
        dataCopyParams.blockCount,
        dataCopyParams.blockLen,
        dataCopyParams.srcStride,
        dataCopyParams.dstStride,
        padParams.isPad,
        padParams.leftPadding,
        padParams.rightPadding,
        padParams.paddingValue};
    return CheckFuncDataCopyPadImpl(chkParams, intriName);
}

template <typename T>
bool CheckFuncDataCopyPad(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &srcGlobal,
    const DataCopyExtParams &dataCopyParams, const DataCopyPadExtParams<T> &padParams, const char *intriName)
{
    check::DataCopyPadApiParams chkParams{
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcGlobal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(Hardware::GM),
        dataCopyParams.blockCount,
        dataCopyParams.blockLen,
        dataCopyParams.srcStride,
        dataCopyParams.dstStride,
        padParams.isPad,
        padParams.leftPadding,
        padParams.rightPadding,
        GetScalarBitcodeValue(padParams.paddingValue)};
    return CheckFuncDataCopyPadImpl(chkParams, intriName);
}

template <typename T>
bool CheckFuncDataCopy(const GlobalTensor<T> &dstGlobal, const LocalTensor<T> &srcLocal,
    const DataCopyParams &repeatParams, const char *intriName)
{
    check::DataCopyApiParams chkParams{
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstGlobal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint8_t>(Hardware::GM),
        static_cast<uint8_t>(srcLocal.GetPosition()),
        repeatParams.blockCount,
        repeatParams.blockLen,
        repeatParams.srcStride,
        repeatParams.dstStride};
    return CheckFuncDataCopyImpl(chkParams, intriName);
}

template <typename T>
bool CheckFuncDataCopySlice(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
    const SliceInfo dstSliceInfo[], const SliceInfo srcSliceInfo[], const uint32_t dimValue, const char* intriName)
{
    bool isGM2UB = true;
    uint32_t srcShapeInfo[K_MAX_SHAPE_DIM];
    uint32_t dstShapeInfo[K_MAX_SHAPE_DIM];
    bool useShapeValue = !(srcSliceInfo[0].shapeValue == 0);
    for (int i = 0; i < dimValue; i++) {
        srcShapeInfo[i] = useShapeValue ? srcSliceInfo[i].shapeValue : srcGlobal.GetShapeInfo().shape[i];
        dstShapeInfo[i] = useShapeValue ? dstSliceInfo[i].shapeValue : dstLocal.GetShapeInfo().shape[i];
    }

    check::DataCopySliceApiParams chkParams{ static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcGlobal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        dimValue,
        dstShapeInfo,
        srcShapeInfo,
        dstSliceInfo,
        srcSliceInfo,
        isGM2UB };
    return CheckFuncDataCopySliceImpl(chkParams, intriName);
}

template <typename T>
bool CheckFuncDataCopySlice(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
    const SliceInfo dstSliceInfo[], const SliceInfo srcSliceInfo[], const uint32_t dimValue, const char* intriName)
{
    bool isGM2UB = false;
    uint32_t srcShapeInfo[K_MAX_SHAPE_DIM];
    uint32_t dstShapeInfo[K_MAX_SHAPE_DIM];
    bool useShapeValue = !(srcSliceInfo[0].shapeValue == 0);
    for (int i = 0; i < dimValue; i++) {
        srcShapeInfo[i] = useShapeValue ? srcSliceInfo[i].shapeValue : srcLocal.GetShapeInfo().shape[i];
        dstShapeInfo[i] = useShapeValue ? dstSliceInfo[i].shapeValue : dstGlobal.GetShapeInfo().shape[i];
    }

    check::DataCopySliceApiParams chkParams{ static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstGlobal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(srcLocal.GetPosition()),
        dimValue,
        dstShapeInfo,
        srcShapeInfo,
        dstSliceInfo,
        srcSliceInfo,
        isGM2UB };
    return CheckFuncDataCopySliceImpl(chkParams, intriName);
}

template <typename T>
bool CheckDataCopyTensorSizeOverflow(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
    const DataCopyParams& repeatParams) {
    const Hardware srcPos = Hardware::GM;
    const Hardware dstPos = GetPhyType((TPosition)dstLocal.GetPosition());
    uint64_t srcMaxOffsetBytes = 0;
    uint64_t dstMaxOffsetBytes = 0;
    BlockMode mode = check::GetBlockMode({ srcPos, dstPos });
    uint8_t biasConvFlag = 0;
    std::string apiInfo = "DataCopy from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(TPosition::GM)) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition()));

    check::CalculateDataCopyMaxOffset<PrimT<T>, PrimT<T>>(
        repeatParams, srcPos, dstPos, mode, srcMaxOffsetBytes, dstMaxOffsetBytes, DeqScale::DEQ_NONE, biasConvFlag);
    return check::ReportTensorSizeOverflow(srcPos, dstPos, 0, dstLocal.GetSize() * sizeof(PrimT<T>),
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T>
bool CheckDataCopyTensorSizeOverflow(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
    const Nd2NzParams& intriParams) {
    if (intriParams.ndNum == 0 || intriParams.nValue == 0 || intriParams.dValue == 0) {
        return true;
    }
    const Hardware srcPos = Hardware::GM;
    const Hardware dstPos = GetPhyType((TPosition)dstLocal.GetPosition());
    uint64_t srcMaxOffsetBytes = 0;
    uint64_t dstMaxOffsetBytes = (intriParams.ndNum - 1) * intriParams.dstNzMatrixStride * sizeof(PrimT<T>) +
        (intriParams.nValue - 1) * intriParams.dstNzNStride * DEFAULT_C0_SIZE +
        (DivCeil(intriParams.dValue * sizeof(PrimT<T>), DEFAULT_C0_SIZE) - 1) *
        intriParams.dstNzC0Stride * DEFAULT_C0_SIZE +
        DEFAULT_C0_SIZE;
    std::string apiInfo = "DataCopy with Nd2NzParams from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(TPosition::GM)) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition()));

    return check::ReportTensorSizeOverflow(srcPos, dstPos, 0, dstLocal.GetSize() * sizeof(PrimT<T>),
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T>
bool CheckDataCopyTensorSizeOverflow(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const Nd2NzParams& intriParams) {
    if (intriParams.ndNum == 0 || intriParams.nValue == 0 || intriParams.dValue == 0) {
        return true;
    }
    const Hardware srcPos = GetPhyType((TPosition)srcLocal.GetPosition());
    const Hardware dstPos = GetPhyType((TPosition)dstLocal.GetPosition());
    uint64_t srcMaxOffsetBytes = ((intriParams.ndNum - 1) * intriParams.srcNdMatrixStride +
        (intriParams.nValue - 1) * intriParams.srcDValue + intriParams.dValue) * sizeof(PrimT<T>);
    uint64_t dstMaxOffsetBytes = (intriParams.ndNum - 1) * intriParams.dstNzMatrixStride * sizeof(PrimT<T>) +
        (intriParams.nValue - 1) * intriParams.dstNzNStride * DEFAULT_C0_SIZE +
        (DivCeil(intriParams.dValue * sizeof(PrimT<T>), DEFAULT_C0_SIZE) - 1) *
        intriParams.dstNzC0Stride * DEFAULT_C0_SIZE +
        DEFAULT_C0_SIZE;
    std::string apiInfo = "DataCopy with Nd2NzParams from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition()));

    return check::ReportTensorSizeOverflow(srcPos, dstPos, srcLocal.GetSize() * sizeof(PrimT<T>),
        dstLocal.GetSize() * sizeof(PrimT<T>),
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T>
bool CheckDataCopyTensorSizeOverflow(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
    const DataCopyParams& repeatParams) {
    const Hardware srcPos = GetPhyType((TPosition)srcLocal.GetPosition());
    const Hardware dstPos = Hardware::GM;
    uint64_t srcMaxOffsetBytes = 0;
    uint64_t dstMaxOffsetBytes = 0;
    BlockMode mode = check::GetBlockMode({ srcPos, dstPos });
    uint8_t biasConvFlag = 0;
    std::string apiInfo = "DataCopy from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(TPosition::GM));

    check::CalculateDataCopyMaxOffset<PrimT<T>, PrimT<T>>(
        repeatParams, srcPos, dstPos, mode, srcMaxOffsetBytes, dstMaxOffsetBytes, DeqScale::DEQ_NONE, biasConvFlag);
    return check::ReportTensorSizeOverflow(srcPos, dstPos, srcLocal.GetSize() * sizeof(PrimT<T>), 0,
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T>
bool CheckDataCopyTensorSizeOverflow(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const DataCopyParams &repeatParams) {
    const Hardware srcPos = GetPhyType((TPosition)srcLocal.GetPosition());
    const Hardware dstPos = GetPhyType((TPosition)dstLocal.GetPosition());
    uint64_t srcMaxOffsetBytes = 0;
    uint64_t dstMaxOffsetBytes = 0;
    BlockMode mode = check::GetBlockMode({ srcPos, dstPos });
    uint8_t biasConvFlag = 0;
    std::string apiInfo = "DataCopy from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition()));

    check::CalculateDataCopyMaxOffset<PrimT<T>, PrimT<T>>(
        repeatParams, srcPos, dstPos, mode, srcMaxOffsetBytes, dstMaxOffsetBytes, DeqScale::DEQ_NONE, biasConvFlag);
    return check::ReportTensorSizeOverflow(srcPos, dstPos,
        srcLocal.GetSize() * sizeof(PrimT<T>), dstLocal.GetSize() * sizeof(PrimT<T>),
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename DstT, typename SrcT>
bool CheckDataCopyTensorSizeOverflow(const LocalTensor<DstT> &dstLocal, const LocalTensor<SrcT> &srcLocal,
    const DataCopyParams &repeatParams) {
    const Hardware srcPos = GetPhyType((TPosition)srcLocal.GetPosition());
    const Hardware dstPos = GetPhyType((TPosition)dstLocal.GetPosition());
    uint64_t srcMaxOffsetBytes = 0;
    uint64_t dstMaxOffsetBytes = 0;
    BlockMode mode = check::GetBlockMode({ srcPos, dstPos });
    uint8_t biasConvFlag = check::IsBiasConv({ srcPos, dstPos }) && (sizeof(PrimT<SrcT>) != sizeof(PrimT<DstT>));
    std::string apiInfo = "DataCopy from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition()));

    check::CalculateDataCopyMaxOffset<PrimT<SrcT>, PrimT<DstT>>(
        repeatParams, srcPos, dstPos, mode, srcMaxOffsetBytes, dstMaxOffsetBytes, DeqScale::DEQ_NONE, biasConvFlag);
    return check::ReportTensorSizeOverflow(srcPos, dstPos,
        srcLocal.GetSize() * sizeof(PrimT<SrcT>), dstLocal.GetSize() * sizeof(PrimT<DstT>),
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T>
bool CheckDataCopyTensorSizeOverflow(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &srcGlobal,
    const SliceInfo dstSliceInfo[], const SliceInfo srcSliceInfo[], const uint32_t dimValue)
{
    const Hardware srcPos = Hardware::GM;
    const Hardware dstPos = GetPhyType((TPosition)dstLocal.GetPosition());
    uint64_t srcMaxOffsetBytes = 0;
    uint64_t dstMaxOffsetBytes = 1;
    std::string apiInfo = "DataCopy with SliceInfo from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(TPosition::GM)) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition()));

    for (uint32_t i = 0; i < dimValue; i++) {
        dstMaxOffsetBytes *= dstSliceInfo[i].shapeValue;
    }
    dstMaxOffsetBytes *= sizeof(PrimT<T>);
    return check::ReportTensorSizeOverflow(srcPos, dstPos, 0, dstLocal.GetSize() * sizeof(PrimT<T>),
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T>
bool CheckDataCopyTensorSizeOverflow(const GlobalTensor<T> &dstGlobal, const LocalTensor<T> &srcLocal,
    const SliceInfo dstSliceInfo[], const SliceInfo srcSliceInfo[], const uint32_t dimValue)
{
    const Hardware srcPos = GetPhyType((TPosition)srcLocal.GetPosition());
    const Hardware dstPos = Hardware::GM;
    uint64_t srcMaxOffsetBytes = 1;
    uint64_t dstMaxOffsetBytes = 0;
    std::string apiInfo = "DataCopy with SliceInfo from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(TPosition::GM));

    for (uint32_t i = 0; i < dimValue; i++) {
        srcMaxOffsetBytes *= dstSliceInfo[i].shapeValue;
    }
    srcMaxOffsetBytes *= sizeof(PrimT<T>);
    return check::ReportTensorSizeOverflow(srcPos, dstPos, srcLocal.GetSize() * sizeof(PrimT<T>), 0,
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T>
bool CheckDataCopyTensorSizeOverflow(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
    const uint32_t calCount)
{
    const Hardware srcPos = Hardware::GM;
    const Hardware dstPos = GetPhyType((TPosition)dstLocal.GetPosition());
    uint64_t srcMaxOffsetBytes = 0;
    uint64_t dstMaxOffsetBytes = calCount * sizeof(PrimT<T>);
    std::string apiInfo = "DataCopy from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(TPosition::GM)) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition()));

    return check::ReportTensorSizeOverflow(srcPos, dstPos, 0, dstLocal.GetSize() * sizeof(PrimT<T>),
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T>
bool CheckDataCopyTensorSizeOverflow(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint32_t calCount)
{
    const Hardware srcPos = GetPhyType((TPosition)srcLocal.GetPosition());
    const Hardware dstPos = GetPhyType((TPosition)dstLocal.GetPosition());
    uint64_t srcMaxOffsetBytes = calCount * sizeof(PrimT<T>);
    uint64_t dstMaxOffsetBytes = calCount * sizeof(PrimT<T>);
    std::string apiInfo = "DataCopy from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition()));

    return check::ReportTensorSizeOverflow(srcPos, dstPos,
        srcLocal.GetSize() * sizeof(PrimT<T>), dstLocal.GetSize() * sizeof(PrimT<T>),
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T>
bool CheckDataCopyTensorSizeOverflow(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
    const uint32_t calCount)
{
    const Hardware srcPos = GetPhyType((TPosition)srcLocal.GetPosition());
    const Hardware dstPos = Hardware::GM;
    uint64_t srcMaxOffsetBytes = calCount * sizeof(PrimT<T>);
    uint64_t dstMaxOffsetBytes = 0;
    std::string apiInfo = "DataCopy from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(TPosition::GM));

    return check::ReportTensorSizeOverflow(srcPos, dstPos,
        srcLocal.GetSize() * sizeof(PrimT<T>), 0,
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T>
bool CheckDataCopyTensorSizeOverflow(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
    const Nz2NdParamsFull& intriParams) {
    if (intriParams.ndNum == 0 || intriParams.nValue == 0 || intriParams.dValue == 0) {
        return true;
    }
    const Hardware srcPos = GetPhyType((TPosition)srcLocal.GetPosition());
    const Hardware dstPos = Hardware::GM;
    const int nzWidth = 16;
    const int srcMatrixStrideUnit = 256;
    uint64_t srcMaxOffsetBytes = (intriParams.ndNum - 1) * intriParams.srcNdMatrixStride * srcMatrixStrideUnit *
        sizeof(PrimT<T>) + (intriParams.dValue / nzWidth - 1) * intriParams.srcNStride * nzWidth * sizeof(PrimT<T>) +
        nzWidth * intriParams.nValue * sizeof(PrimT<T>);
    uint64_t dstMaxOffsetBytes = 0;
    std::string apiInfo = "DataCopy with Nz2NdParamsFull from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(TPosition::GM));

    return check::ReportTensorSizeOverflow(srcPos, dstPos,
        srcLocal.GetSize() * sizeof(PrimT<T>), 0,
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T>
bool CheckDataCopyTensorSizeOverflow(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
    const DataCopyParams& intriParams, const DataCopyEnhancedParams& enhancedParams) {
    if (intriParams.blockCount == 0 || intriParams.blockLen == 0) {
        return true;
    }
    const Hardware srcPos = Hardware::GM;
    const Hardware dstPos = GetPhyType((TPosition)dstLocal.GetPosition());
    uint64_t srcMaxOffsetBytes = 0;
    uint64_t dstMaxOffsetBytes = 0;
    BlockMode mode = check::GetBlockMode({ srcPos, dstPos }, enhancedParams.blockMode);
    uint8_t biasConvFlag = 0;
    std::string apiInfo = "DataCopy with DataCopyEnhancedParams from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(TPosition::GM)) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition()));

    check::CalculateDataCopyMaxOffset<PrimT<T>, PrimT<T>>(
        intriParams, srcPos, dstPos, mode, srcMaxOffsetBytes, dstMaxOffsetBytes,
        enhancedParams.deqScale, biasConvFlag);
    return check::ReportTensorSizeOverflow(srcPos, dstPos, 0, dstLocal.GetSize() * sizeof(PrimT<T>),
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T>
bool CheckDataCopyTensorSizeOverflow(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
    const DataCopyParams& intriParams, const DataCopyEnhancedParams& enhancedParams) {
    if (intriParams.blockCount == 0 || intriParams.blockLen == 0) {
        return true;
    }
    const Hardware srcPos = GetPhyType((TPosition)srcLocal.GetPosition());
    const Hardware dstPos = Hardware::GM;
    uint64_t srcMaxOffsetBytes = 0;
    uint64_t dstMaxOffsetBytes = 0;
    BlockMode mode = check::GetBlockMode({ srcPos, dstPos }, enhancedParams.blockMode);
    uint8_t biasConvFlag = 0;
    std::string apiInfo = "DataCopy with DataCopyEnhancedParams from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(TPosition::GM));

    check::CalculateDataCopyMaxOffset<PrimT<T>, PrimT<T>>(
        intriParams, srcPos, dstPos, mode, srcMaxOffsetBytes, dstMaxOffsetBytes,
        enhancedParams.deqScale, biasConvFlag);
    return check::ReportTensorSizeOverflow(srcPos, dstPos, srcLocal.GetSize() * sizeof(PrimT<T>), 0,
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T>
bool CheckDataCopyTensorSizeOverflow(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const DataCopyParams &intriParams, const DataCopyEnhancedParams &enhancedParams) {
    if (intriParams.blockCount == 0 || intriParams.blockLen == 0) {
        return true;
    }
    const Hardware srcPos = GetPhyType((TPosition)srcLocal.GetPosition());
    const Hardware dstPos = GetPhyType((TPosition)dstLocal.GetPosition());
    uint64_t srcMaxOffsetBytes = 0;
    uint64_t dstMaxOffsetBytes = 0;
    BlockMode mode = check::GetBlockMode({ srcPos, dstPos }, enhancedParams.blockMode);
    uint8_t biasConvFlag = 0;
    std::string apiInfo = "DataCopy with DataCopyEnhancedParams from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition()));

    check::CalculateDataCopyMaxOffset<PrimT<T>, PrimT<T>>(
        intriParams, srcPos, dstPos, mode, srcMaxOffsetBytes, dstMaxOffsetBytes,
        enhancedParams.deqScale, biasConvFlag);
    return check::ReportTensorSizeOverflow(srcPos, dstPos,
        srcLocal.GetSize() * sizeof(PrimT<T>), dstLocal.GetSize() * sizeof(PrimT<T>),
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T, typename U>
bool CheckDataCopyTensorSizeOverflow(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal,
    const DataCopyCO12DstParams& intriParams) {
    if (intriParams.nSize == 0) {
        return true;
    }
    const Hardware srcPos = GetPhyType((TPosition)srcLocal.GetPosition());
    const Hardware dstPos = GetPhyType((TPosition)dstLocal.GetPosition());
    uint64_t srcMaxOffsetBytes = 0;
    uint64_t dstMaxOffsetBytes = 0;
    std::string apiInfo = "DataCopy with DataCopyCO12DstParams from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition()));

    check::CalculateDataCopyMaxOffset<PrimT<U>, PrimT<T>>(
        srcPos, dstPos, intriParams, srcMaxOffsetBytes, dstMaxOffsetBytes);
    return check::ReportTensorSizeOverflow(srcPos, dstPos,
        srcLocal.GetSize() * sizeof(PrimT<U>), dstLocal.GetSize() * sizeof(PrimT<T>),
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T, typename U>
bool CheckDataCopyTensorSizeOverflow(const GlobalTensor<T>& dstGlobal, const LocalTensor<U>& srcLocal,
    const DataCopyCO12DstParams& intriParams) {
    if (intriParams.nSize == 0) {
        return true;
    }
    const Hardware srcPos = GetPhyType((TPosition)srcLocal.GetPosition());
    const Hardware dstPos = Hardware::GM;
    uint64_t srcMaxOffsetBytes = 0;
    uint64_t dstMaxOffsetBytes = 0;
    std::string apiInfo = "DataCopy with DataCopyCO12DstParams from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(TPosition::GM));

    check::CalculateDataCopyMaxOffset<PrimT<U>, PrimT<T>>(
        srcPos, dstPos, intriParams, srcMaxOffsetBytes, dstMaxOffsetBytes);
    return check::ReportTensorSizeOverflow(srcPos, dstPos,
        srcLocal.GetSize() * sizeof(PrimT<U>), 0,
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename SrcT, typename DstT>
bool CheckDataCopyTensorSizeOverflow(const LocalTensor<DstT> &dstLocal, const LocalTensor<SrcT> &srcLocal,
    const DataCopyParams &intriParams, const DataCopyEnhancedParams &enhancedParams) {
    if (intriParams.blockCount == 0 || intriParams.blockLen == 0) {
        return true;
    }
    const Hardware srcPos = GetPhyType((TPosition)srcLocal.GetPosition());
    const Hardware dstPos = GetPhyType((TPosition)dstLocal.GetPosition());
    uint64_t srcMaxOffsetBytes = 0;
    uint64_t dstMaxOffsetBytes = 0;
    BlockMode mode = check::GetBlockMode({ srcPos, dstPos }, enhancedParams.blockMode);
    uint8_t biasConvFlag = check::IsBiasConv({ srcPos, dstPos }) && (sizeof(PrimT<SrcT>) != sizeof(PrimT<DstT>));
    std::string apiInfo = "DataCopy with DataCopyEnhancedParams from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition()));

    check::CalculateDataCopyMaxOffset<PrimT<SrcT>, PrimT<DstT>>(
        intriParams, srcPos, dstPos, mode, srcMaxOffsetBytes, dstMaxOffsetBytes,
        enhancedParams.deqScale, biasConvFlag, enhancedParams.sidStoreMode);
    return check::ReportTensorSizeOverflow(srcPos, dstPos,
        srcLocal.GetSize() * sizeof(PrimT<SrcT>), dstLocal.GetSize() * sizeof(PrimT<DstT>),
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T>
bool CheckDataCopyPadTensorSizeOverflow(const LocalTensor<T> &dstLocal,
    const GlobalTensor<T> &srcGlobal, const DataCopyParams &dataCopyParams, const DataCopyPadParams &padParams) {
    const Hardware srcPos = Hardware::GM;
    const Hardware dstPos = GetPhyType((TPosition)dstLocal.GetPosition());
    if (dstPos != Hardware::UB) {
        return true;
    }
    uint64_t srcMaxOffsetBytes = 0;
    uint64_t paddingSize = (padParams.leftPadding + padParams.rightPadding) * sizeof(PrimT<T>);
    uint64_t dstMaxOffsetBytes = dataCopyParams.blockCount *
        AlignUp(dataCopyParams.blockLen + paddingSize, DEFAULT_C0_SIZE) +
        (dataCopyParams.blockCount - 1) * dataCopyParams.dstStride * DEFAULT_C0_SIZE;
    std::string apiInfo = "DataCopyPad with DataCopyParams and DataCopyPadParams from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(TPosition::GM)) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition()));

    return check::ReportTensorSizeOverflow(srcPos, dstPos,
        0, dstLocal.GetSize() * sizeof(PrimT<T>),
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T>
bool CheckDataCopyPadTensorSizeOverflow(const GlobalTensor<T> &dstGlobal,
    const LocalTensor<T> &srcLocal, const DataCopyParams &dataCopyParams) {
    const Hardware srcPos = GetPhyType((TPosition)srcLocal.GetPosition());
    const Hardware dstPos = Hardware::GM;
    if (srcPos != Hardware::UB) {
        return true;
    }
    uint64_t srcMaxOffsetBytes = dataCopyParams.blockCount * AlignUp(dataCopyParams.blockLen, DEFAULT_C0_SIZE) +
        (dataCopyParams.blockCount - 1) * dataCopyParams.srcStride * DEFAULT_C0_SIZE;
    uint64_t dstMaxOffsetBytes = 0;
    std::string apiInfo = "DataCopyPad with DataCopyParams from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(TPosition::GM));

    return check::ReportTensorSizeOverflow(srcPos, dstPos,
        srcLocal.GetSize() * sizeof(PrimT<T>), 0,
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T, typename U>
bool CheckDataCopyPadTensorSizeOverflow(const LocalTensor<T> &dstLocal,
    const GlobalTensor<T> &srcGlobal,
    const DataCopyExtParams &dataCopyParams, const DataCopyPadExtParams<U> &padParams) {
    const Hardware srcPos = Hardware::GM;
    const Hardware dstPos = GetPhyType((TPosition)dstLocal.GetPosition());
    if (dstPos != Hardware::UB) {
        return true;
    }
    uint64_t srcMaxOffsetBytes = 0;
    uint64_t paddingSize = (padParams.leftPadding + padParams.rightPadding) * sizeof(PrimT<T>);
    uint64_t dstMaxOffsetBytes = dataCopyParams.blockCount *
        AlignUp(dataCopyParams.blockLen + paddingSize, DEFAULT_C0_SIZE) +
        (dataCopyParams.blockCount - 1) * dataCopyParams.dstStride * DEFAULT_C0_SIZE;
    std::string apiInfo = "DataCopyPad with DataCopyExtParams and DataCopyPadExtParams from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(TPosition::GM)) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition()));

    return check::ReportTensorSizeOverflow(srcPos, dstPos,
        0, dstLocal.GetSize() * sizeof(PrimT<T>),
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T>
bool CheckDataCopyPadTensorSizeOverflow(const GlobalTensor<T> &dstGlobal,
    const LocalTensor<T> &srcLocal, const DataCopyExtParams &dataCopyParams) {
    const Hardware srcPos = GetPhyType((TPosition)srcLocal.GetPosition());
    const Hardware dstPos = Hardware::GM;
    if (srcPos != Hardware::UB) {
        return true;
    }
    uint64_t srcMaxOffsetBytes = dataCopyParams.blockCount * AlignUp(dataCopyParams.blockLen, DEFAULT_C0_SIZE) +
        (dataCopyParams.blockCount - 1) * dataCopyParams.srcStride * DEFAULT_C0_SIZE;
    uint64_t dstMaxOffsetBytes = 0;
    std::string apiInfo = "DataCopyPad with DataCopyExtParams from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(TPosition::GM));

    return check::ReportTensorSizeOverflow(srcPos, dstPos,
        srcLocal.GetSize() * sizeof(PrimT<T>), 0,
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T>
bool CheckDataCopyPadTensorSizeOverflow(const LocalTensor<T> &dstLocal,
    const LocalTensor<T> &srcLocal, const DataCopyParams &dataCopyParams, const Nd2NzParams &nd2nzParams) {
    if (dataCopyParams.blockCount == 0 || nd2nzParams.nValue == 0 ||
        nd2nzParams.dValue == 0 || nd2nzParams.ndNum == 0) {
        return true;
    }
    const Hardware srcPos = GetPhyType((TPosition)srcLocal.GetPosition());
    const Hardware dstPos = GetPhyType((TPosition)dstLocal.GetPosition());
    uint64_t srcMaxOffsetBytes = dataCopyParams.blockCount * AlignUp(dataCopyParams.blockLen, DEFAULT_C0_SIZE) +
        (dataCopyParams.blockCount - 1) * dataCopyParams.srcStride * DEFAULT_C0_SIZE;
    uint64_t dstMaxOffsetBytes = (nd2nzParams.ndNum - 1) * nd2nzParams.dstNzMatrixStride * sizeof(PrimT<T>) +
        (nd2nzParams.nValue - 1) * nd2nzParams.dstNzNStride * DEFAULT_C0_SIZE +
        (DivCeil(nd2nzParams.dValue * sizeof(PrimT<T>), DEFAULT_C0_SIZE) - 1) *
        nd2nzParams.dstNzC0Stride * DEFAULT_C0_SIZE +
        DEFAULT_C0_SIZE;
    std::string apiInfo = "DataCopyPad with DataCopyParams and Nd2NzParams from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition()));

    return check::ReportTensorSizeOverflow(srcPos, dstPos,
        srcLocal.GetSize() * sizeof(PrimT<T>), dstLocal.GetSize() * sizeof(PrimT<T>),
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

template <typename T>
bool CheckDataCopyPadTensorSizeOverflow(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const DataCopyExtParams &dataCopyParams, const Nd2NzParams &nd2nzParams) {
    if (dataCopyParams.blockCount == 0 || nd2nzParams.nValue == 0 ||
        nd2nzParams.dValue == 0 || nd2nzParams.ndNum == 0) {
        return true;
    }
    const Hardware srcPos = GetPhyType((TPosition)srcLocal.GetPosition());
    const Hardware dstPos = GetPhyType((TPosition)dstLocal.GetPosition());
    uint64_t srcMaxOffsetBytes = dataCopyParams.blockCount * AlignUp(dataCopyParams.blockLen, DEFAULT_C0_SIZE) +
        (dataCopyParams.blockCount - 1) * dataCopyParams.srcStride * DEFAULT_C0_SIZE;
    uint64_t dstMaxOffsetBytes = (nd2nzParams.ndNum - 1) * nd2nzParams.dstNzMatrixStride * sizeof(PrimT<T>) +
        (nd2nzParams.nValue - 1) * nd2nzParams.dstNzNStride * DEFAULT_C0_SIZE +
        (DivCeil(nd2nzParams.dValue * sizeof(PrimT<T>), DEFAULT_C0_SIZE) - 1) *
        nd2nzParams.dstNzC0Stride * DEFAULT_C0_SIZE +
        DEFAULT_C0_SIZE;
    std::string apiInfo = "DataCopyPad with DataCopyExtParams and Nd2NzParams from " +
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())) +
        " to " + ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition()));

    return check::ReportTensorSizeOverflow(srcPos, dstPos,
        srcLocal.GetSize() * sizeof(PrimT<T>), dstLocal.GetSize() * sizeof(PrimT<T>),
        srcMaxOffsetBytes, dstMaxOffsetBytes, apiInfo);
}

} // namespace AscendC
#endif

#endif