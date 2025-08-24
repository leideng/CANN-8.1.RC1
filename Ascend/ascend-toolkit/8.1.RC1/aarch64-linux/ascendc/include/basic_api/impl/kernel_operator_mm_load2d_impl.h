/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file kernel_operator_mm_load2d_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_MM_LOAD2D_H
#define ASCENDC_MODULE_OPERATOR_MM_LOAD2D_H
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
__aicore__ inline void LoadDataImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LoadData2DParams& loadDataParams)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncLoadData2d(dstLocal, srcLocal, loadDataParams, "LoadData with LoadData2DParams")) {
        ASCENDC_REPORT_CHECK_ERROR("LoadData with LoadData2DParams", KernelFuncType::NONE_MODE);
    }
#endif
    CheckTensorPos<T>(srcLocal, Hardware::L1, "srcLocal", "A1 / B1", "LoadData with LoadData2DParams");
    CheckTensorAlign<T>(srcLocal, ONE_BLK_SIZE, "srcLocal", "LoadData with LoadData2DParams");
    CheckTensorAlign<T>(dstLocal, VALUE_512, "dstLocal", "LoadData with LoadData2DParams");
    const Hardware dstScope = GetPhyType((TPosition)dstLocal.GetPosition());
    if (dstScope == Hardware::L0A) {
        LoadData2DL12L0ACal((__ca__ PrimT<T>*)dstLocal.GetPhyAddr(),
                            (__cbuf__ PrimT<T>*)srcLocal.GetPhyAddr(), loadDataParams);
    } else if (dstScope == Hardware::L0B) {
        LoadData2DL12L0BCal((__cb__ PrimT<T>*)dstLocal.GetPhyAddr(),
                            (__cbuf__ PrimT<T>*)srcLocal.GetPhyAddr(), loadDataParams);
    } else {
        ASCENDC_CHECK_TPOSITION(false, "dstLocal", "A2 / B2",
            "LoadData with LoadData2DParams",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
    }
}

template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void LoadDataImpl(const LocalTensor<T>& dstLocal,
    const GlobalTensor<T>& srcLocal, const LoadData2DParams& loadDataParams)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncLoadData2d(dstLocal, srcLocal, loadDataParams, "LoadData with LoadData2DParams")) {
        ASCENDC_REPORT_CHECK_ERROR("LoadData with LoadData2DParams", KernelFuncType::NONE_MODE);
    }
#endif
    const Hardware dstScope = GetPhyType((TPosition)dstLocal.GetPosition());
    if (dstScope == Hardware::L0A) {
        CheckTensorAlign<T>(dstLocal, VALUE_512, "dstLocal", "LoadData with LoadData2DParams");
        LoadData2DGM2L0ACal((__ca__ PrimT<T>*)dstLocal.GetPhyAddr(),
                            (__gm__ PrimT<T>*)srcLocal.GetPhyAddr(), loadDataParams);
    } else if (dstScope == Hardware::L0B) {
        CheckTensorAlign<T>(dstLocal, VALUE_512, "dstLocal", "LoadData with LoadData2DParams");
        LoadData2DGM2L0BCal((__cb__ PrimT<T>*)dstLocal.GetPhyAddr(),
                            (__gm__ PrimT<T>*)srcLocal.GetPhyAddr(), loadDataParams);
    } else if (dstScope == Hardware::L1) {
        CheckTensorAlign<T>(dstLocal, ONE_BLK_SIZE, "dstLocal",
            "LoadData with LoadData2DParams");
        LoadData2DGM2L1Cal((__cbuf__ PrimT<T>*)dstLocal.GetPhyAddr(),
                           (__gm__ PrimT<T>*)srcLocal.GetPhyAddr(), loadDataParams);
    } else {
        ASCENDC_CHECK_TPOSITION(false, "dstLocal", "A1 / B1 / A2 / B2",
            "LoadData with LoadData2DParams",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
    }
}

/* **************************************************************************************************
 * LoadData 2d with transpose                                             *
 * ************************************************************************************************* */
/*
 * @ingroup DataLoad
 * @brief Cube data loading
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] loadDataParams.startIndex Fractal matrix ID in unit of fractal nums depending on dtype
 * @param [in] loadDataParams.repeatTimes repeat times
 * @param [in] loadDataParams.srcStride src block stride in unit of fractal nums depending on dtype
 * @param [in] loadDataParams.dstGap interval between the previous tail and the next fractal head in unit of one 512byte
 * fractal
 * @param [in] loadDataParams.dstFracGap dst fractal gap in unit of one 512byte fractal
 */
template <typename T>
__aicore__ inline void LoadDataWithTransposeImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LoadData2dTransposeParams& loadDataParams)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncLoadDataTranspose(dstLocal, srcLocal, loadDataParams, "LoadDataWithTranspose")) {
        ASCENDC_REPORT_CHECK_ERROR("LoadDataWithTranspose with LoadData2dTransposeParams", KernelFuncType::NONE_MODE);
    }
#endif
    CheckTensorAlign<T>(srcLocal, ONE_BLK_SIZE, "srcLocal", "LoadDataWithTranspose"); // L1 32B align
#if !defined(__DAV_M310__)
    CheckTensorAlign<T>(dstLocal, VALUE_512, "dstLocal", "LoadDataWithTranspose");    // L0A/L0B 512B align
#endif
    CheckTensorPos<T>(srcLocal, Hardware::L1, "srcLocal", "A1 / B1", "LoadDataWithTranspose");
    const Hardware dstScope = GetPhyType((TPosition)dstLocal.GetPosition());
    if (dstScope == Hardware::L0A) {
        LoadData2DL12L0ATransposeCal((__ca__ PrimT<T>*)dstLocal.GetPhyAddr(),
            (__cbuf__ PrimT<T>*)srcLocal.GetPhyAddr(),
            loadDataParams);
    } else if (dstScope == Hardware::L0B) {
        LoadData2DL12L0BTransposeCal((__cb__ PrimT<T>*)dstLocal.GetPhyAddr(),
            (__cbuf__ PrimT<T>*)srcLocal.GetPhyAddr(),
            loadDataParams);
    } else {
        ASCENDC_CHECK_TPOSITION(false, "dstLocal", "A2/B2","LoadDataWithTranspose with LoadData2dTransposeParams",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
    }
}

/*
 * @ingroup DataLoad
 * @brief Cube data loading
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] loadDataParams.startIndex Fractal matrix ID in unit of one 512byte fractal
 * @param [in] loadDataParams.repeatTimes repeat times
 * @param [in] loadDataParams.srcStride src block stride in unit of one 512byte fractal
 * @param [in] loadDataParams.dstGap interval between the previous tail and the next fractal head in unit of one 512byte
 * fractal
 * @param [in] loadDataParams.dstFracGap dst fractal gap in unit of one 512byte fractal
 * @param [in] loadDataParams.srcFracGap src fractal gap in unit of one 512byte fractal
 */
template <typename T>
__aicore__ inline void LoadDataWithTransposeImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LoadData2dTransposeParamsV2& loadDataParams)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncLoadDataTranspose(dstLocal, srcLocal, loadDataParams, "LoadDataWithTranspose")) {
        ASCENDC_REPORT_CHECK_ERROR("LoadDataWithTranspose with LoadData2dTransposeParamsV2", KernelFuncType::NONE_MODE);
    }
#endif
    CheckTensorAlign<T>(srcLocal, ONE_BLK_SIZE, "srcLocal", "LoadDataWithTranspose"); // L1 32B align
#if !defined(__DAV_M310__)
    CheckTensorAlign<T>(dstLocal, VALUE_512, "dstLocal", "LoadDataWithTranspose");          // L0A/L0B 512B align
#endif
    CheckTensorPos<T>(srcLocal, Hardware::L1, "srcLocal", "A1 / B1", "LoadDataWithTranspose");
    const Hardware dstScope = GetPhyType((TPosition)dstLocal.GetPosition());
    if (dstScope == Hardware::L0B) {
        LoadData2DL12L0BTransposeCal((__cb__ PrimT<T>*)dstLocal.GetPhyAddr(),
            (__cbuf__ PrimT<T>*)srcLocal.GetPhyAddr(),
            loadDataParams);
    } else {
        ASCENDC_CHECK_TPOSITION(false, "dstLocal", "B2", "LoadDataWithTranspose with LoadData2dTransposeParamsV2",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
    }
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
__aicore__ inline void LoadDataImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LoadData2DParamsV2& loadDataParams)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncLoadData2dv2(dstLocal, srcLocal, loadDataParams, "LoadData with LoadData2DParamsV2")) {
        ASCENDC_REPORT_CHECK_ERROR("LoadData with LoadData2DParamsV2", KernelFuncType::NONE_MODE);
    }
#endif
    CheckTensorPos<T>(srcLocal, Hardware::L1, "srcLocal", "A1 / B1",
        "LoadData with LoadData2DParamsV2");
    const Hardware dstScope = GetPhyType((TPosition)dstLocal.GetPosition());
    if (dstScope == Hardware::L0A) {
        LoadData2DL12L0ACal((__ca__ PrimT<T>*)dstLocal.GetPhyAddr(),
                            (__cbuf__ PrimT<T>*)srcLocal.GetPhyAddr(), loadDataParams);
    } else if (dstScope == Hardware::L0B) {
        LoadData2DL12L0BCal((__cb__ PrimT<T>*)dstLocal.GetPhyAddr(),
                            (__cbuf__ PrimT<T>*)srcLocal.GetPhyAddr(), loadDataParams);
    } else {
        ASCENDC_CHECK_TPOSITION(false, "dstLocal", "A2 / B2",
            "LoadData with LoadData2DParamsV2",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
    }
}

template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void LoadDataImpl(const LocalTensor<T>& dstLocal,
    const GlobalTensor<T>& srcLocal, const LoadData2DParamsV2& loadDataParams)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncLoadData2dv2(dstLocal, srcLocal, loadDataParams, "LoadData with LoadData2DParamsV2")) {
        ASCENDC_REPORT_CHECK_ERROR("LoadData with LoadData2DParamsV2", KernelFuncType::NONE_MODE);
    }
#endif
    const Hardware dstScope = GetPhyType((TPosition)dstLocal.GetPosition());
    if (dstScope == Hardware::L0A) {
        LoadData2DGM2L0ACal((__ca__ PrimT<T>*)dstLocal.GetPhyAddr(),
                            (__gm__ PrimT<T>*)srcLocal.GetPhyAddr(), loadDataParams);
    } else if (dstScope == Hardware::L0B) {
        LoadData2DGM2L0BCal((__cb__ PrimT<T>*)dstLocal.GetPhyAddr(),
                            (__gm__ PrimT<T>*)srcLocal.GetPhyAddr(), loadDataParams);
    } else if (dstScope == Hardware::L1) {
        LoadData2DGM2L1Cal((__cbuf__ PrimT<T>*)dstLocal.GetPhyAddr(),
                           (__gm__ PrimT<T>*)srcLocal.GetPhyAddr(), loadDataParams);
    } else {
        ASCENDC_CHECK_TPOSITION(false, "dstLocal", "A1 / B1 / A2 / B2",
            "LoadData with LoadData2DParamsV2",
            ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_MM_LOAD2D_H