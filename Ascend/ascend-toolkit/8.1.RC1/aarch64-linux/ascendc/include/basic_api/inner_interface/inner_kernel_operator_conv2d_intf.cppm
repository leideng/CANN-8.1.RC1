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
 * \file inner_kernel_operator_conv2d_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_CONV2D_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_CONV2D_INTERFACE_H
#include "kernel_tensor.h"
#include "impl/kernel_operator_conv2d_base_impl.h"
#include "kernel_operator_data_copy_intf.h"
#include "kernel_struct_data_copy.h"

namespace AscendC {
// T should be featureMap matrix dtype
template <typename T>
[[deprecated("NOTICE: GetConv2dTiling has been deprecated and will be removed in the next version. "
        "Please do not use it!")]]
__aicore__ inline Conv2dTilling GetConv2dTiling(Conv2dParams& conv2dParams)
{
    Conv2dTilling tilling;
    GetTypeforC0<T>(conv2dParams, tilling);

    tilling.loopMode = LoopMode::MODE_NM;

    tilling.strideH = conv2dParams.stride[0];
    tilling.strideW = conv2dParams.stride[1];
    tilling.dilationH = conv2dParams.dilation[0];
    tilling.dilationW = conv2dParams.dilation[1];
    tilling.hi = conv2dParams.imgShape[0];
    tilling.wi = conv2dParams.imgShape[1];
    tilling.ho = (tilling.hi + conv2dParams.padList[2] + conv2dParams.padList[3] -
        tilling.dilationH * (conv2dParams.kernelShape[0] - 1) - 1) /
        tilling.strideH +
        1;
    tilling.wo = (tilling.wi + conv2dParams.padList[0] + conv2dParams.padList[1] -
        tilling.dilationW * (conv2dParams.kernelShape[1] - 1) - 1) /
        tilling.strideW +
        1;

    tilling.height = conv2dParams.kernelShape[0];
    tilling.width = conv2dParams.kernelShape[1];

    tilling.howo = tilling.ho * tilling.wo;

    tilling.mNum = tilling.howo;
    tilling.nNum = conv2dParams.cout;
    tilling.kNum = conv2dParams.cin * tilling.height * tilling.width;

    CalculateConv2dTiling(tilling);

    return tilling;
}

/*
 * @ingroup：Conv2D
 * @brief：Given an input tensor and a weight tensor, perform the convolution 2-D operation, and output the
 * result tensor
 * @param [out] dstLocal output LocalTensor
 * @param [in] bias input LocalTensor
 * @param [in] featureMap input LocalTensor
 * @param [in] weight input LocalTensor
 * @param [in] intriParams.imgShape Shape of "featureMap"
 * @param [in] intriParams.kernel Shape shape of "weight"
 * @param [in] intriParams.stride Stride of convolution
 * @param [in] intriParams.cin Fractal layout parameter,cin = c1 * c0
 * @param [in] intriParams.cout Fractal layout parameter
 * @param [in] intriParams.padList Padding rows/columns
 * @param [in] intriParams.dilation Void convolution parameter
 * @param [in] intriParams.initY dst initialization parameters
 * @param [in] intriParams.partialSum Judge whether the result is moved out
 */
template <typename dst_T, typename src_T>
[[deprecated("NOTICE: Conv2D has been deprecated and will be removed in the next version. "
        "Please do not use it!")]]
__aicore__ inline __in_pipe__(MTE2)
    __out_pipe__(MTE3) void Conv2D(const LocalTensor<dst_T> &dstLocal, const LocalTensor<src_T> &featureMap,
    const LocalTensor<src_T> &weight, Conv2dParams &conv2dParams, Conv2dTilling &tilling)
{
    if (conv2dParams.initY == 2) {
        return;
    }

    Conv2D(dstLocal, dstLocal, featureMap, weight, conv2dParams, tilling);
}

template <typename dst_T, typename src_T>
[[deprecated("NOTICE: Conv2D has been deprecated and will be removed in the next version. "
        "Please do not use it!")]]
__aicore__ inline __in_pipe__(MTE2)__out_pipe__(MTE3) void Conv2D(const LocalTensor<dst_T> &dstLocal,
    const LocalTensor<dst_T> &bias, const LocalTensor<src_T> &featureMap, const LocalTensor<src_T> &weight,
    Conv2dParams &conv2dParams, Conv2dTilling &tilling)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckConv2DParams(dstLocal, bias, featureMap, weight, conv2dParams, tilling)) {
        return;
    }
#endif

#if __CCE_AICORE__ < 220
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    DataCopyEnhancedParams enhancedParams;
    enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;
#endif

    const Hardware dstScope = GetPhyType((TPosition)dstLocal.GetPosition());
    LocalTensor<dst_T> L0c;
    if (dstScope == Hardware::L0C) {
        L0c = dstLocal[0];
    } else {
#if __CCE_AICORE__ < 220
        TBuffAddr tbufc;
        tbufc.logicPos = (uint8_t)TPosition::C2;
        L0c.SetAddr(tbufc);
        L0c.InitBuffer(0, TOTAL_L0C_SIZE / sizeof(PrimT<dst_T>));

        dataCopyParams.blockLen = dstLocal.GetSize() * sizeof(PrimT<dst_T>) / 1024;
        DataCopy(L0c, dstLocal, dataCopyParams, enhancedParams);
#endif
    }

    if (tilling.loopMode == LoopMode::MODE_NM) {
        Conv2DExecNm(L0c, bias, featureMap, weight, conv2dParams, tilling);
    } else if (tilling.loopMode == LoopMode::MODE_MN) {
        Conv2DExecMn(L0c, bias, featureMap, weight, conv2dParams, tilling);
    } else {
        // other mode are not supported
    }

#if __CCE_AICORE__ < 220
    if ((!conv2dParams.partialSum) && (dstScope == Hardware::UB)) {
        pipe_barrier(PIPE_ALL);
        dataCopyParams.blockLen = tilling.roundM * tilling.roundN * sizeof(PrimT<dst_T>) / 1024;
        DataCopy(dstLocal, L0c, dataCopyParams, enhancedParams);
    }
#endif
}
} // namespace AscendC
#endif // ASCENDC_MODULE_INNER_OPERATOR_CONV2D_INTERFACE_H
