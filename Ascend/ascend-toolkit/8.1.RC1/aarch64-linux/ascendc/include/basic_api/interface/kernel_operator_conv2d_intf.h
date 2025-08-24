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
 * \file kernel_operator_conv2d_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_CONV2D_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_CONV2D_INTERFACE_H
#include "kernel_tensor.h"
#include "impl/kernel_operator_conv2d_base_impl.h"
#include "kernel_operator_data_copy_intf.h"

namespace AscendC {
// T should be featureMap matrix dtype
template <typename T> __aicore__ inline Conv2dTilling GetConv2dTiling(Conv2dParams& conv2dParams);

/*
 * @ingroup：Conv2D
 * @brief：Given an input tensor and a weight tensor, perform the convolution 2-D operation, and output the
 * result tensor
 * @param [out] dstLocal output LocalTensor
 * @param [in] bias input LocalTensor
 * @param [in] featureMap input LocalTensor
 * @param [in] weight input LocalTensor
 * @param [in] conv2dParams.imgShape Shape of "featureMap"
 * @param [in] conv2dParams.kernel Shape shape of "weight"
 * @param [in] conv2dParams.stride Stride of convolution
 * @param [in] conv2dParams.cin Fractal layout parameter,cin = c1 * c0
 * @param [in] conv2dParams.cout Fractal layout parameter
 * @param [in] conv2dParams.padList Padding rows/columns
 * @param [in] conv2dParams.dilation Void convolution parameter
 * @param [in] conv2dParams.initY dst initialization parameters
 * @param [in] conv2dParams.partialSum Judge whether the result is moved out
 */
template <typename dst_T, typename src_T>
__aicore__ inline __in_pipe__(MTE2)
    __out_pipe__(MTE3) void Conv2D(const LocalTensor<dst_T> &dstLocal, const LocalTensor<src_T> &featureMap,
    const LocalTensor<src_T> &weight, Conv2dParams &conv2dParams, Conv2dTilling &tilling);

template <typename dst_T, typename src_T>
__aicore__ inline __in_pipe__(MTE2)__out_pipe__(MTE3) void Conv2D(const LocalTensor<dst_T> &dstLocal,
    const LocalTensor<dst_T> &bias, const LocalTensor<src_T> &featureMap, const LocalTensor<src_T> &weight,
    Conv2dParams &conv2dParams, Conv2dTilling &tilling);
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_CONV2D_INTERFACE_H
