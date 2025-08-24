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
#ifndef OP_API_INC_CONVOLUTION_H_
#define OP_API_INC_CONVOLUTION_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief convolution接口，计算并获取workspace大小
 * @domain aclnn_ops_infer
 *
 * @param [in] input: npu，feature map
 * device侧的aclTensor，数据类型浮点类型FLOAT16，FLOAT32，FLOAT64
 * 支持非连续的Tensor，数据格式支持ND、NCHW、NHWC、HWCN、NDHWC、NCDHW
 * @param [in] weight: npu, kernels
 * device侧的aclTensor，数据类型与input一致
 * 支持非连续的Tensor，数据格式与input一致
 * @param [in] bias: npu，偏置
 * device侧的aclTensor，数据类型与input一致
 * 支持非连续的Tensor，数据格式与input一致
 * @param [in] stride: 步长
 * int64的数组，数组长度需等于input的维度-2（也等于kernel size -1），例：2D卷积的步长数组的有效长度是2位
 * @param [in] padding: 补边
 * int64的数组，数组长度需等于input的维度-2（也等于kernel size -1），例：2D卷积的padding数组的有效长度是2位
 * @param [in] dilation: kernel中元素的间隔，>1代表空洞卷积
 * int64的数组，数组长度需等于input的维度-2（也等于kernel size -1），例：2D卷积的dilation数组的有效长度是2位
 * @param [in] transposed: 是否转置
 * bool，True代表转置卷积
 * @param [in] outputPadding：转置卷积时生效，对输出的补边
 * int64的数组，数组长度需等于input的维度-2，值必须分别小于stride或者dilation的最大值，例：2D转置卷积的dilation数组的有效长度是2位
 * @param [in] groups：分组数，表示从输入通道到输出通道的块链接个数
 * int64，大于0且能整除input和output的通道数， input通道数 = weight通道数*groups
 * @param [out] output: npu
 * device侧的aclTensor，数据类型与input一致
 * broadcast之后的shape，数据格式与input一致
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnConvolutionGetWorkspaceSize(const aclTensor* input, const aclTensor* weight,
                                                       const aclTensor* bias, const aclIntArray* stride,
                                                       const aclIntArray* padding, const aclIntArray* dilation,
                                                       bool transposed, const aclIntArray* outputPadding,
                                                       const int64_t groups, aclTensor* output, int8_t cubeMathType,
                                                       uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnConvTbcGetWorkspaceSize(const aclTensor* self, const aclTensor* weight,
                                                   const aclTensor* bias, const int64_t pad, aclTensor* output,
                                                   int8_t cubeMathType, uint64_t* workspaceSize,
                                                   aclOpExecutor** executor);

/**
 * @brief aclnnConvDepthwise2d的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnConvDepthwise2dGetWorkspaceSize(const aclTensor* self, const aclTensor* weight,
                                                           const aclIntArray* kernelSize, const aclTensor* bias,
                                                           const aclIntArray* stride, const aclIntArray* padding,
                                                           const aclIntArray* dilation, aclTensor* out,
                                                           int8_t cubeMathType, uint64_t* workspaceSize,
                                                           aclOpExecutor** executor);

/**
 * @brief convolution接口，进行kernellaunch
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由aclnnConvolutionGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。调用该接口后，executor不再可用
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnConvolution(void* workspace, const uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

ACLNN_API aclnnStatus aclnnConvTbc(void* workspace, const uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream);

ACLNN_API aclnnStatus aclnnConvDepthwise2d(void* workspace, const uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_CONVOLUTION_H_
