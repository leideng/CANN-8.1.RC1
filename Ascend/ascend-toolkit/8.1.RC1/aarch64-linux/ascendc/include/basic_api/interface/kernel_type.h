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
 * \file kernel_type.h
 * \brief
 */
#ifndef KERNEL_TYPE_H
#define KERNEL_TYPE_H

#define DT_FLOAT 0            // float32
#define DT_FLOAT16 1          // half
#define DT_INT8 2             // int8
#define DT_INT32 3            // int32
#define DT_UINT8 4            // u8
#define DT_INT16 6            // int16
#define DT_UINT16 7           // u16
#define DT_INT64 9            // int64
#define DT_UINT32 8           // u32
#define DT_UINT64 10          // u64
#define DT_BOOL 11
#define DT_DOUBLE 12
#define DT_STRING 13          // string
#define DT_DUAL_SUB_INT8 14   // dual output int8
#define DT_DUAL_SUB_UINT8 15  // dual output u8
#define DT_COMPLEX64 16       // complex64
#define DT_COMPLEX128 17      // complex128
#define DT_QINT8 18           // qint8
#define DT_QINT16 19          // qint16
#define DT_QINT32 20          // qint32
#define DT_QUINT8 21          // quint8
#define DT_QUINT16 22         // quint16
#define DT_RESOURCE 23        // resource
#define DT_STRING_REF 24      // string ref
#define DT_DUAL 25            // dual output
#define DT_VARIANT 26         // dt_variant
#define DT_BF16 27            // bf16
#define DT_UNDEFINED 28       // Indicate a DataType field has not been set.
#define DT_INT4 29            // int4
#define DT_UINT1 30           // u1
#define DT_INT2 31            // int2
#define DT_UINT2 32           // u2
#define DT_MAX 33             // Mark the boundaries of AscendCData type

#define FORMAT_NCHW 0  // NCHW Tensor
#define FORMAT_NHWC 1  // NHWC Tensor
#define FORMAT_ND 2 // ND Tensor
#define FORMAT_NC1HWC0 3 // NC1HWC0
#define FORMAT_FRACTAL_Z 4  // FRACTAL_Z for cube
#define FORMAT_NC1C0HWPAD 5 // NC1C0HWPAD
#define FORMAT_NHWC1C0 6 // NHWC1C0 Tensor
#define FORMAT_FSR_NCHW 7 // FSR NCHW Tensor
#define FORMAT_FRACTAL_DECONV 8 // DECONV
#define FORMAT_C1HWNC0 9 // C1HWNC0
#define FORMAT_FRACTAL_DECONV_TRANSPOSE 10  // TRANSPOSE
#define FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS 11
#define FORMAT_NC1HWC0_C04 12 // NC1HWC0 C0 is 4
#define FORMAT_FRACTAL_Z_C04 13 // FRACZ C0 is 4
#define FORMAT_CHWN 14 // CHWN
#define FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS 15
#define FORMAT_HWCN 16 // HWCN
#define FORMAT_NC1KHKWHWC0 17 // KHKW kernel h& kernel w maxpooling max output
#define FORMAT_BN_WEIGHT 18 // Batch Normalization layer
#define FORMAT_FILTER_HWCK 19 // filter input tensor
#define FORMAT_HASHTABLE_LOOKUP_LOOKUPS 20
#define FORMAT_HASHTABLE_LOOKUP_KEYS 21
#define FORMAT_HASHTABLE_LOOKUP_VALUE 22
#define FORMAT_HASHTABLE_LOOKUP_OUTPUT 23
#define FORMAT_HASHTABLE_LOOKUP_HITS 24
#define FORMAT_C1HWNCoC0 25  // C1HWNCoC0
#define FORMAT_MD 26
#define FORMAT_NDHWC 27 // NDHWC
#define FORMAT_FRACTAL_ZZ 28 // ZZ for cube
#define FORMAT_FRACTAL_NZ 29 // NZ for cube
#define FORMAT_NCDHW 30 // NCDHW
#define FORMAT_DHWCN 31 // 3D filter input tensor
#define FORMAT_NDC1HWC0 32 // NDC1HWC0
#define FORMAT_FRACTAL_Z_3D 33 // 05jgfd9
#define FORMAT_CN 34 // CN
#define FORMAT_NC 35 // NC
#define FORMAT_DHWNC 36 // DHWNCX
#define FORMAT_FRACTAL_Z_3D_TRANSPOSE 37 // 3D filter(transpose) input tensor
#define FORMAT_FRACTAL_ZN_LSTM 38 // For LSTM Net
#define FORMAT_FRACTAL_Z_G 39
#define FORMAT_RESERVED 40
#define FORMAT_ALL 41
#define FORMAT_NULL 42
#define FORMAT_ND_RNN_BIAS 43 // Bias Format for RNN
#define FORMAT_FRACTAL_ZN_RNN 44 // ZN for RNN
#define FORMAT_NYUV 45
#define FORMAT_NYUV_A 46
#define FORMAT_NCL 47
// Add new formats here

#define FORMAT_MAX 0xff
#endif