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
 * \file foreach_op_compute.h
 * \brief
 */
#ifndef __FOREACH_OP_COMPUTE_H__
#define __FOREACH_OP_COMPUTE_H__
#include "kernel_operator.h"
namespace foreachOp {

// op based ascend C api
template <typename DataType>
__aicore__ void addOnTensorList(LocalTensor<DataType> dst, LocalTensor<DataType> src1, LocalTensor<DataType> src2,
                                uint32_t dataCount) {
  Add(dst, src1, src2, dataCount);
}

template <typename DataType, typename ScalarType>
__aicore__ void addOnTensorListWithScalar(LocalTensor<DataType> dst, LocalTensor<DataType> src1,
                                          LocalTensor<DataType> src2, ScalarType alpha, uint32_t dataCount) {
  Muls(src2, src2, alpha, dataCount);
  Add(dst, src1, src2, dataCount);
}

template <typename DataType, typename ScalarType>
__aicore__ void addsOnTensorListAndScalar(LocalTensor<DataType> dst, LocalTensor<DataType> src1, ScalarType alpha,
                                          uint32_t dataCount) {
  Adds(dst, src1, alpha, dataCount);
}

template <typename DataType>
__aicore__ void expOnTensorList(LocalTensor<DataType> dst, LocalTensor<DataType> src1, uint32_t dataCount) {
  Exp(dst, src1, dataCount);
}

template <typename DataType>
__aicore__ void absOnTensorList(LocalTensor<DataType> dst, LocalTensor<DataType> src1, uint32_t dataCount) {
  Abs(dst, src1, dataCount);
}

// input + scalar * tensor1 * tensor2
template <typename DataType>
__aicore__ void addcmulOnTensorList(LocalTensor<DataType> dst, LocalTensor<DataType> src1, LocalTensor<DataType> src2,
                                    LocalTensor<DataType> src3, uint32_t dataCount) {
  Mul(dst, src2, src3, dataCount);
  Add(dst, dst, src1, dataCount);
}
}  // namespace foreachOp
#endif  // __FOREACH_OP_COMPUTE_H__
