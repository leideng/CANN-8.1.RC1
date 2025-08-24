/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
#include "sparse_to_dense.h"

#include <securec.h>
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace aicpu {
const char *const SPARSETODENSE = "SparseToDense";
}

namespace aicpu {
uint32_t SparseToDenseCpuKernel::SparseToDense(const CpuKernelContext &ctx, 
                                               SparseTensor &st,
                                               Tensor *indices,
                                               Tensor *output) {
  KERNEL_LOG_INFO("Start to execute SparseToDense");
  if (indices == nullptr || output == nullptr) {
    KERNEL_LOG_ERROR("Indices or output tensor is nullptr.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  DataType dt = static_cast<DataType>(output->GetDataType());
  switch (dt) {
    case DT_INT8:
      return EigenSparseToDense<int8_t>(ctx, st, indices, output);
    case DT_UINT8:
      return EigenSparseToDense<uint8_t>(ctx, st, indices, output);
    case DT_INT16:
      return EigenSparseToDense<int16_t>(ctx, st, indices, output);
    case DT_UINT16:
      return EigenSparseToDense<uint16_t>(ctx, st, indices, output);
    case DT_INT32:
      return EigenSparseToDense<int32_t>(ctx, st, indices, output);
    case DT_INT64:
      return EigenSparseToDense<int64_t>(ctx, st, indices, output);
    case DT_FLOAT16:
      return EigenSparseToDense<Eigen::half>(ctx, st, indices, output);
    case DT_FLOAT:
      return EigenSparseToDense<float>(ctx, st, indices, output);
    case DT_BOOL:
      return EigenSparseToDense<bool>(ctx, st, indices, output);
    case DT_DOUBLE:
      return EigenSparseToDense<double>(ctx, st, indices, output);
    default:
      KERNEL_LOG_ERROR("Sparse to dense can't support this data type [%d].",
                       dt);
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

uint32_t SparseToDenseCpuKernel::ValidParam(const CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("Start to execute ValidParam");
  // valid input and output nullptr
  Tensor *indices_tensor = ctx.Input(0);
  Tensor *shape_tensor = ctx.Input(1);
  Tensor *sparse_values = ctx.Input(2);
  Tensor *default_value_tensor = ctx.Input(3);
  Tensor *output_tensor = ctx.Output(0);
  bool validNull =
      ((output_tensor == nullptr) || default_value_tensor == nullptr ||
       (sparse_values == nullptr) || (indices_tensor == nullptr) ||
       (shape_tensor == nullptr));
  if (validNull) {
    KERNEL_LOG_ERROR("Got input or output param is nullptr.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // valid shape nullptr
  auto output_shape = shape_tensor->GetTensorShape();
  auto values_shape = sparse_values->GetTensorShape();
  auto default_value_shape = default_value_tensor->GetTensorShape();
  auto indices_shape = indices_tensor->GetTensorShape();
  bool validShapeNull =
      ((default_value_shape == nullptr) || values_shape == nullptr ||
       (output_shape == nullptr) || (indices_shape == nullptr));
  if (validShapeNull) {
    KERNEL_LOG_ERROR("Got input shape is nullptr.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // sparse_indices
  if (indices_shape->GetDims() > 2) {
    KERNEL_LOG_ERROR(
        "Sparse_indices should be a scalar, vector, or matrix, got dim "
        "size [%d].",
        indices_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  const int64_t elems_num =
      indices_shape->GetDims() > 0 ? indices_shape->GetDimSize(0) : 1;
  const int64_t dims_num =
      indices_shape->GetDims() > 1 ? indices_shape->GetDimSize(1) : 1;

  // output_shape
  if (output_shape->GetDims() != 1) {
    KERNEL_LOG_ERROR("Output_shape should be a vector, and got dim size [%d].",
                     output_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (shape_tensor->NumElements() != dims_num) {
    KERNEL_LOG_ERROR(
        "Output_shape has incorrect number of elements [%lld], should be [%lld]",
        shape_tensor->NumElements(), dims_num);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // valid data type
  int32_t IndiceType = indices_tensor->GetDataType();
  int32_t outShapeType = shape_tensor->GetDataType();
  bool validIndiceType = ((IndiceType != DT_INT32) && (IndiceType != DT_INT64));
  bool validShapeType =
      ((outShapeType != DT_INT32) && (outShapeType != DT_INT64));
  if (validShapeType || validIndiceType) {
    KERNEL_LOG_ERROR("Valid indice or output shape data type failed, indiceType [%d], "
        "shapeType [%d].", IndiceType, outShapeType);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // sparse_values
  int32_t values_dims_size = values_shape->GetDims();
  if ((values_dims_size != 0) && (values_dims_size != 1)) {
    KERNEL_LOG_ERROR(
        "Values_shape should be a scalar or a vector, got dim size [%d].",
        values_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if ((values_dims_size == 1) && (sparse_values->NumElements() != elems_num)) {
    KERNEL_LOG_ERROR(
        "Values_shape has incorrect number of elements [%lld], should be [%lld]",
        sparse_values->NumElements(), elems_num);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // default_value
  if (default_value_shape->GetDims() != 0) {
    KERNEL_LOG_ERROR("Default_value should be a scalar, and got dim size [%d].",
                     default_value_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_LOG_INFO("Execute ValidParam end.");
  return KERNEL_STATUS_OK;
}

uint32_t SparseToDenseCpuKernel::Compute(CpuKernelContext &ctx) {
  if (ValidParam(ctx) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Valid sparse to dense param error.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *indices_tensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(indices_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Indices_tensor is null")
  Tensor *shape_tensor = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(shape_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Shape_tensor is null")
  Tensor *sparse_values = ctx.Input(2);
  KERNEL_CHECK_NULLPTR(sparse_values, KERNEL_STATUS_PARAM_INVALID,
                       "Sparse_values is null")
  Tensor *default_value_tensor = ctx.Input(3);
  KERNEL_CHECK_NULLPTR(default_value_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Default_value_tensor is null")
  Tensor *output_tensor = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Output_tensor is null")

  auto output_shape = shape_tensor->GetTensorShape();
  std::vector<int64_t> dense_shape;
  std::vector<int64_t> order;
  int64_t output_size = 1;
  for (int32_t index = 0; index < output_shape->GetDimSize(0); ++index) {
    if (shape_tensor->GetDataType() == DT_INT32) {
      int32_t *temp_dim = reinterpret_cast<int32_t *>(shape_tensor->GetData());
      dense_shape.emplace_back(static_cast<int64_t>(temp_dim[index]));
    } else {
      int64_t *temp_dim = reinterpret_cast<int64_t *>(shape_tensor->GetData());
      dense_shape.emplace_back(temp_dim[index]);
    }
    order.push_back(dense_shape[index]);
    output_size *= dense_shape[index];
  }

  std::iota(order.begin(), order.end(), 0);

  SparseTensor st;
  if (st.CreateSparseTensor(indices_tensor, sparse_values, dense_shape,
                            order) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Create sparse tensor failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  AttrValue *validate_indices = ctx.GetAttr("validate_indices");
  if (validate_indices == nullptr) {
    KERNEL_LOG_ERROR("Get attr:validate_indices failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (validate_indices->GetBool()) {
    if (st.IndicesValid(ctx) != KERNEL_STATUS_OK) {
      KERNEL_LOG_ERROR("Indices is valid.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  // set default value
  auto type_size =
      GetSizeByDataType(static_cast<DataType>(output_tensor->GetDataType()));
  if (type_size < 1) {
    KERNEL_LOG_ERROR("Don't support output tensor types");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  char *default_value_addr =
      reinterpret_cast<char *>(default_value_tensor->GetData());
  char *output_addr = reinterpret_cast<char *>(output_tensor->GetData());
  for (int index = 0; index < output_size; ++index) {
    int cpyRet = memcpy_s(output_addr + (index * type_size), type_size,
                          default_value_addr, type_size);
    if (cpyRet < 0) {
      KERNEL_LOG_ERROR("memcpy_s default value failed index [%d], type_size [%d].",
                       index, type_size);
      return KERNEL_STATUS_INNER_ERROR;
    }
  }
  if (SparseToDense(ctx, st, indices_tensor, output_tensor) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Sparse_to_dense excute failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(SPARSETODENSE, SparseToDenseCpuKernel);
}  // namespace aicpu
