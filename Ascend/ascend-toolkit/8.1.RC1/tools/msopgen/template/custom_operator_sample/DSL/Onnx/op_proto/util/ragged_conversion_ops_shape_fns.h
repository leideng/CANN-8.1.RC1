/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file ragged_conversion_ops_shape_fns.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_UTIL_RAGGED_CONVERSION_OPS_SHAPE_FNS_H_
#define OPS_BUILT_IN_OP_PROTO_UTIL_RAGGED_CONVERSION_OPS_SHAPE_FNS_H_

#include <vector>
#include "graph/operator.h"

namespace ge {

namespace {

enum class RowPartitionType { FIRST_DIM_SIZE, VALUE_ROWIDS, ROW_LENGTHS, ROW_SPLITS, ROW_LIMITS, ROW_STARTS };

typedef struct {
  int64_t size = 1;
  std::string name;
} Dim;

typedef struct {
  std::vector<Dim> dims;
  bool unknown_rank = false;
} TensorShape;

}  //  namespace

/**
 * make shape from shape proto
 * @param input_shape input tensor shape info
 * @param Shape output shape
 * @return multiply result
 */
graphStatus MakeShapeFromTensorShape(const TensorShape& input_shape, Shape& out, const char* op_name);

/**
 * check shape proto is valid shape or not
 * @param shape Tensor shape info
 * @return status whether infershape success
 */
graphStatus IsValidShape(const TensorShape& shape, const char* op_name);

/**
 * Combine ragged tensor to tensor shapes
 * @param ragged_rank ragged rank
 * @param shape shape proto
 * @param value_shape value shape info
 * @param output_shape output shape info
 * @return status whether infershape success
 */
graphStatus CombineRaggedTensorToTensorShapes(int32_t ragged_rank, const TensorShape& shape,
                                              const TensorShape& value_shape, TensorShape& output_shape,
                                              const char* op_name);

/**
 * Fills the output proto with the shape defined by the handle.
 * @param handle shape handle
 * @param shape_info tensor shape info
 * @return void
 */
void ShapeHandleToTensorShape(Shape handle, TensorShape& shape_info);

/**
 * infer RaggedTensorToTensor op shape
 * @param op Operator which need to infershape
 * @return status whether infershape success
 */
graphStatus RaggedTensorToTensorShapeFn(Operator& op);

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_UTIL_RAGGED_CONVERSION_OPS_SHAPE_FNS_H_
