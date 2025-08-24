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
 * \file reduce_all.cc
 * \brief
 */
#include "reduce_all.h"
#include <string>
#include <vector>
#include "util/util.h"
#include "op_log.h"
#include "error_util.h"
#include "graph/utils/node_utils.h"
#include <chrono>

namespace ge {
using std::string;
const static bool prof_switch = std::getenv("REDUCE_INFER_PROF") != nullptr;

// Obtains the value of the constant tensor.
static void GetAllConstValue(const Tensor& data, std::vector<int64_t>& const_vec, ge::DataType axisType) {
  const uint8_t* constData = data.GetData();
  if (axisType == ge::DT_INT32) {
    size_t size = data.GetSize() / sizeof(int32_t);

    for (size_t i = 0; i < size; ++i) {
      const_vec.push_back((int64_t)(*((int32_t*)constData + i)));
    }
  } else if (axisType == ge::DT_INT64) {
    size_t size = data.GetSize() / sizeof(int64_t);

    for (size_t i = 0; i < size; ++i) {
      const_vec.push_back((int64_t)(*((int64_t*)constData + i)));
    }
  }
}

static bool InferReduceShape(const ge::Operator& op, const string& input_name, const string& axis_name,
                             const string& keep_dims_name, ge::TensorDesc& result_desc) {
  // indicates that GE should process related attributes during online infer shape
  vector<string> input_infer_depends = {"axes"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);

  result_desc = op.GetInputDesc(input_name);
  auto shape = result_desc.GetShape();
  std::vector<int64_t> shapeVector = shape.GetDims();
  int64_t dim_num = shape.GetDimNum();

  if (shapeVector.size() == 0) {
    OP_LOGI(op.GetName().c_str(), "input shape vector size is 0, is scalar.");
    result_desc.SetShape({});
    result_desc.SetShapeRange({});
    return true;
  }

  if (shapeVector[0] == -2) {
    std::vector<int64_t> oShapeVector;
    oShapeVector.push_back(-2);
    Shape oShape(oShapeVector);
    result_desc.SetShape(oShape);
    result_desc.SetShapeRange({});
    return true;
  }

  std::vector<std::pair<int64_t, int64_t>> input_shape_range;
  op_desc->MutableInputDesc(input_name)->GetShapeRange(input_shape_range);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  MakeUpShapeRange(shapeVector, input_shape_range);
  if (input_shape_range.size() != (uint32_t)dim_num) {
    OP_LOGI(op.GetName().c_str(), "reset input shape range.");
    input_shape_range.clear();
    MakeUpShapeRange(shapeVector, input_shape_range);
  }

  bool keep_dims;
  if (GRAPH_SUCCESS != op.GetAttr(keep_dims_name, keep_dims)) {
    OP_LOGE(op.GetName().c_str(), "GetAttr of %s failed.", keep_dims_name.c_str());
    return false;
  }

  ge::TensorDesc axis_desc;
  axis_desc = op.GetInputDesc(axis_name);
  auto axis_shape = axis_desc.GetShape();
  auto axis_type = axis_desc.GetDataType();
  std::vector<int64_t> axis_shapeVector = axis_shape.GetDims();
  int64_t axis_dimNum = axis_shape.GetDimNum();

  if (!axis_shapeVector.empty() && axis_shapeVector[0] > dim_num) {
    OP_LOGE(op.GetName().c_str(), "The size of axisnode must be less than inputx dim_num.");
    return false;
  }

  if (axis_dimNum == 1 && axis_shapeVector[0] == 0) {
    result_desc.SetShape(shape);
    result_desc.SetShapeRange(input_shape_range);
    OP_LOGI(op.GetName().c_str(), "axis dim num is 1 and axis shape vector[0] is 0.");
    return true;
  }

  Tensor data;
  // axis unknown
  if (GRAPH_SUCCESS != op.GetInputConstData(axis_name, data)) {
    OP_LOGI(op.GetName().c_str(), "GetInputConstData of %s failed, enter axis unknown scenario.", axis_name.c_str());

    std::vector<int64_t> oShapeVector;

    if (axis_dimNum > 1) {
      OP_LOGE(op.GetName().c_str(), "The dim number of axis must be one or zero, but actual is %d.", axis_dimNum);
      return false;
    }

    if (keep_dims) {
      for (int64_t item = 0; item < dim_num; ++item) {
        int64_t range_min_value = 1;
        int64_t range_max_value = input_shape_range[item].second;
        output_shape_range.push_back(std::make_pair(range_min_value, range_max_value));

        if (range_max_value == 1) {
          oShapeVector.push_back(1);
        } else {
          oShapeVector.push_back(-1);
        }
      }

      Shape oShape(oShapeVector);
      result_desc.SetShape(oShape);
      result_desc.SetShapeRange(output_shape_range);
    } else {
      if (!axis_shapeVector.empty() && (axis_shapeVector[0] == -1 || axis_shapeVector[0] == -2)) {
        OP_LOGI(op.GetName().c_str(), "Can't get reduce axis number.");

        oShapeVector.push_back(-2);
        Shape oShape(oShapeVector);
        result_desc.SetShape(oShape);
        result_desc.SetShapeRange({});
      } else {
        int64_t output_dimNum = 0;
        if (axis_dimNum == 0) {
          output_dimNum = dim_num - 1;
        } else {
          output_dimNum = dim_num - axis_shapeVector[0];
        }
        OP_LOGI(op.GetName().c_str(), "Get output dim num %d.", output_dimNum);

        int64_t range_min_value = input_shape_range[0].first;
        int64_t range_max_value = input_shape_range[0].second;
        for (uint32_t item = 0; item < shapeVector.size(); ++item) {
          if (input_shape_range[item].first < range_min_value) {
            range_min_value = input_shape_range[item].first;
          }

          if (input_shape_range[item].second == -1) {
            range_max_value = -1;
          }
          if (range_max_value != -1 && input_shape_range[item].second > range_max_value) {
            range_max_value = input_shape_range[item].second;
          }
        }

        for (int64_t item = 0; item < output_dimNum; ++item) {
          oShapeVector.push_back(-1);
          output_shape_range.push_back(std::make_pair(range_min_value, range_max_value));
        }

        Shape oShape(oShapeVector);
        result_desc.SetShape(oShape);
        result_desc.SetShapeRange(output_shape_range);
      }
    }

    // axis known
  } else {
    std::vector<int64_t> axis{};
    size_t size = data.GetSize();
    if (size != 0) {
      GetAllConstValue(data, axis, axis_type);
    }

    // reduce axis is empty, reduce all
    if (axis.size() == 0) {
      for (size_t i = 0; i < shapeVector.size(); ++i) {
        axis.push_back(i);
      }
    }

    // convert reduce axis
    for (size_t i = 0; i < axis.size(); ++i) {
      if (axis[i] < -dim_num || axis[i] > (dim_num - 1)) {
        OP_LOGE(op.GetName().c_str(), "reduce verify failed, axis: %d, dim_num:%d.", axis[i], dim_num);
        return false;
      }
      if (axis[i] < 0) {
        axis[i] = dim_num + axis[i];
      }
    }

    std::vector<int64_t> oShapeVector;
    std::vector<int64_t>::iterator tmp;
    for (int64_t item = 0; item < dim_num; ++item) {
      tmp = std::find(axis.begin(), axis.end(), item);
      if (tmp != axis.end()) {
        // item in axis
        if (keep_dims) {
          // If keepDims is true, current dimesion set to 1
          oShapeVector.push_back(1);
          output_shape_range.push_back(std::make_pair(1, 1));
        }
      } else {
        // item is not in ConstValueAxis
        oShapeVector.push_back(shapeVector[item]);
        output_shape_range.push_back(input_shape_range[item]);
      }
    }

    // clear output shape range during static shape
    bool is_static_shape = true;
    for (uint32_t i = 0; i < shapeVector.size(); ++i) {
      if (shapeVector[i] == -1) {
        is_static_shape = false;
        break;
      }
    }
    if (is_static_shape) {
      output_shape_range.clear();
    }

    Shape oShape(oShapeVector);
    result_desc.SetShape(oShape);
    result_desc.SetShapeRange(output_shape_range);
  }

  return true;
}

static bool CheckReduceInfo(const ge::Operator& op, const size_t& input_size, const size_t& axis_size,
                            const string& keep_dims_name, bool& keep_dims) {
  if (GRAPH_SUCCESS != op.GetAttr(keep_dims_name, keep_dims)) {
    OP_LOGE(op.GetName().c_str(), "GetAttr of %s failed.", keep_dims_name.c_str());
    return false;
  }
  return true;
}

static bool CheckReduceDInfo(const ge::Operator& op, const size_t& input_size, const string& keep_dims_name,
                             const string& axis_name, bool& keep_dims, std::vector<int64_t>& axis) {
  if (GRAPH_SUCCESS != op.GetAttr(keep_dims_name, keep_dims)) {
    OP_LOGE(op.GetName().c_str(), "GetAttr of %s failed.", keep_dims_name.c_str());
    return false;
  }
  if (GRAPH_SUCCESS != op.GetAttr(axis_name, axis)) {
    OP_LOGE(op.GetName().c_str(), "GetAttr of %s failed.", axis_name.c_str());
    return false;
  }
  if (axis.size() > input_size) {
    OP_LOGE(op.GetName().c_str(), "size of axis is illegal.");
    return false;
  }

  return true;
}

template <typename T>
static void GetTensorValue(const GeTensor* data, std::vector<int64_t>& vec_dim) {
  int32_t size = data->GetData().GetSize() / sizeof(T);
  void* data_ptr = (void*)data->GetData().GetData();
  if (data_ptr == nullptr) {
    return;
  }
  for (int32_t i = 0; i < size; i++) {
    T dim = *((T*)data_ptr + i);
    vec_dim.push_back((int64_t)dim);
  }
}

static bool ConvertAxis(std::vector<int64_t>& axis, int64_t input_length) {
  // Convert reduce axis
  for (size_t i = 0; i < axis.size(); ++i) {
    if (axis[i] < -input_length || axis[i] > (input_length - 1)) {
      OP_LOGE("Op[Reduce]", "reduce verify failed, axis: %d, input_length: %d", axis[i], input_length);
      return false;
    }
    if (axis[i] < 0) {
      axis[i] = input_length + axis[i];
    }
  }
  // All Reduce
  if (axis.size() == 0) {
    for (size_t i = 0; i < (size_t)input_length; ++i) {
      axis.push_back(i);
    }
  }
  return true;
}

static void DoKnownBranch(const bool& keep_dims, const DataType& type, std::vector<int64_t>& input_shape,
                          std::vector<int64_t>& axis_value, GeTensorDescPtr& output_desc) {
  /* Work In Situations:
   * 1. runtime for dynamic
   * 2. const case for dynamic
   * 3. static case
   * Don't set range in the branch
   * */
  size_t length = input_shape.size();
  std::vector<int64_t> reduce_flag(length);
  for (auto item : axis_value) {
    reduce_flag[item] = 1;
  }

  std::vector<int64_t> output_shape(length);
  if (keep_dims) {
    for (size_t idx = 0; idx < length; ++idx) {
      output_shape[idx] = reduce_flag[idx] == 1 ? 1 : input_shape[idx];
    }
  } else {
    size_t i0 = 0;
    for (size_t idx = 0; idx < length; ++idx) {
      if (reduce_flag[idx] == 0) {
        output_shape[i0] = input_shape[idx];
        i0++;
      }
    }
    output_shape.resize(i0);
  }

  output_desc->SetShape(GeShape(output_shape));
  output_desc->SetDataType(type);
  return;
}

static void DoAxisKnown(const bool& keep_dims, const std::vector<int64_t>& axis,
                        const std::vector<int64_t>& input_shape,
                        const std::vector<std::pair<int64_t, int64_t>>& input_shape_range,
                        std::vector<int64_t>& output_shape,
                        std::vector<std::pair<int64_t, int64_t>>& output_shape_range) {
  size_t input_length = input_shape.size();
  if (keep_dims) {
    output_shape = input_shape;
    output_shape_range = input_shape_range;
    for (auto item : axis) {
      output_shape[item] = 1;
      output_shape_range[item] = std::make_pair<int64_t, int64_t>(1, 1);
    }
  } else {
    std::vector<int64_t> reduce_flag(input_length);
    for (auto item : axis) {
      reduce_flag[item] = 1;
    }
    for (size_t idx = 0; idx < input_length; ++idx) {
      if (reduce_flag[idx] == 0) {
        output_shape.push_back(input_shape[idx]);
        output_shape_range.push_back(input_shape_range[idx]);
      }
    }
  }

  return;
}

static void DoAxisUnKnown(const bool& keep_dims, const std::vector<int64_t>& axis_shape,
                          const std::vector<int64_t>& input_shape,
                          const std::vector<std::pair<int64_t, int64_t>>& input_shape_range,
                          std::vector<int64_t>& output_shape,
                          std::vector<std::pair<int64_t, int64_t>>& output_shape_range) {
  size_t input_length = input_shape.size();
  size_t axis_length = axis_shape.size();
  if (keep_dims) {
    for (size_t item = 0; item < input_length; ++item) {
      int64_t range_min_value = 0;
      int64_t range_max_value = input_shape_range[item].second;
      output_shape_range.push_back(std::make_pair(range_min_value, range_max_value));
      if (range_max_value == 1) {
        output_shape.push_back(1);
      } else {
        output_shape.push_back(-1);
      }
    }
  } else {
    int64_t output_dimNum = axis_length == 0 ? (int64_t)input_length - 1 : (int64_t)input_length - axis_shape[0];
    int64_t range_min_value = input_shape_range[0].first;
    int64_t range_max_value = input_shape_range[0].second;
    for (size_t item = 0; item < input_shape.size(); ++item) {
      if (input_shape_range[item].first < range_min_value) {
        range_min_value = input_shape_range[item].first;
      }

      if (input_shape_range[item].second == -1) {
        range_max_value = -1;
      }
      if (range_max_value != -1 && input_shape_range[item].second > range_max_value) {
        range_max_value = input_shape_range[item].second;
      }
    }

    for (int64_t item = 0; item < output_dimNum; ++item) {
      output_shape.push_back(-1);
      output_shape_range.push_back(std::make_pair(range_min_value, range_max_value));
    }
  }
  return;
}

static bool InferReduceShapeProcess(const ge::Operator& op, const string& input_name, const string& axis_name,
                                    const string& keep_dims_name) {
  // Get input|output|axis desc
  const vector<string> depends = {"axes"};
  PREPARE_DYNAMIC_SHAPE(depends);
  auto input_desc = op_desc->MutableInputDesc(input_name);
  auto axis_desc = op_desc->MutableInputDesc(axis_name);
  auto output_desc = op_desc->MutableOutputDesc("y");

  vector<int64_t> input_shape = input_desc->MutableShape().GetDims();
  vector<int64_t> axis_shape = axis_desc->MutableShape().GetDims();
  auto input_type = input_desc->GetDataType();
  auto axis_type = axis_desc->GetDataType();
  size_t input_length = input_shape.size();
  size_t axis_length = axis_shape.size();

  if (input_length == 0) {
    output_desc->SetShape({});
    output_desc->SetDataType(input_type);
    return true;
  }
  if (input_shape[0] == -2) {
    std::vector<int64_t> output_shape(1, -2);
    output_desc->SetShape(GeShape(output_shape));
    output_desc->SetDataType(input_type);
    return true;
  }
  if (!axis_shape.empty() && axis_shape[0] == 0) {
    OP_LOGD(op.GetName().c_str(), "axis_shape[0] is 0");
    output_desc->SetShape(GeShape(input_shape));
    output_desc->SetDataType(input_type);
    return true;
  }
  // Get const data
  auto axis_idx = static_cast<uint32_t>(op_desc->GetInputIndexByName(axis_name));
  const GeTensor *axis_tensor = OpDescUtils::GetInputConstData(op, axis_idx);
  std::vector<int64_t> axis;
  if (axis_tensor != nullptr) {
    if (axis_type == DT_INT32) {
      GetTensorValue<int32_t>(axis_tensor, axis);
    } else if (axis_type == DT_INT64) {
      GetTensorValue<int64_t>(axis_tensor, axis);
    } else {
      OP_LOGE(op.GetName().c_str(), "axis_type is illegal");
      return false;
    }
    // Convert "-1" -> "length-1";
    if (!ConvertAxis(axis, (int64_t)input_length)) {
      OP_LOGE(op.GetName().c_str(), "axis_value is illegal");
      return false;
    }
  } else {
    OP_LOGD(op.GetName().c_str(), "GetInputConstData Failed");
  }

  // Get attr
  bool keep_dims = false;
  if (!CheckReduceInfo(op, input_length, axis_length, keep_dims_name, keep_dims)) {
    OP_LOGE(op.GetName().c_str(), "Inputs and attrs are illegal");
    return false;
  }

  /* Main Process:
   * 1. Special Branch
   * 2. DoKnown Branch
   * 3. UnKnown Branch
   * */
  // Special Branch
  if (!axis_shape.empty() && (axis_shape[0] == -1 || axis_shape[0] == -2) && (!keep_dims)) {
    OP_LOGD(op.GetName().c_str(), "[Special Branch]: axis_shape[0] is -1 or -2.");
    std::vector<int64_t> output_shape;
    output_shape.push_back(-2);
    output_desc->SetShape(GeShape(output_shape));
    output_desc->SetDataType(input_type);
    return true;
  }

  // Special Branch for if axis has redundant axis value
  if (!axis_shape.empty() && axis_shape[0] > static_cast<int64_t>(input_length) && (!keep_dims)) {
    OP_LOGD(op.GetName().c_str(), "[Special Branch]: axis_shape[0] is more than input_length,"
            "if axis has redundant axis value");
    std::vector<int64_t> output_shape;
    output_shape.push_back(-2);
    output_desc->SetShape(GeShape(output_shape));
    output_desc->SetDataType(input_type);
    return true;
  }
  // DoKnown Branch && UnKnown Branch
  if ((!IsUnknown(input_shape)) && (!IsUnknown(axis_shape)) && (axis_tensor != nullptr)) {
    OP_LOGD(op.GetName().c_str(), "[DoKnown Branch]: shape and axis are known.");
    DoKnownBranch(keep_dims, input_type, input_shape, axis, output_desc);
  } else {
    OP_LOGD(op.GetName().c_str(), "[UnKnown Branch]: one of inputs is unknown at least.");
    std::vector<int64_t> output_shape;
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    std::vector<std::pair<int64_t, int64_t>> input_shape_range;
    input_desc->GetShapeRange(input_shape_range);
    // If InputShapeRange is None, MakeUpShapeRange will set range.
    MakeUpShapeRange(input_shape, input_shape_range);

    // Split as axis known and axis unknown
    if (axis_tensor != nullptr) {
      OP_LOGD(op.GetName().c_str(), "[UnKnown Branch]: axis is known.");
      DoAxisKnown(keep_dims, axis, input_shape, input_shape_range, output_shape, output_shape_range);
    } else {
      OP_LOGD(op.GetName().c_str(), "[UnKnown Branch]: axis is unknown.");
      DoAxisUnKnown(keep_dims, axis_shape, input_shape, input_shape_range, output_shape, output_shape_range);
    }

    output_desc->SetDataType(input_type);
    output_desc->SetShape(GeShape(output_shape));
    output_desc->SetShapeRange(output_shape_range);
  }

  return true;
}

static bool InferReduceDShapeProcess(const ge::Operator& op, const string& input_name, const string& axis_name,
                                     const string& keep_dims_name) {
  // Get input|output desc
  const vector<string> depends = {};
  PREPARE_DYNAMIC_SHAPE(depends);
  auto input_desc = op_desc->MutableInputDesc(input_name);
  auto output_desc = op_desc->MutableOutputDesc("y");

  vector<int64_t> input_shape = input_desc->MutableShape().GetDims();
  auto input_type = input_desc->GetDataType();
  size_t input_length = input_shape.size();

  /* Main Process:
   * 1. Special Branch
   * 2. DoKnown Branch
   * 3. UnKnown Branch
   * */
  // Special Branch
  if (input_length == 0) {
    OP_LOGD(op.GetName().c_str(), "[Special Branch]: input_shape size is 0.");
    output_desc->SetShape({});
    output_desc->SetDataType(input_type);
    return true;
  }
  if (input_shape[0] == -2) {
    OP_LOGD(op.GetName().c_str(), "[Special Branch]: input_shape is -2.");
    std::vector<int64_t> output_shape(1, -2);
    output_desc->SetShape(GeShape(output_shape));
    output_desc->SetDataType(input_type);
    return true;
  }

  // Get attr: axis and keep_dims
  bool keep_dims = false;
  std::vector<int64_t> axis;
  if (!CheckReduceDInfo(op, input_length, keep_dims_name, axis_name, keep_dims, axis)) {
    OP_LOGE(op.GetName().c_str(), "KeepDims or Axis is illegal");
    return false;
  }

  // Convert "-1" -> "length-1";
  if (!ConvertAxis(axis, (int64_t)input_length)) {
    OP_LOGE(op.GetName().c_str(), "axis_value is illegal");
    return false;
  }

  // DoKnown Branch and UnKnown Branch
  if (!IsUnknown(input_shape)) {
    OP_LOGD(op.GetName().c_str(), "[DoKnown Branch]: input_shape is known.");
    DoKnownBranch(keep_dims, input_type, input_shape, axis, output_desc);
  } else {
    OP_LOGD(op.GetName().c_str(), "[UnKnown Branch]: input_shape is unknown.");
    std::vector<int64_t> output_shape(input_length);
    std::vector<std::pair<int64_t, int64_t>> output_shape_range(input_length);
    std::vector<std::pair<int64_t, int64_t>> input_shape_range;
    input_desc->GetShapeRange(input_shape_range);
    // If InputShapeRange is None, MakeUpShapeRange will set range.
    MakeUpShapeRange(input_shape, input_shape_range);

    // MainProcess
    std::vector<int64_t> reduce_flag(input_length);
    for (auto item : axis) {
      reduce_flag[item] = 1;
    }

    if (keep_dims) {
      for (size_t idx = 0; idx < input_length; ++idx) {
        output_shape[idx] = reduce_flag[idx] == 1 ? 1 : input_shape[idx];
        output_shape_range[idx] =
            reduce_flag[idx] == 1 ? std::make_pair<int64_t, int64_t>(1, 1) : input_shape_range[idx];
      }
    } else {
      size_t i0 = 0;
      for (size_t idx = 0; idx < input_length; ++idx) {
        if (reduce_flag[idx] == 0) {
          output_shape[i0] = input_shape[idx];
          output_shape_range[idx] = input_shape_range[idx];
          i0++;
        }
      }
      output_shape.resize(i0);
      output_shape_range.resize(i0);
    }

    output_desc->SetShape(GeShape(output_shape));
    output_desc->SetDataType(input_type);
    output_desc->SetShapeRange(output_shape_range);
  }
  return true;
}

static bool InferReduceDShape(const ge::Operator& op, const string& input_name, const string& axis_name,
                              const string& keep_dims_name, ge::TensorDesc& result_desc) {
  result_desc = op.GetInputDesc(input_name);
  auto shape = result_desc.GetShape();
  std::vector<int64_t> shapeVector = shape.GetDims();
  int64_t dimNum = shape.GetDimNum();

  if (shapeVector.size() == 1 && shapeVector[0] == -2) {
    std::vector<int64_t> oShapeVector;
    oShapeVector.push_back(-2);
    Shape oShape(oShapeVector);
    result_desc.SetShape(oShape);
    return true;
  }

  std::vector<int64_t> axis;
  if (GRAPH_SUCCESS != op.GetAttr(axis_name, axis)) {
    OP_LOGE(op.GetName().c_str(), "GetAttr of %s failed.", axis_name.c_str());
    return false;
  }

  bool keep_dims;
  if (GRAPH_SUCCESS != op.GetAttr(keep_dims_name, keep_dims)) {
    OP_LOGE(op.GetName().c_str(), "GetAttr of %s failed.", keep_dims_name.c_str());
    return false;
  }

  if (axis.empty()) {
    for (size_t i = 0; i < shapeVector.size(); ++i) {
      axis.push_back(i);
    }
  }

  for (size_t i = 0; i < axis.size(); ++i) {
    if (axis[i] < -dimNum || axis[i] > (dimNum - 1)) {
      OP_LOGE(op.GetName().c_str(), "the axis of reduce verify failed.");
      return false;
    }
    if (axis[i] < 0) {
      axis[i] = dimNum + axis[i];
    }
  }

  // infer output shape range
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  std::vector<std::pair<int64_t, int64_t>> input_shape_range;
  op_desc->MutableInputDesc(input_name)->GetShapeRange(input_shape_range);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  MakeUpShapeRange(shapeVector, input_shape_range);
  if (input_shape_range.size() != (uint32_t)dimNum) {
    OP_LOGI(op.GetName().c_str(), "reset input shape range.");
    input_shape_range.clear();
    MakeUpShapeRange(shapeVector, input_shape_range);
  }

  std::vector<int64_t> oShapeVector;
  std::vector<int64_t>::iterator tmp;
  for (int64_t item = 0; item < dimNum; ++item) {
    tmp = std::find(axis.begin(), axis.end(), item);
    if (tmp != axis.end()) {
      // item in axis
      if (keep_dims) {
        // If keepDims is true, current dimesion set to 1
        oShapeVector.push_back(1);
        output_shape_range.push_back(std::make_pair(1, 1));
      }
    } else {
      // item is not in ConstValueAxis
      oShapeVector.push_back(shapeVector[item]);
      output_shape_range.push_back(input_shape_range[item]);
    }
  }

  // clear output shape range during static shape
  bool is_static_shape = true;
  for (uint32_t i = 0; i < shapeVector.size(); ++i) {
    if (shapeVector[i] == -1) {
      is_static_shape = false;
      break;
    }
  }
  if (is_static_shape) {
    output_shape_range.clear();
  }

  Shape oShape(oShapeVector);
  result_desc.SetShape(oShape);
  result_desc.SetShapeRange(output_shape_range);
  return true;
}

// ----------------ReduceAll Op-------------------
// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ReduceAllInferShape) {
  const vector<string> depend_names = {"axes"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  OP_LOGI(op.GetName().c_str(), "Enter ReduceAll proto inferfunction!");
  ge::TensorDesc result_desc;
  if (!InferReduceShape(op, "x", "axes", "keep_dims", result_desc)) {
    return GRAPH_FAILED;
  }
  auto shape = result_desc.GetShape();
  auto dtype = result_desc.GetDataType();
  std::vector<std::pair<int64_t, int64_t>> range;
  result_desc.GetShapeRange(range);

  // update output desc
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(shape);
  output_desc.SetDataType(dtype);
  if (range.size() > 0) {
    output_desc.SetShapeRange(range);
  }
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(ReduceAll, ReduceAllInferShape);
// ----------------ReduceAll END-------------------

}  // namespace ge