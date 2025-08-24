/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file op_util.cc
 * \brief util function of op
 */
#include <graph/utils/type_utils.h>
#include "op_util.h"
#include "error_util.h"

namespace ops {
using namespace std;

/*
 * @brief: get datatype string from enum
 * @param [in] type: enum datatype
 * @return string: datatype string
 */
std::string ToString(const ge::DataType& type) {
  return ge::TypeUtils::DataTypeToSerialString(type);
}

/*
 * @brief: get format string from enum
 * @param [in] format: enum format
 * @return string: format string
 */
std::string ToString(const ge::Format& format) {
  return ge::TypeUtils::FormatToSerialString(format);
}

std::vector<int64_t> ToVector(const gert::Shape& shape) {
  size_t shape_size = shape.GetDimNum();
  std::vector<int64_t> shape_vec(shape_size, 0);

  for (size_t i = 0; i < shape_size; i++) {
    shape_vec[i] = shape.GetDim(i);
  }
  return shape_vec;
}

std::string ToString(const gert::Shape& shape) {
  return ge::DebugString(ToVector(shape));
}

std::string ToString(const gert::Shape* shape) {
  return ge::DebugString(ToVector(*shape));
}

std::string ToString(const std::vector<int64_t>& shape) {
  return ge::DebugString(shape);
}

std::string ToString(const std::vector<gert::Shape>& shapes) {
  string str = "[";
  for (gert::Shape shape : shapes) {
    str += ToString(shape);
    str += ", ";
  }
  str += "]";
  return str;
}

// round down
int64_t GetFloorDiv(int64_t u_value, int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }
  res_value = u_value / d_value;
  return res_value;
}

// round up
int64_t GetCeilDiv(int64_t u_value, int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }
  res_value = (u_value + d_value - 1) / d_value;
  return res_value;
}

int64_t GetDivisorAlign(const int64_t u_value, const int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }

  res_value = u_value / d_value * d_value;

  return res_value;
}

int64_t GetMod(const int64_t l_value, const int64_t r_value) {
  if (r_value == 0) {
    return l_value;
  }
  return l_value % r_value;
}
}  // namespace ops
