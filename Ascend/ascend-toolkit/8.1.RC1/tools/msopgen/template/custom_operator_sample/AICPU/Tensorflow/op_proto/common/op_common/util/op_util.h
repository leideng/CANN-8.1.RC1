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
 * \file op_util.h
 * \brief
 */

#ifndef OP_UTIL_H_
#define OP_UTIL_H_

#include <string>
#include <vector>
#include <sstream>
#include "runtime/continuous_vector.h"
#include "runtime/shape.h"
#include "runtime/tensor.h"
#include "types.h"

namespace opcommon {
template <typename T1, typename T2>
bool IsDimValid(const T1 shapeSize, const T2 dimValue) {
  const int64_t minimumNum = static_cast<int64_t>(shapeSize) * (-1);
  const int64_t maximumNum = static_cast<int64_t>(shapeSize) - 1;

  return static_cast<int64_t>(dimValue) >= minimumNum && static_cast<int64_t>(dimValue) <= maximumNum;
}
std::string ToString(const ge::DataType& type);
std::string ToString(const ge::Format& format);
std::string ToString(const gert::Shape& shape);
std::string ToString(const gert::Shape* shape);
std::string ToString(const std::vector<int64_t>& shape);
std::string ToString(const std::vector<gert::Shape>& shapes);
std::vector<int64_t> ToVector(const gert::Shape& shape);

template <typename T>
std::string DebugString(const std::vector<T>& v) {
  std::ostringstream oss;
  oss << "[";
  if (v.size() > 0) {
    for (size_t i = 0; i < v.size() - 1; ++i) {
      oss << v[i] << ", ";
    }
    oss << v[v.size() - 1];
  }
  oss << "]";
  return oss.str();
}

template <typename T>
std::string DebugString(const std::vector<std::pair<T, T>>& v) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << "(" << v[i].first << ", " <<v[i].second << ")";
  }
  oss << "]";
  return oss.str();
}

template <typename T>
std::vector<T> ToVector(const gert::TypedContinuousVector<T>& vec) {
  constexpr size_t vecSize = vec.GetSize();
  constexpr std::vector<T> vecT(vecSize, 0);

  for (size_t i = 0; i < vecSize; i++) {
    vecT[i] = *(vec.GetData() + i);
  }
  return vecT;
}

template <typename T>
std::string ToString(const gert::TypedContinuousVector<T>& vec) {
  return DebugString(ToVector(vec));
}

template <typename T>
std::string ToString(const gert::TypedContinuousVector<T>* vec) {
  return DebugString(ToVector(*vec));
}

template <typename T>
std::string ToString(const T* value, size_t size) {
  std::string r = "[";
  for (size_t i = 0; i < size; i++) {
    r = r + std::to_string(value[i]) + ", ";
  }
  r = r + "]";
  return r;
}

template <typename T>
std::string ConcatString(const T& arg) {
  std::ostringstream oss;
  oss << arg;
  return oss.str();
}

template <typename T, typename... Ts>
std::string ConcatString(const T& arg, const Ts& ... argLeft) {
  std::ostringstream oss;
  oss << arg;
  oss << ConcatString(argLeft...);
  return oss.str();
}
} // namespace opcommon

#endif // OP_UTIL_H_