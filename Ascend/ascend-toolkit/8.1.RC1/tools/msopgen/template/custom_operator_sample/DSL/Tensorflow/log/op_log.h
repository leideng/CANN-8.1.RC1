/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file op_log.h
 * \brief
 */
#ifndef GE_OP_LOG_H
#define GE_OP_LOG_H

#include <string>
#include <type_traits>
#include "graph/node.h"

#if !defined( __ANDROID__) && !defined(ANDROID)
#else
#include <util/Log.h>
#endif

#define OPPROTO_SUBMOD_NAME "OP_PROTO"

template <class T>
typename std::enable_if<std::is_same<std::string, typename std::decay<T>::type>::value, const char*>::type
get_cstr(const T& name) {
  return name.c_str();
}

template <class T>
typename std::enable_if<std::is_same<const char*, typename std::decay<T>::type>::value, const char*>::type
get_cstr(T name) {
  return name;
}

template <class T>
typename std::enable_if<std::is_same<char*, typename std::decay<T>::type>::value, const char*>::type
get_cstr(T name) {
  return name;
}

inline const std::string& get_op_info(const std::string& str) {
  return str;
}

inline const char* get_op_info(const char* str) {
  return str;
}

inline std::string get_op_info(const ge::NodePtr& node) {
  return node != nullptr ? node->GetType() + ":" + node->GetName() : "nil";
}

inline std::string get_op_info(const ge::OpDescPtr& node) {
  return node != nullptr ? node->GetType() + ":" + node->GetName() : "nil";
}

template <class T>
typename std::enable_if<std::is_same<std::string, typename std::decay<T>::type>::value, const char*>::type
get_op_name(const T& name) {
  return name.c_str();
}

template <class T>
constexpr bool is_ge_operator_type() {
  return std::is_base_of<ge::Operator, typename std::decay<T>::type>::value;
}

template <class T>
typename std::enable_if<is_ge_operator_type<T>(), std::string>::type get_op_info(const T& op) {
  ge::AscendString name;
  ge::AscendString type;
  auto get_name_ret = op.GetName(name);
  auto get_type_ret = op.GetOpType(type);
  std::string op_info = get_type_ret == ge::GRAPH_SUCCESS ? type.GetString() : "nil";
  op_info += ":";
  op_info += get_name_ret == ge::GRAPH_SUCCESS ? name.GetString() : "nil";
  return op_info;
}

template <class T>
typename std::enable_if<std::is_same<const char*, typename std::decay<T>::type>::value, const char*>::type
get_op_name(T name) {
  return name;
}

template <class T>
typename std::enable_if<std::is_same<char*, typename std::decay<T>::type>::value, const char*>::type
get_op_name(T name) {
  return name;
}

template <typename T>
std::string TbeGetName(const T& op) {
  ge::AscendString op_ascend_name;
  ge::graphStatus ret = op.GetName(op_ascend_name);
  if (ret != ge::GRAPH_SUCCESS) {
    std::string op_name = "None";
    return op_name;
  }
  return op_ascend_name.GetString();
}

#if !defined( __ANDROID__) && !defined(ANDROID)
#define OP_LOGI(opname, ...)
#define OP_LOGW(opname, ...)
#define OP_LOGE(opname, ...)
#define OP_LOGD(opname, ...)
#define OP_LOGE_WITHOUT_REPORT(opname, ...)
#define GE_OP_LOGI(opname, ...)
#define GE_OP_LOGW(opname, ...)
#define GE_OP_LOGE(opname, ...)
#define GE_OP_LOGD(opname, ...)
#define FUSION_PASS_LOGI(...)
#define FUSION_PASS_LOGW(...)
#define FUSION_PASS_LOGE(...)
#define FUSION_PASS_LOGD(...)
#else
#define OP_LOGI(opname, ...)
#define OP_LOGW(opname, ...)
#define OP_LOGE(opname, ...)
#define OP_LOGD(opname, ...)
#define OP_LOGE_WITHOUT_REPORT(opname, ...)
#define FUSION_PASS_LOGI(...)
#define FUSION_PASS_LOGW(...)
#define FUSION_PASS_LOGE(...)
#define FUSION_PASS_LOGD(...)
#endif

#if !defined( __ANDROID__) && !defined(ANDROID)
#else
#define D_OP_LOGI(opname, fmt, ...)
#define D_OP_LOGW(opname, fmt, ...)
#define D_OP_LOGE(opname, fmt, ...)
#define D_OP_LOGD(opname, fmt, ...)
#define D_FUSION_PASS_LOGI(fmt, ...)
#define D_FUSION_PASS_LOGW(fmt, ...)
#define D_FUSION_PASS_LOGE(fmt, ...)
#define D_FUSION_PASS_LOGD(fmt, ...)
#endif

#define OP_LOGE_IF(condition, return_value, op_name, fmt, ...)                                                 \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do {                                                                                                         \
    if (condition) {                                                                                           \
      OP_LOGE(get_op_name(op_name), fmt, ##__VA_ARGS__);                                                       \
      return return_value;                                                                                     \
    }                                                                                                          \
  } while (0)

#define OP_LOGW_IF(condition, op_name, fmt, ...)                                                               \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do {                                                                                                         \
    if (condition) {                                                                                           \
      OP_LOGE(get_op_name(op_name), fmt, ##__VA_ARGS__);                                                       \
    }                                                                                                          \
  } while (0)

#endif //GE_OP_LOG_H
