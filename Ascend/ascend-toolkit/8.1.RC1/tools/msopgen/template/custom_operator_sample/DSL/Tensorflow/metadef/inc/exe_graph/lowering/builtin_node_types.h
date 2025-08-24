/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef METADEF_CXX_INC_EXE_GRAPH_LOWERING_BUILTIN_NODE_TYPES_H_
#define METADEF_CXX_INC_EXE_GRAPH_LOWERING_BUILTIN_NODE_TYPES_H_
#include <cstring>
#include "graph/types.h"

namespace gert {
// 子图的输出，InnerNetOutput有多个输入，没有输出，每个InnerNetOutput的输入，对应相同Index的parent node的输出
constexpr ge::char_t const *kInnerNetOutput = "InnerNetOutput";

// 子图的输入，InnerData没有输入，有一个个输出，代表其对应的parent node的输入。
// InnerData具有一个key为index的属性，该属性类型是int32，代表该InnerData对应的parent node输入的index
constexpr ge::char_t const *kInnerData = "InnerData";

// 图的输出，NetOutput出现在Main图上，表达执行完成后，图的输出。NetOutput有多个输入，没有输出，每个输入对应相同Index的图输出
constexpr ge::char_t const *kNetOutput = "NetOutput";

// 图的输入，Data没有输入，有一个个输出，代表图的输入。
// Data具有一个key为index的属性，该属性类型是int32，代表图的输入的Index
constexpr ge::char_t const *kData = "Data";

// 图的输出，未来NetOutput的会被OutputData所代替
// OutputData只在Main图上出现，执行完成后，图的输出会被写入到OutputData
// OutputData没有输入，有多个输出，每个输出对应相同Index的图输出
constexpr ge::char_t const *kOutputData = "OutputData";

// 常量节点，该节点没有输入，有一个输出，代表常量的值
// 常量节点有一个属性"value"代表该常量节点的值，value是一段二进制，常量节点本身不关注其内容的格式
constexpr ge::char_t const *kConst = "Const";

// 常量输入节点，该节点没有输入，有一个输出，
// 其值在lowering阶段不可获得，加载时由外部传入，且在执行过程中不会被改变
// ConstData具有一个key为type的属性，该属性类型是int32，由此代表ConstData的类型，也代表顺序
// 详见air仓ConstDataType枚举
constexpr ge::char_t const *kConstData = "ConstData";

inline bool IsTypeData(const ge::char_t *const node_type) {
  return strcmp(kData, node_type) == 0;
}
inline bool IsTypeInnerData(const ge::char_t *const node_type) {
  return strcmp(kInnerData, node_type) == 0;
}
inline bool IsTypeInnerNetOutput(const ge::char_t *const node_type) {
  return strcmp(kInnerNetOutput, node_type) == 0;
}
inline bool IsTypeNetOutput(const ge::char_t *const node_type) {
  return strcmp(kNetOutput, node_type) == 0;
}
inline bool IsTypeConst(const ge::char_t *const node_type) {
  return strcmp(kConst, node_type) == 0;
}
inline bool IsTypeConstData(const ge::char_t *const node_type) {
  return strcmp(kConstData, node_type) == 0;
}
inline bool IsTypeOutputData(const ge::char_t *const node_type) {
  return strcmp(kOutputData, node_type) == 0;
}
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_LOWERING_BUILTIN_NODE_TYPES_H_
