/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "kernel_register_data.h"

namespace gert {
namespace {
ge::graphStatus NullCreator(const ge::FastNode *node, KernelContext *context) {
  (void) node;
  (void) context;
  return ge::GRAPH_SUCCESS;
}
}  // namespace
KernelRegisterData::KernelRegisterData(const ge::char_t *kernel_type) : kernel_type_(kernel_type) {
  funcs_.outputs_creator = NullCreator;
  funcs_.trace_printer = nullptr;
  critical_section_ = "";
  funcs_.profiling_info_filler = nullptr;
  funcs_.data_dump_info_filler = nullptr;
  funcs_.exception_dump_info_filler = nullptr;
}
}  // namespace gert
