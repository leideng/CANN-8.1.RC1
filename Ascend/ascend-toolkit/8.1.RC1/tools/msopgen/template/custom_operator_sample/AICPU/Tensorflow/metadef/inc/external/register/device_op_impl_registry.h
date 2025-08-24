/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*!
 * \file device_op_impl_registry.h
 * \brief
 */

#ifndef REGISTER_DEVICE_OP_IMPL_REGISTRY_H
#define REGISTER_DEVICE_OP_IMPL_REGISTRY_H
#include <functional>
#include "graph/compiler_def.h"
#include "exe_graph/runtime/tiling_context.h"

namespace optiling {
using SinkTilingFunc = std::function<ge::graphStatus(gert::TilingContext *context)>;

class DeviceOpImplRegisterImpl;
class DeviceOpImplRegister {
public:
  DeviceOpImplRegister(const char *opType);
  ~DeviceOpImplRegister();
  DeviceOpImplRegister(DeviceOpImplRegister &&other) noexcept;
  DeviceOpImplRegister(const DeviceOpImplRegister &other);
  DeviceOpImplRegister &operator=(const DeviceOpImplRegister &) = delete;
  DeviceOpImplRegister &operator=(DeviceOpImplRegister &&) = delete;
  DeviceOpImplRegister &Tiling(SinkTilingFunc func);

private:
  std::unique_ptr<DeviceOpImplRegisterImpl> impl_;
};
}  // namespace optiling

#define DEVICE_IMPL_OP_OPTILING(optype)                                                                                \
  static optiling::DeviceOpImplRegister VAR_UNUSED g_deviceOpImplRegister##optype =                                    \
      optiling::DeviceOpImplRegister(#optype)
#endif