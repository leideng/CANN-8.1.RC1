/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_REGISTER_STREAM_MANAGE_FUNC_REGISTRY_H
#define INC_REGISTER_STREAM_MANAGE_FUNC_REGISTRY_H

#include <map>
#include <mutex>
#include "runtime/rt.h"
#include "common/ge_common/debug/ge_log.h"

namespace ge {
// acl action types
enum class MngActionType : uint32_t {
  DESTROY_STREAM,
  DESTROY_CONTEXT,
  RESET_DEVICE,
};

typedef union {
  rtStream_t stream;
  rtContext_t context;
  int32_t device_id;
} MngResourceHandle;

enum class StreamMngFuncType : uint32_t {
  ACLNN_STREAM_CALLBACK,  // aclnn callback function for destroying sub-stream
};

using StreamMngFunc = uint32_t (*)(MngActionType action_type, MngResourceHandle handle);

class StreamMngFuncRegistry {
 public:
  static StreamMngFuncRegistry &GetInstance();
  Status TryCallStreamMngFunc(const StreamMngFuncType func_type, MngActionType action_type, MngResourceHandle handle);
  void Register(const StreamMngFuncType func_type, StreamMngFunc const manage_func);
  StreamMngFunc LookUpStreamMngFunc(const StreamMngFuncType func_type);

  StreamMngFuncRegistry(const StreamMngFuncRegistry &other) = delete;
  StreamMngFuncRegistry &operator=(const StreamMngFuncRegistry &other) = delete;

 private:
  StreamMngFuncRegistry() = default;
  ~StreamMngFuncRegistry() = default;

  std::mutex mutex_;
  std::map<StreamMngFuncType, StreamMngFunc> type_to_func_;
};

class StreamMngFuncRegister {
 public:
  StreamMngFuncRegister(const StreamMngFuncType func_type, StreamMngFunc const manage_func);
};
}  // namespace ge

#ifdef __GNUC__
#define ATTRIBUTE_USED __attribute__((used))
#else
#define ATTRIBUTE_USED
#endif
#define REG_STREAM_MNG_FUNC(type, func) REG_STREAM_MNG_FUNC_UNIQ_HELPER(type, func, __COUNTER__)
#define REG_STREAM_MNG_FUNC_UNIQ_HELPER(type, func, counter) REG_STREAM_MNG_FUNC_UNIQ(type, func, counter)
#define REG_STREAM_MNG_FUNC_UNIQ(type, func, counter)                                                                  \
  static ::ge::StreamMngFuncRegister register_stream_mng_func_##counter ATTRIBUTE_USED =                               \
      ge::StreamMngFuncRegister(type, func)

#endif  // INC_REGISTER_STREAM_MANAGE_FUNC_REGISTRY_H
