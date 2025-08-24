/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_GRAPH_RUNTIME_INFERENCE_CONTEXT_H_
#define INC_GRAPH_RUNTIME_INFERENCE_CONTEXT_H_

#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include "external/graph/ge_error_codes.h"
#include "external/graph/tensor.h"
#include "ge_attr_value.h"

namespace ge {
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY RuntimeInferenceContext {
 public:
  graphStatus SetTensor(int64_t node_id, int32_t output_id, GeTensorPtr tensor);
  graphStatus GetTensor(const int64_t node_id, int32_t output_id, GeTensorPtr &tensor) const;
  void Release();

 private:
  std::map<int64_t, std::vector<GeTensorPtr>> ge_tensors_;
  mutable std::mutex mu_;
};
} // namespace ge

#endif // INC_GRAPH_RUNTIME_INFERENCE_CONTEXT_H_
