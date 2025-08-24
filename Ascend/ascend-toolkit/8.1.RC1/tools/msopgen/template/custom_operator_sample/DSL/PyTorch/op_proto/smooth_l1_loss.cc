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
#include "smooth_l1_loss.h"
#include "util/util.h"

namespace ge {
    // ----------------SmoothL1Loss-------------------
    IMPLEMT_COMMON_INFERFUNC(SmoothL1LossInferShape) {
      if (OneInOneOutDynamicInfer(op, "predict", {"loss"})) {
        return GRAPH_SUCCESS;
      }
      return GRAPH_FAILED;
    }
    COMMON_INFER_FUNC_REG(SmoothL1Loss, SmoothL1LossInferShape);
    // ----------------SmoothL1Loss END-------------------
}  // namespace ge