/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
 * \file sigmoid_cross_entropy_with_logits_grad_v2.cc
 * \brief
 */
#include "sigmoid_cross_entropy_with_logits_grad_v2.h"
#include "util/util.h"

namespace ge {
// ----------------SigmoidCrossEntropyWithLogitsGradV2 Begin-------------------
IMPLEMT_VERIFIER(SigmoidCrossEntropyWithLogitsGradV2,
                 SigmoidCrossEntropyWithLogitsGradV2Verity) {
    std::vector<int64_t> predict_shape_dim =
        op.GetInputDesc("predict").GetShape().GetDims();
    std::vector<int64_t> target_shape_dim =
        op.GetInputDesc("target").GetShape().GetDims();
    for (size_t i = 0; i < predict_shape_dim.size(); i++) {
        if ((predict_shape_dim[i] != target_shape_dim[i])) {
            printf(op.GetName().c_str(),
                "the input shape of predict and target should be same");
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SigmoidCrossEntropyWithLogitsGradV2InferShape) {
    if (OneInOneOutDynamicInfer(op, "predict", {"gradient"})) {
        return GRAPH_SUCCESS;
    }
    return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(SigmoidCrossEntropyWithLogitsGradV2,
                      SigmoidCrossEntropyWithLogitsGradV2InferShape);
VERIFY_FUNC_REG(SigmoidCrossEntropyWithLogitsGradV2,
                SigmoidCrossEntropyWithLogitsGradV2Verity);
// ----------------SigmoidCrossEntropyWithLogitsGradV2 END---------------------
}  // namespace ge