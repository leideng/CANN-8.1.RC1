/*
 * Copyright (C)  2020. Huawei Technologies Co., Ltd. All rights reserved.
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
 * \file decode_bbox_v2.cc
 * \brief
 */
#include "decode_bbox_v2.h"
#include <cmath>
#include <string>
#include <vector>

#define CHECK(cond, result_expr) \
    do {                         \
        if (cond) {              \
            result_expr;         \
        }                        \
    } while (0)

namespace ge {
// ----------------DecodeBboxV2-------------------
IMPLEMT_VERIFIER(DecodeBboxV2, DecodeBboxV2Verify) {
    // check shape
    auto box_predictions_shape = op.GetInputDesc("boxes").GetShape().GetDims();
    CHECK(box_predictions_shape.empty(), return GRAPH_FAILED);
    auto anchors_shape = op.GetInputDesc("anchors").GetShape().GetDims();
    CHECK(anchors_shape.empty(), return GRAPH_FAILED);

    // check attr decode_clip
    float decode_clip = 0;
    CHECK(ge::GRAPH_SUCCESS != op.GetAttr("decode_clip", decode_clip),
          return GRAPH_FAILED);
    CHECK(decode_clip > 10 || decode_clip < 0,
          return GRAPH_FAILED);

    // check attr reversed_box
    bool reversed_box = false;
    CHECK(ge::GRAPH_SUCCESS != op.GetAttr("reversed_box", reversed_box),
          return GRAPH_FAILED);

    // check attr scales
    std::vector<float> scales_list;
    CHECK(ge::GRAPH_SUCCESS != op.GetAttr("scales", scales_list),
          return GRAPH_FAILED);

    CHECK(scales_list.size() != 4, return GRAPH_FAILED);
    // check shape
    int64_t box_predictions_shape_n = 1;
    for (uint32_t i = 0; i < box_predictions_shape.size(); i++) {
        box_predictions_shape_n = box_predictions_shape_n * box_predictions_shape[i];
    }
    int64_t anchors_shape_n = 1;
    for (uint32_t i = 0; i < anchors_shape.size(); i++) {
        anchors_shape_n = anchors_shape_n * anchors_shape[i];
    }
    CHECK(box_predictions_shape_n != anchors_shape_n, return GRAPH_FAILED);
    int64_t box_predictions_shape_D = box_predictions_shape[box_predictions_shape.size() - 1];
    int64_t box_predictions_shape_N = box_predictions_shape[0];
    int64_t anchors_shape_D = anchors_shape[anchors_shape.size() - 1];
    int64_t anchors_shape_N = anchors_shape[0];
    if (reversed_box == false) {
        CHECK(box_predictions_shape_D != 4 || anchors_shape_D != 4,
              return GRAPH_FAILED);
    }
    if (reversed_box == true) {
        CHECK(box_predictions_shape_N != 4 || anchors_shape_N != 4,
              return GRAPH_FAILED);
    }

    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DecodeBboxV2, ELMTWISE_INFER_SHAPEANDTYPE("boxes", "y"));

// Registered verify function
VERIFY_FUNC_REG(DecodeBboxV2, DecodeBboxV2Verify);
}  // namespace ge
