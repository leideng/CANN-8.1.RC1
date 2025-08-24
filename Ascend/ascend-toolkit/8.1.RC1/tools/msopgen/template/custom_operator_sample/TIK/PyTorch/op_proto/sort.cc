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
 * \file sort.cc
 * \brief
 */
#include "sort.h"
#include "util/util.h"


namespace ge {
// --------------------sort----------------------------
IMPLEMT_INFERFUNC(Sort, SortInferShape) {
      auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
      auto x_desc = op_desc->MutableInputDesc(0);
      std::vector<std::pair<int64_t, int64_t>> range;

      //out_sorted
      auto output_sorted_desc = op_desc->MutableOutputDesc(0);
      output_sorted_desc->SetShape(x_desc->GetShape());
      output_sorted_desc->SetDataType(x_desc->GetDataType());

      //out_indices
      auto output_indices_desc = op_desc->MutableOutputDesc(1);
      output_indices_desc->SetShape(x_desc->GetShape());
      output_indices_desc->SetDataType(DT_INT32);

      if(x_desc->GetShapeRange(range) == GRAPH_SUCCESS){
        output_sorted_desc->SetShapeRange(range);
        output_indices_desc->SetShapeRange(range);
      }
      return GRAPH_SUCCESS;
    }

    IMPLEMT_VERIFIER(Sort, SortVerify) {
      return GRAPH_SUCCESS;
    }

    INFER_FUNC_REG(Sort, SortInferShape);
    VERIFY_FUNC_REG(Sort, SortVerify);
    // --------------------sort---------------------------
}  // namespace ge
