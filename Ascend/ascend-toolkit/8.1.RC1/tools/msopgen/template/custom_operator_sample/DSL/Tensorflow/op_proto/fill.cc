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

/*!
 * \file fill.cc
 * \brief
 */
#include "fill.h"

#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"

#include "util/util.h"
#include "util/common_shape_fns.h"
#include "error_util.h"
#include "op_log.h"

namespace ge {
    // ----------------Fill Op Begin-------------------
    template <typename T>
    static void CaclDims(const GeTensor* data, std::vector<int64_t>& vec_dim) {
      int32_t size = data->GetData().GetSize() / sizeof(T);
      for (int32_t i = 0; i < size; i++) {
        void* data_ptr = (void*)data->GetData().GetData();
        if (data_ptr == nullptr) {
          return;
        }
        T dim = *((T*)data_ptr + i);
        vec_dim.push_back(dim);
      }
    }

    template <typename T>
    static void CaclDims(const Tensor& data, std::vector<int64_t>& vec_dim) {
      int32_t size = data.GetSize() / sizeof(T);
      for (int32_t i = 0; i < size; i++) {
        T dim = *((T*)data.GetData() + i);
        vec_dim.push_back(dim);
      }
    }

    IMPLEMT_COMMON_INFERFUNC(FillInferShape) {
      std::vector<int64_t> vec_dim;
      auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
      op_desc->SetOpInferDepends({"dims"});

      TensorDesc td = op.GetOutputDesc("y");

      auto dim_idx = static_cast<uint32_t>(op_desc->GetInputIndexByName("dims"));
      const GeTensor *data = OpDescUtils::GetInputConstData(op, dim_idx);
      if (data == nullptr) {
        GE_OP_LOGW(op.GetName().c_str(), "Get constValue failed of [dims]");
        auto shape = op.GetInputDesc("dims").GetShape();
        int64_t dim_value;
        dim_value = shape.GetDim(0);
        std::vector<std::pair<int64_t, int64_t>> range_output;
        for (int64_t m = 0; m < dim_value; m++) {
          vec_dim.push_back(-1);
          range_output.push_back(std::make_pair(1, -1));
        }
        if (vec_dim.empty()) {
          vec_dim.push_back(-2);
        }
        for (uint64_t i = 0; i < vec_dim.size(); i++) {
          OP_LOGD(op.GetName().c_str(), "fill no const infershape dims value [%d] is [%d]", i, vec_dim[i]);
        }
        OP_LOGD(op.GetName().c_str(), "fill no const infershape dims value done");
        td.SetShape(Shape(vec_dim));
        td.SetDataType(op.GetInputDesc("value").GetDataType());
        td.SetShapeRange(range_output);

        (void)op.UpdateOutputDesc("y", td);
        return GRAPH_SUCCESS;
      } else {
        DataType data_type = data->GetTensorDesc().GetDataType();
        std::vector<int64_t> vec_dim;
        if (data_type == DT_INT32) {
          CaclDims<int32_t>(data, vec_dim);
        } else if (data_type == DT_INT64) {
          CaclDims<int64_t>(data, vec_dim);
        } else {
          std::string err_msg = GetInputInvalidErrMsg("constValue");
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
          return GRAPH_PARAM_INVALID;
        }

        int64_t fused_output = std::accumulate(vec_dim.begin(), vec_dim.end(), 1, std::multiplies<int64_t>());
        OP_LOGD(op.GetName().c_str(), "fused_output dims value done [%d]", fused_output);
        std::vector<std::pair<int64_t, int64_t>> range_output;

        td.SetShape(Shape(vec_dim));
        td.SetDataType(op.GetInputDesc("value").GetDataType());

        for (auto& dim_val : vec_dim) {
          range_output.push_back(std::make_pair(dim_val, dim_val));
        }

        td.SetShapeRange(range_output);

        (void)op.UpdateOutputDesc("y", td);
        return GRAPH_SUCCESS;
      }
    }

    COMMON_INFER_FUNC_REG(Fill, FillInferShape);
    // ----------------Fill Op End-------------------
}  // namespace ge