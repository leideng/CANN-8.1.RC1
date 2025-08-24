/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "graph/utils/op_desc_utils_ex.h"

#include "common/ge_common/util.h"
#include "common/util/trace_manager/trace_manager.h"
#include "graph/normal_graph/operator_impl.h"
#include "graph/operator_factory_impl.h"
#include "graph/common_error_codes.h"
#include "graph/ge_context.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/transformer_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/utils/recover_ir_utils.h"
#include "common/util/mem_utils.h"
#include "common/checker.h"
#include "debug/ge_op_types.h"
#include "mmpa/mmpa_api.h"

namespace ge {
namespace {
std::function<ge::graphStatus(ge::Operator &)> TryGetV1InferFunc(const OpDescPtr &op_desc) {
  auto infer_func = op_desc->GetInferFunc();
  if (infer_func != nullptr) {
    return infer_func;
  }
  return OperatorFactoryImpl::GetInferShapeFunc(op_desc->GetType());
}
bool EnableIgnoreInferError() {
  static std::string ignore_infer_error_env_name = "IGNORE_INFER_ERROR";
  char env_value[MMPA_MAX_PATH];
  INT32 ret = mmGetEnv(ignore_infer_error_env_name.c_str(), env_value, MMPA_MAX_PATH);
  if (ret != EN_OK) {
    GELOGD("Can not get env [%s]. Disable ignore infer validation.", ignore_infer_error_env_name.c_str());
    return false;
  }

  std::string env_str_value = std::string(env_value);
  GELOGI("Got value of env[%s] is [%s].", ignore_infer_error_env_name.c_str(), env_str_value.c_str());
  return !env_str_value.empty();
}
}

graphStatus OpDescUtilsEx::CallInferFuncV2(const OpDescPtr &op_desc, Operator &op) {
  const auto call_infer_data_type = OperatorFactoryImpl::GetInferDataTypeFunc();
  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  if ((call_infer_data_type == nullptr) || (call_infer_shape_v2 == nullptr) || (call_infer_shape_range == nullptr)) {
    GELOGW("[Call][InferFuncV2] Node %s(%s) has no infer func v2 either v1. Please check op proto to make sure at "
           "least has one.",
           op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return GRAPH_FAILED;
  }
  if (op_desc->GetIrInputs().empty() && op_desc->GetIrOutputs().empty() && op_desc->GetAllOutputsDescSize() != 0U) {
    GE_CHK_STATUS_RET(RecoverIrUtils::RecoverOpDescIrDefinition(op_desc), "Failed recover ir def for %s %s",
                      op_desc->GetNamePtr(), op_desc->GetTypePtr());
  }
  GE_WARN_ASSERT_GRAPH_SUCCESS(call_infer_data_type(op_desc),
                               "[Call][InferFuncV2]Failed to infer data_type of node %s[%s].", op_desc->GetNamePtr(),
                               op_desc->GetTypePtr());
  GE_WARN_ASSERT_GRAPH_SUCCESS(call_infer_shape_v2(op, op_desc),
                               "[Call][InferFuncV2]Failed to infer shape of node %s[%s].", op_desc->GetNamePtr(),
                               op_desc->GetTypePtr());
  GE_WARN_ASSERT_GRAPH_SUCCESS(call_infer_shape_range(op, op_desc),
                               "[Call][InferFuncV2]Failed to infer shape_range of node %s[%s].", op_desc->GetNamePtr(),
                               op_desc->GetTypePtr());
  return GRAPH_SUCCESS;
}

graphStatus OpDescUtilsEx::CallInferFuncV1(const OpDescPtr &op_desc, Operator &op) {
  NodeShapeTransUtils transformer(op_desc);
  const auto is_init_success = transformer.Init();
  if (!is_init_success) {
    GELOGE(GRAPH_FAILED, "[Call][Init] for transformer failed");
    return GRAPH_FAILED;
  }
  if (!transformer.CatchFormatAndShape()) {
    GELOGE(GRAPH_FAILED, "[Call][CatchFormatAndShape] for transformer failed!");
    return GRAPH_FAILED;
  }
  graphStatus graph_status = GRAPH_SUCCESS;
  {
    const auto &node_ptr = NodeUtilsEx::GetNodeFromOperator(op);
    const bool empty_name = (node_ptr == nullptr) || (node_ptr->GetOwnerComputeGraph() == nullptr);
    const auto &graph_name = empty_name ? std::string("")
                                        : node_ptr->GetOwnerComputeGraph()->GetName();
    TraceOwnerGuard guard("OP", op_desc->GetName() + ":infershape", graph_name);
    auto infer_func = op_desc->GetInferFunc();
    graph_status = infer_func(op);
  }
  if ((graph_status != GRAPH_SUCCESS) && (graph_status != GRAPH_NODE_NEED_REPASS)) {
    GELOGE(GRAPH_FAILED, "[Call][InferFuncV1] for %s(%s) failed. ret:%u", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
           graph_status);
    return GRAPH_FAILED;
  }
  if (!transformer.UpdateFormatAndShape()) {
    GELOGE(GRAPH_FAILED, "[Call][UpdateFormatAndShape] for transformer failed!");
    return GRAPH_FAILED;
  }
  return graph_status;
}

graphStatus OpDescUtilsEx::CallInferFunc(const OpDescPtr &op_desc, Operator &op) {
  GE_CHECK_NOTNULL(op_desc, ", Op is null for Infer Shape.");
  graphStatus ret;
  // NoOp算子有原型，无infershape, 无数据输出, 此类需要跳过infer
  const bool has_io = (op_desc->GetInputsSize() != 0U || op_desc->GetOutputsSize() != 0U);
  const bool need_infer =
      (has_io && (OperatorFactory::IsExistOp(op_desc->GetTypePtr()) || (op_desc->GetInferFunc() != nullptr)));
  if (!need_infer) {
    // todo 这是一个特殊错误码，早期版本与接口调用方约定的错误码，调用方认为这不是个错误，并且会依据该错误码作额外工作
    // 如，映射到已有的IR上等行为，或无痛地跳过infershape（如netoutput/framework
    // 这里暂时为了v1动态shape执行时保留该错误码，后续整改
    GELOGD("Node %s(%s) no io or no prototype so does not need infer.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    ret = GRAPH_PARAM_INVALID;
  } else {
    // priority of use infer func v1
    // when v2 func is ready, remove v1 func, it will automatically follow the V2 process
    auto infer_func = TryGetV1InferFunc(op_desc);
    bool can_support_rt1 = (infer_func != nullptr);
    GELOGD("Op %s[%s] Call InferShapeFuncV%s", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
           can_support_rt1 ? "1" : "2");
    if (can_support_rt1) {
      op_desc->AddInferFunc(infer_func);
      ret = CallInferFuncV1(op_desc, op);
    } else {
      ret = CallInferFuncV2(op_desc, op);
      // 临时方案，避免客户自定义算子交付件不完备，为了快速恢复用例，提供临时环境变量
      // 后续引导客户正确补充交付件以后，删除该环境变量
      static bool enable_fast_ignore_infer_error = EnableIgnoreInferError();
      if (enable_fast_ignore_infer_error) {
        ret = (ret == GRAPH_SUCCESS) ? GRAPH_SUCCESS : GRAPH_PARAM_INVALID;
      } else if (ret != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR(
            "EZ9999",
            "Call InferShapeAndType for node:%s(%s) failed. You can ignore this validation by exporting IGNORE_INFER_ERROR=1 if necessary, but it is highly recommended to fix this problem.",
            op_desc->GetNamePtr(), op_desc->GetTypePtr());
      }
    }
  }

  if (ret == GRAPH_SUCCESS) {
    GE_ASSERT_SUCCESS(InferShapeByOutputShapesAttr(op_desc), "[Infer][ByShapeValue] failed, op = %s",
                      op_desc->GetNamePtr());
  }
  return ret;
}

graphStatus OpDescUtilsEx::InferShapeByOutputShapesAttr(const OpDescPtr &op_desc) {
  std::vector<std::vector<int64_t>> shape_values;
  const bool got = ge::AttrUtils::GetListListInt(op_desc, ATTR_NAME_PRESET_OUTPUT_SHAPES, shape_values);
  if (!got) {
    GELOGD("Do not need infer op = %s by shape value, got = %d, shape_values = %zu.",
           op_desc->GetNamePtr(), got, shape_values.size());
    return GRAPH_SUCCESS;
  }
  GE_ASSERT_TRUE(op_desc->GetAllOutputsDescSize() == static_cast<uint32_t>(shape_values.size()),
                 "op = %s has output size = %u, but shape values size = %zu.", op_desc->GetNamePtr(),
                 op_desc->GetAllOutputsDescSize(), shape_values.size());
  size_t output_idx = 0UL;
  for (const auto &shape_value : shape_values) {
    const auto &output_desc = op_desc->MutableOutputDesc(output_idx);
    GE_ASSERT_NOTNULL(output_desc, "[Get][Output] failed, id = %zu, op = %s.", output_idx, op_desc->GetNamePtr());
    output_idx++;
    const auto output_shape = GeShape(shape_value);
    GE_ASSERT_TRUE(TensorUtils::IsShapeEqual(output_desc->GetShape(), output_shape),
                   "[Check][ShapeEqual] op = %s inferred shape is %s, but shape value set shape is %s, is not same.",
                   op_desc->GetNamePtr(), output_desc->GetShape().ToString().c_str(), output_shape.ToString().c_str());
    output_desc->SetShape(output_shape);
    output_desc->SetOriginShape(output_shape);
    GELOGD("Update op = %s output[%zu] shape = %s", op_desc->GetNamePtr(), output_idx,
           ToString(output_shape.GetDims()).c_str());
  }
  return GRAPH_SUCCESS;
}

graphStatus OpDescUtilsEx::CallInferFormatFunc(const OpDescPtr &op_desc, Operator &op) {
  GE_CHECK_NOTNULL(op_desc, ", Op is null for Infer Format.");
  auto infer_format_func = op_desc->GetInferFormatFunc();
  if (infer_format_func != nullptr) {
    return static_cast<graphStatus>(infer_format_func(op));
  }
  infer_format_func = OperatorFactoryImpl::GetInferFormatFunc(op_desc->GetType());
  if (infer_format_func == nullptr) {
    return op_desc->DefaultInferFormat();
  }
  op_desc->AddInferFormatFunc(infer_format_func);
  return infer_format_func(op);
}

graphStatus OpDescUtilsEx::CallInferValueRangeFunc(const OpDescPtr &op_desc, Operator &op) {
  GE_CHECK_NOTNULL(op_desc, ", Op is null for Infer ValueRange.");
  auto infer_value_range_func = op_desc->GetInferValueRangeFunc();
  if (infer_value_range_func != nullptr) {
    return static_cast<graphStatus>(infer_value_range_func(op));
  }

  const InferValueRangePara infer_value_range_param = OperatorFactoryImpl::GetInferValueRangePara(op_desc->GetType());
  if (!infer_value_range_param.is_initialized) {
    REPORT_CALL_ERROR("E18888", "Node %s does not register func to infer value range.", op_desc->GetName().c_str());
    GELOGE(GRAPH_PARAM_INVALID, "Node %s does not register func to infer value range.", op_desc->GetName().c_str());
    return GRAPH_PARAM_INVALID;
  }

  infer_value_range_func = infer_value_range_param.infer_value_func;
  if (infer_value_range_func == nullptr) {
    REPORT_CALL_ERROR("E18888", "Value range infer func of node %s has been registered, but infer func is nullptr.",
                      op_desc->GetName().c_str());
    GELOGE(GRAPH_PARAM_INVALID, "Value range infer func of node %s has been registered, but infer func is nullptr.",
           op_desc->GetName().c_str());
    return GRAPH_PARAM_INVALID;
  }
  op_desc->AddInferValueRangeFunc(infer_value_range_func);
  return infer_value_range_func(op);
}

graphStatus OpDescUtilsEx::OpVerify(const OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(op_desc, ", Op is null for Infer Verify.");
  auto verify_func = op_desc->GetVerifyFunc();
  if (verify_func == nullptr) {
    verify_func = OperatorFactoryImpl::GetVerifyFunc(op_desc->GetType());
  }
  if (verify_func != nullptr) {
    Operator op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    const graphStatus ret = static_cast<graphStatus>(verify_func(op));
    op_desc->AddVerifierFunc(verify_func);
    op.BreakConnect();
    return ret;
  }
  return GRAPH_SUCCESS;
}

graphStatus OpDescUtilsEx::InferShapeAndType(const OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(op_desc, ", Op is null for Infer Shape.");
  auto infer_func = op_desc->GetInferFunc();
  if (infer_func == nullptr) {
    infer_func = OperatorFactoryImpl::GetInferShapeFunc(op_desc->GetType());
    if (infer_func == nullptr) {
      GELOGW("[InferShape][Check] %s does not have infer_func.", op_desc->GetName().c_str());
      /// The infer_func has not been added for each operator in the current operator information library.
      /// No infer_func added operator skips the call
      /// and directly uses the shape information passed down by the upper framework
      return GRAPH_SUCCESS;
    }
  }
  Operator op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  const graphStatus ret = static_cast<graphStatus>(infer_func(op));
  op_desc->AddInferFunc(infer_func);
  op.BreakConnect();
  return ret;
}

graphStatus OpDescUtilsEx::InferDataSlice(const OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(op_desc, ", Op is null for Infer Slice.");
  auto infer_data_slice_func = op_desc->GetInferDataSliceFunc();
  if (infer_data_slice_func == nullptr) {
    infer_data_slice_func = OperatorFactoryImpl::GetInferDataSliceFunc(op_desc->GetType());
    if (infer_data_slice_func == nullptr) {
      GELOGW("[InferDataSlice][Check] %s does not have infer data slice func.", op_desc->GetName().c_str());
      return NO_DEPENDENCE_FUNC;
    }
  }
  Operator op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  const graphStatus ret = static_cast<graphStatus>(infer_data_slice_func(op));
  op_desc->AddInferDataSliceFunc(infer_data_slice_func);
  op.BreakConnect();
  return ret;
}

void OpDescUtilsEx::SetType(OpDescPtr &op_desc, const std::string &type) {
  // If the type changes, IR related variables should be modified accordingly
  auto op = OperatorFactory::CreateOperator("tmp", type.c_str());
  op.BreakConnect();

  op_desc->SetType(type);
  op_desc->SetIrRelated(OpDescUtils::GetOpDescFromOperator(op));
  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(),
                   op_desc->GetName(), "type", "", "", type);
}

void OpDescUtilsEx::ResetFuncHandle(OpDescPtr &op_desc) {
  op_desc->AddInferFunc(nullptr);
  op_desc->AddInferFormatFunc(nullptr);
  op_desc->AddInferValueRangeFunc(nullptr);
  op_desc->AddVerifierFunc(nullptr);
  op_desc->AddInferDataSliceFunc(nullptr);
}

void OpDescUtilsEx::SetTypeAndResetFuncHandle(OpDescPtr &op_desc, const std::string &type) {
  SetType(op_desc, type);
  ResetFuncHandle(op_desc);
}

void OpDescUtilsEx::UpdateShapeAndDType(const GeTensorDescPtr &src, const GeTensorDescPtr &dst) {
  dst->SetOriginShape(src->GetOriginShape());
  dst->SetShape(src->GetShape());
  dst->SetDataType(src->GetDataType());
  dst->SetOriginDataType(src->GetOriginDataType());
  std::vector<std::pair<int64_t, int64_t>> src_shape_range;
  src->GetShapeRange(src_shape_range);
  dst->SetShapeRange(src_shape_range);
  dst->SetOriginShapeRange(src_shape_range);
  ge::TensorUtils::SetRealDimCnt(*dst, static_cast<uint32_t>(src->GetShape().GetDims().size()));
}
} // namespace ge
