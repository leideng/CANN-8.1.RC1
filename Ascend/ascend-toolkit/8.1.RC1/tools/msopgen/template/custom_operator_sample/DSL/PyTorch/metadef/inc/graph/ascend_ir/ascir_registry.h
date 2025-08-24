/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef AUTOFUSE_ASCIR_REGISTRY_H
#define AUTOFUSE_ASCIR_REGISTRY_H
#include <functional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <algorithm>
#include "ascend_ir/ascend_ir_check.h"
#include "external/graph/types.h"
#include "op_desc.h"
#include "ir/ir_data_type_symbol_store.h"
namespace ge {
namespace ascir {
using ApplyOutputView = std::function<std::string(const std::string &var)>;
struct ViewPolicy {
 public:
  enum ViewType : int64_t {
    kElementWise = 0,
    kReduce,
    kBroadCast,
    kInvalid,
  };
  ViewPolicy(uint32_t element_wise_input_index) : use_input_index(element_wise_input_index) {
    view_type = kElementWise;
  }
  ViewPolicy(uint32_t reduce_input_index, std::string reduce_axis_attr_name) : use_input_index(reduce_input_index),
                                                                               reduce_axis_attr_name(std::move(
                                                                                   reduce_axis_attr_name)) {
    view_type = kReduce;
  }

  explicit ViewPolicy(std::vector<uint32_t> broad_cast_input_indexs) : broad_cast_input_indexs(std::move(
      broad_cast_input_indexs)) {
    view_type = kBroadCast;
  }

  ViewType view_type{kInvalid};
  uint32_t use_input_index{UINT32_MAX};
  std::string reduce_axis_attr_name;
  std::vector<uint32_t> broad_cast_input_indexs;
};

inline ViewPolicy ReduceView(uint32_t index, const std::string &attr_name) {
  return ViewPolicy(index, attr_name);
}
inline ViewPolicy BroadCastView(const std::vector<uint32_t> &broad_cast_input_indexs) {
  return ViewPolicy(broad_cast_input_indexs);
}

struct DtypePolicy {
  enum PolicyType : int64_t {
    kUseInput = 0,
    kPromptInput,
    kUseDtype,
    kInvalid,
  };
 public:
  DtypePolicy(uint32_t use_input_index) : use_input_index(use_input_index) {
    policy_type = kUseInput;
  };
  DtypePolicy(ge::DataType data_type) : data_type(data_type) {
    policy_type = kUseDtype;
  };
  PolicyType policy_type{kInvalid};
  uint32_t use_input_index{UINT32_MAX};
  ge::DataType data_type{ge::DataType::DT_UNDEFINED};
};

inline DtypePolicy PromptDtype(uint32_t index) {
  auto policy = DtypePolicy(index);
  policy.policy_type = DtypePolicy::kPromptInput;
  return policy;
}
// TTODO: c++的类ABI兼容性不好，后面考虑换成C接口实现
struct AscIrAttrDef {
  std::string name;
  std::string asc_ir_type;
  std::string ge_ir_type;
};
enum CalcTmpBufSizeFuncType : int64_t {
  CommonType = 0,
  CustomizeType,
};
struct CalcTmpBufSizeFunc {
  std::string func_name;
  CalcTmpBufSizeFuncType func_type = CalcTmpBufSizeFuncType::CommonType;
  CalcTmpBufSizeFunc() = default;
  CalcTmpBufSizeFunc(std::string name, const CalcTmpBufSizeFuncType type) : func_name(std::move(name)), func_type(type) {}
};
struct AscIrDef {
  using CodeGenerator = void (*)(const AscIrDef &def, std::stringstream &ss);
  bool IsAttrExisted(const std::string &attr_name) const {
    return std::find_if(attr_defs.begin(), attr_defs.end(), [&attr_name](const AscIrAttrDef &asc_ir_attr_def) {
      return asc_ir_attr_def.name == attr_name;
    }) != attr_defs.end();
  }
  std::string file_path;
  int64_t line;
  std::string type;

  // 当前只有必选输入一种，没有其他类型，因此暂时简单处理，后续有复杂的optional后，defs的类型就不是string了
  std::vector<std::pair<std::string, IrInputType>> input_defs;
  std::vector<std::pair<std::string, IrOutputType>> output_defs;
  std::vector<AscIrAttrDef> attr_defs;

  std::vector<ViewPolicy> output_views_policy;
  std::vector<DtypePolicy> output_dtypes_policy;

  bool start_node{false};
  CodeGenerator infer_data_type_generator;
  CodeGenerator infer_view_generator;
  IRDataTypeSymbolStore dtype_symbol_store;
  std::string comment;
  CalcTmpBufSizeFunc calc_tmp_buf_size_func;
};

inline std::string UpdateViewIfCrossLoop(const std::string &trans_infos,
                                         const std::string &input_api_sched_axis,
                                         const std::string &op_attr_sched_axis,
                                         const std::string &tie_expression) {
  return "AxisUtils::UpdateViewIfCrossLoop(" + trans_infos + ", " + input_api_sched_axis + ", " + op_attr_sched_axis
      + ", " + "std::move(" + tie_expression + "))";
}

inline void GenChosenInputView(const AscIrDef &def, const uint32_t chosen_input_index, std::stringstream &ss) {
  ss << def.input_defs[chosen_input_index].first + "_tmp = " << "{*"
     << def.input_defs[chosen_input_index].first << "_in.axis, *"
     << def.input_defs[chosen_input_index].first << "_in.repeats, *"
     << def.input_defs[chosen_input_index].first << "_in.strides};"
     << std::endl;
}

inline void DefineChosenInputView(const AscIrDef &def, const ViewPolicy &policy,
                                  uint32_t &chosen_input_index,
                                  std::unordered_set<uint32_t> &chosen_input_index_set,
                                  std::stringstream &ss) {
  ss << "  // set tmp view to store input view and apply view transform" << std::endl;
  const std::string view_type("View ");
  ASCIR_ASSERT(policy.use_input_index < def.input_defs.size());
  chosen_input_index = policy.use_input_index;
  ss << "  ";
  if (chosen_input_index_set.insert(chosen_input_index).second) {
    ss << view_type;
  }
  GenChosenInputView(def, chosen_input_index, ss);
}

inline void SameDataTypeFromInput(const AscIrDef &def, std::stringstream &ss, const char *input_name) {
  for (const auto &output_def : def.output_defs) {
    ss << "  op." << output_def.first << ".dtype = static_cast<ge::DataType>(" << input_name << "_in.dtype);"
       << std::endl;
  }
}

inline void GenerateViewUpdateCode(const AscIrDef &def,
                                   const std::pair<size_t, size_t> out_to_chosen_input,
                                   const ApplyOutputView &apply_output_view,
                                   std::stringstream &ss,
                                   bool &gen_trans_infos_instance) {
  const size_t output_index = out_to_chosen_input.first;
  const size_t chosen_input_index = out_to_chosen_input.second;
  if (!gen_trans_infos_instance) {
    ss << "  auto trans_infos = CodeGenUtils::GetOwnerGraphAscAttr(op." << def.output_defs[output_index].first
       << ".GetOwnerOp())" << "->trans_info_road;"
       << std::endl;
    gen_trans_infos_instance = true;
  }

  const std::string which_input_api_sched_axis = def.output_defs[output_index].first + "_in_api_sched_axis";
  ss << "  auto " << which_input_api_sched_axis << " = CodeGenUtils::GetOwnerOpAscAttr("
     << def.input_defs[chosen_input_index].first << "_in.GetOwnerOp())"
     << "->sched.axis;" << std::endl;
  std::string view = def.input_defs[chosen_input_index].first + "_tmp";
  ss << "  {" << std::endl << "    const auto &[axes, repeats, strides] = ";
  std::string val =
      UpdateViewIfCrossLoop("trans_infos", which_input_api_sched_axis, "op.attr.sched.axis", view).append(".second");
  // 应用输出的语义变换
  if (!(apply_output_view == nullptr)) {
    ss << apply_output_view(val) << ";" << std::endl;
  } else {
    ss << val << ";" << std::endl;
  }
  ss << "    std::tie(*op." << def.output_defs[output_index].first << ".axis, *op."
     << def.output_defs[output_index].first << ".repeats, *op." << def.output_defs[output_index].first
     << ".strides) = std::make_tuple(axes, repeats, strides);" << std::endl
     << "  }" << std::endl;
}

inline ApplyOutputView GenApplyOutputViewFunc(const AscIrDef &def, const size_t output_index,
                                              uint32_t &chosen_input_index,
                                              std::stringstream &ss) {
  (void) chosen_input_index;
  const auto &policy = def.output_views_policy[output_index];
  ApplyOutputView apply_output_view;
  switch (policy.view_type) {
    case ViewPolicy::kElementWise:
      break;
    case ViewPolicy::kReduce:
      if (!def.IsAttrExisted(policy.reduce_axis_attr_name)) {
        return apply_output_view;
      }
      apply_output_view = [&def, output_index](const std::string &var) -> std::string {
        return "AxisUtils::ReduceView(" + var + ", " + def.output_views_policy[output_index].reduce_axis_attr_name +
            ")";
      };
      break;
    case ViewPolicy::kBroadCast: // TTODO 广播代码后续支持
    case ViewPolicy::kInvalid:
    default:
      ss << "unsupported policy type: " << policy.view_type << std::endl;
      break;
  }
  return apply_output_view;
}

inline void InferViewByPolicy(const AscIrDef &def, std::stringstream &ss) {
  if (def.output_defs.size() != def.output_views_policy.size()) {
    std::string error_info =
        std::string("view_policy's size ").append(std::to_string(def.output_views_policy.size())).append(
            " should be equal with output_defs's size ").append(std::to_string(def.output_defs.size()));
    ss << error_info;
    return;
  }
  bool gen_trans_infos_instance = false;
  std::unordered_set<uint32_t> chosen_input_index_set;
  for (size_t output_index = 0U; output_index < def.output_views_policy.size(); ++output_index) {
    uint32_t chosen_input_index = 0U;
    DefineChosenInputView(def, def.output_views_policy[output_index], chosen_input_index, chosen_input_index_set, ss);
    GenerateViewUpdateCode(def, std::make_pair(output_index, chosen_input_index),
                           GenApplyOutputViewFunc(def, output_index, chosen_input_index, ss), ss,
                           gen_trans_infos_instance);
  }
}

inline void InferDtypeByPolicy(const AscIrDef &def, std::stringstream &ss) {
  if (def.output_defs.size() != def.output_dtypes_policy.size()) {
    std::string error_info =
        std::string("dtype_policy's size ").append(std::to_string(def.output_dtypes_policy.size())).append(
            "should be equal with output_defs's size ").append(std::to_string(def.output_defs.size()));
    ss << error_info;
    return;
  }
  for (size_t output_index = 0U; output_index < def.output_dtypes_policy.size(); ++output_index) {
    const auto &policy = def.output_dtypes_policy[output_index];
    switch (policy.policy_type) {
      case DtypePolicy::kUseInput:ASCIR_ASSERT(policy.use_input_index < def.input_defs.size());
        ss << "  op." << def.output_defs[output_index].first << ".dtype = static_cast<ge::DataType>("
           << def.input_defs[policy.use_input_index].first << "_in.dtype);" << std::endl;
        break;
      case DtypePolicy::kPromptInput:ASCIR_ASSERT(policy.use_input_index < def.input_defs.size());
        ss << "  op." << def.output_defs[output_index].first
           << ".dtype = DtypeTransformUtils::Prompt(static_cast<ge::DataType>("
           << def.input_defs[policy.use_input_index].first << "_in.dtype));" << std::endl;
        break;
      case DtypePolicy::kUseDtype:
        ss << "  op." << def.output_defs[output_index].first << ".dtype = static_cast<ge::DataType>(" << policy.data_type
           << ");" << std::endl;
        break;
      case DtypePolicy::kInvalid:
      default:ss << "unsupported policy type: " << policy.policy_type << std::endl;
    }
  }
}

inline void SameDataTypeFromFirstInput(const AscIrDef &def, std::stringstream &ss) {
  if (!def.input_defs.empty()) {
    SameDataTypeFromInput(def, ss, def.input_defs[0].first.c_str());
  }
}
inline void SameDataTypeFromSecondInput(const AscIrDef &def, std::stringstream &ss) {
  if (def.input_defs.size() > 1U) {
    SameDataTypeFromInput(def, ss, def.input_defs[1].first.c_str());
  }
}
inline void SameViewFromInput(const AscIrDef &def, std::stringstream &ss, const char *input_name) {
  for (const auto &output_def : def.output_defs) {
    ss << "  op." << output_def.first << ".axis = " << input_name << "_in.axis;" << std::endl;
    ss << "  op." << output_def.first << ".repeats = " << input_name << "_in.repeats;" << std::endl;
    ss << "  op." << output_def.first << ".strides = " << input_name << "_in.strides;" << std::endl;
  }
}
inline void SameViewFromFirstInput(const AscIrDef &def, std::stringstream &ss) {
  if (!def.input_defs.empty()) {
    SameViewFromInput(def, ss, def.input_defs[0].first.c_str());
  }
}

class AscirRegistry {
 public:
  static AscirRegistry &GetInstance();
  void RegisterAscIr(const std::string &type, const AscIrDef &def);

  const std::unordered_map<std::string, AscIrDef> &GetAll() const;

 private:
  std::unordered_map<std::string, AscIrDef> types_to_ascir_;
};
}  // namespace ascir
}  // namespace ge
#endif  // AUTOFUSE_ASCIR_REGISTRY_H
