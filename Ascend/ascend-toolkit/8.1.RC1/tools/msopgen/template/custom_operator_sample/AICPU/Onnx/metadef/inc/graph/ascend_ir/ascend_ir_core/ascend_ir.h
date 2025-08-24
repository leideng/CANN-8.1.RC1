/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef GRAPH_ASCEND_IR_H
#define GRAPH_ASCEND_IR_H

#include <string>
#include <memory>
#include "attr_store.h"
#include "graph/compute_graph.h"
#include "symbolic.h"
#include "graph/node.h"
#include "graph/anchor.h"
#include "debug/ge_util.h"
#include "graph/utils/op_desc_utils.h"
#include "external/graph/operator.h"
#include "graph/utils/type_utils.h"
#include "graph/ascend_ir/ascend_ir_check.h"
#include "graph/expression/const_values.h"
#include "ascend_ir_def.h"

namespace ge {
struct DiffAxesInfo {
  std::vector<AxisId> add_axes;
  std::vector<AxisId> del_axes;
};
struct View {
  std::vector<int64_t> axis_ids;
  std::vector<ge::Expression> repeats;
  std::vector<ge::Expression> strides;
};
using TransInfoRoadOfGraph = std::vector<OneTransInfo>;

// 默认实现
template<typename T>
std::string ViewMemberToString(const std::vector<T> &vec) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    oss << vec[i];
    if (i < vec.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

// 特化实现，针对 ge::Expression 类型
template<>
inline std::string ViewMemberToString(const std::vector<ge::Expression> &vec) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    const ge::Expression &expr = vec[i];
    oss << (expr.Str() != nullptr ? expr.Str().get() : std::string("null"));
    if (i < vec.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

inline std::string ViewToString(const View &view) {
  std::string result = "{ axis: " + ViewMemberToString(view.axis_ids) +
      ", repeats: " + ViewMemberToString(view.repeats) +
      ", strides: " + ViewMemberToString(view.strides) +
      " }";
  return result;
}

class AscOutputAttrDataType {
 public:
  AscOutputAttrDataType(ge::Operator *op, uint32_t output_index) : op_(op), output_index_(output_index) {}

  void operator=(const ge::DataType &value) {
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(op_);
    const auto desc = ge::OpDescUtils::GetOpDescFromOperator(*op_);
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(desc);
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(desc->MutableOutputDesc(output_index_));
    desc->MutableOutputDesc(output_index_)->SetDataType(value);
  }

  operator ge::DataType() const {
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(op_);
    const auto desc = ge::OpDescUtils::GetOpDescFromOperator(*op_);
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(desc);
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(desc->MutableOutputDesc(output_index_));
    return desc->MutableOutputDesc(output_index_)->GetDataType();
  };

 private:
  ge::Operator *op_;
  uint32_t output_index_{UINT32_MAX};
};

class AscOutputAttrFormat {
 public:
  AscOutputAttrFormat(ge::Operator *op, uint32_t output_index) : op_(op), output_index_(output_index) {}

  void operator=(const ge::Format &value) {
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(op_);
    const auto desc = ge::OpDescUtils::GetOpDescFromOperator(*op_);
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(desc);
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(desc->MutableOutputDesc(output_index_));
    desc->MutableOutputDesc(output_index_)->SetFormat(value);
  }

  operator ge::Format() const {
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(op_);
    const auto desc = ge::OpDescUtils::GetOpDescFromOperator(*op_);
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(desc);
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(desc->MutableOutputDesc(output_index_));
    return desc->MutableOutputDesc(output_index_)->GetFormat();
  };

 private:
  ge::Operator *op_{};
  uint32_t output_index_{UINT32_MAX};
};

struct AscOpOutput {
 public:
  class AscOpOutputOffsetHelper {
   public:
    explicit AscOpOutputOffsetHelper(AscOpOutput &output) : output_(output) {}
    ~AscOpOutputOffsetHelper() = default;
    void operator=(const AscOpOutput &asc_op_output) {
      CHECK_NOTNULL_WITH_THROW_EXCEPTION(asc_op_output.op_);
      output_.op_ = asc_op_output.op_;
      output_.output_index = asc_op_output.output_index;
      output_.dtype = asc_op_output.dtype;
      output_.format = asc_op_output.format;
      output_.axis = asc_op_output.axis;
      output_.repeats = asc_op_output.repeats;
      output_.strides = asc_op_output.strides;
      *asc_op_output.vectorized_axis = output_.load_vectorized_axes_;
      output_.vectorized_axis = asc_op_output.vectorized_axis;
      output_.mem = asc_op_output.mem;
      output_.que = asc_op_output.que;
      output_.buf = asc_op_output.buf;
      output_.opt = asc_op_output.opt;
    }
   private:
    AscOpOutput &output_;
  };
  template<uint32_t INPUT_INDEX>
  friend class AscOpInput;
  template<uint32_t INPUT_INDEX>
  friend class AscOpDynamicInput;
  friend class VectorizedOutTensor;
  AscOpOutput(ge::Operator *op, uint32_t output_index)
      : op_(op), output_index(output_index), dtype(op, output_index), format(op, output_index) {
    TryInitTensorAttr();
  }
  explicit AscOpOutput(std::vector<int64_t> axis_ids)
      : op_(nullptr), load_vectorized_axes_(std::move(axis_ids)), output_index(0), dtype(nullptr, output_index),
        format(nullptr, output_index), repeats(nullptr), strides(nullptr), vectorized_axis(nullptr), mem(nullptr),
        que(nullptr), buf(nullptr), opt(nullptr) {}
  AscOpOutput(const AscOpOutput &output) : AscOpOutput(output.op_, output.output_index) {}
  AscOpOutput(AscOpOutput &&output)  noexcept : AscOpOutput(output.op_, output.output_index) {}

  void SetContiguousView(const std::vector<Axis> &axes) {
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(axis, "output tensor should bind to API by API function or by AutoOffset");
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(repeats, "output tensor should bind to API by API function or by AutoOffset");
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(strides, "output tensor should bind to API by API function or by AutoOffset");
    std::vector<AxisId> axes_ids;
    std::vector<ge::Expression> tmp_repeats;
    std::vector<ge::Expression> tmp_strides;
    axes_ids.reserve(axes.size());
    tmp_repeats.reserve(axes.size());
    tmp_strides.reserve(axes.size());

    std::for_each(axes.rbegin(), axes.rend(),
                  [&axes_ids, &tmp_repeats, &tmp_strides](const Axis &tmp_axis) {
                    if (tmp_strides.empty()) {
                      tmp_strides.emplace_back(sym::kSymbolOne);
                    } else {
                      tmp_strides.emplace_back(*tmp_repeats.rbegin() * *tmp_strides.rbegin());
                    }
                    tmp_repeats.emplace_back(tmp_axis.size);
                    axes_ids.emplace_back(tmp_axis.id);
                  });
    std::reverse(axes_ids.begin(), axes_ids.end());
    std::reverse(tmp_repeats.begin(), tmp_repeats.end());
    std::reverse(tmp_strides.begin(), tmp_strides.end());

    *axis = axes_ids;
    *repeats = tmp_repeats;
    *strides = tmp_strides;
  }

  const ge::Operator &GetOwnerOp() const {
    return *op_;
  }

  ge::Operator &MutableOwnerOp() {
    return *op_;
  }

  void TryInitTensorAttr() {
    auto tensor_attr_ptr = AscTensorAttr::GetTensorAttrPtr(op_, output_index);
    if (tensor_attr_ptr == nullptr) {
      return;
    }
    auto &tensor_attr = *tensor_attr_ptr;
    axis = &tensor_attr.axis;
    repeats = &tensor_attr.repeats;
    strides = &tensor_attr.strides;
    vectorized_axis = &tensor_attr.vectorized_axis;
    mem = &tensor_attr.mem;
    que = &tensor_attr.que;
    buf = &tensor_attr.buf;
    opt = &tensor_attr.opt;
  }
  AscOpOutput &Use(const AscOpOutput &used_out) {
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(op_, "output tensor should bind to API by API function or by AutoOffset");
    CHECK_BOOL_WITH_THROW_EXCEPTION(PARAM_INVALID, !HasBindToContainer(),
                                    " this tensor has been bound to a que or buf, can not be repeated bound.");
    CHECK_BOOL_WITH_THROW_EXCEPTION(PARAM_INVALID, used_out.HasBindToContainer(),
                                    " tensor to be used has not been bound to any que or buf.");
    if (used_out.que->id != kIdNone) {
      UseTQue(used_out.mem->position, used_out.que->depth, used_out.que->buf_num, used_out.que->id);
    }
    if (used_out.buf->id != kIdNone) {
      UseTBuf(used_out.mem->position, used_out.buf->id);
    }
    return *this;
  }

  AscOpOutput &TQue(const Position pos, const int64_t depth, const int64_t buf_num) {
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(op_, "output tensor should bind to API by API function or by AutoOffset");
    UseTQue(pos, depth, buf_num);
    return *this;
  }

  AscOpOutput &TBuf(const Position pos) {
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(op_, "output tensor should bind to API by API function or by AutoOffset");
    UseTBuf(pos);
    return *this;
  }
  AscOpOutputOffsetHelper AutoOffset() {
    return AscOpOutputOffsetHelper(*this);
  }

 private:
  int64_t GenContainerId();
  void UseTQue(const Position pos, const int64_t depth, const int64_t buf_num, const int64_t id = kIdNone);
  void UseTBuf(const Position pos, const int64_t id = kIdNone);
  bool HasBindToContainer() const;
  ge::Operator *op_;
  std::vector<int64_t> load_vectorized_axes_;
 public:
  uint32_t output_index{UINT32_MAX};
  AscOutputAttrDataType dtype;
  AscOutputAttrFormat format;
  std::vector<int64_t> *axis;
  std::vector<ge::Expression> *repeats;
  std::vector<ge::Expression> *strides;
  std::vector<int64_t> *vectorized_axis;
  MemAttr *mem;
  MemQueAttr *que;
  MemBufAttr *buf;
  MemOptAttr *opt;
};

class VectorizedOutTensor {
 public:
  explicit VectorizedOutTensor(std::vector<int64_t> vectorized_axis) : vectorized_axis_(std::move(vectorized_axis)) {
  }
  VectorizedOutTensor &operator=(const VectorizedOutTensor &) = delete;
  VectorizedOutTensor(const VectorizedOutTensor &) = delete;
  explicit operator AscOpOutput() const {
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(op_);
    return AscOpOutput(op_, output_index_);
  }
  void operator=(AscOpOutput &&asc_op_output) {
    // 不支持更改归属
    if (op_ != nullptr) {
      AscendString name;
      ASCIR_ASSERT_GRAPH_SUCCESS(op_->GetName(name));
      GELOGE(FAILED, "Tensor has been bind to %s", name.GetString());
      return;
    }
    op_ = asc_op_output.op_;
    output_index_ = asc_op_output.output_index;
    // 修改归属op的向量化轴信息
    AscTensorAttr::GetTensorAttr(op_, output_index_).vectorized_axis = vectorized_axis_;
  }
 private:
  std::vector<int64_t> vectorized_axis_;
  ge::Operator *op_{nullptr};
  uint32_t output_index_{UINT32_MAX};
};

void AddEdgeForNode(const ge::Operator &src_op, int32_t src_index, ge::Operator &dst_op, int32_t dst_index);
void LinkByIrIndex(const ge::Operator &src_op, uint32_t src_ir_index, ge::Operator &dst_op, uint32_t dst_ir_index,
                   uint32_t dynamic_index = 0U);
void SetDynamicInputNumByIrIndex(ge::Operator &op, uint32_t ir_index, uint32_t dynamic_num);

template<uint32_t INPUT_INDEX>
class AscOpInput {
 public:
  explicit AscOpInput(ge::Operator *op) : op_(op) {}

  AscOpInput<INPUT_INDEX> &operator=(const AscOpOutput &output) {
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(op_);
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(output.op_);
    LinkByIrIndex(*output.op_, output.output_index, *this->op_, INPUT_INDEX);
    return *this;
  }

 private:
  ge::Operator *op_;
};

struct AscTensor {
  friend struct AscNodeOutputs;
  friend struct AscNodeInputs;
  explicit AscTensor(const ge::OutDataAnchor &an) : attr(AscTensorAttr::GetTensorAttr(an)), anchor(an) {}
  ~AscTensor() = default;
  AscTensorAttr &attr;              // not owner
  const ge::OutDataAnchor &anchor;  // not owner
};

struct AscNodeOutputs {
  explicit AscNodeOutputs(ge::Node *node) : node_(node) {
    Init();
  }
  AscTensor &operator[](uint32_t index);
  std::vector<AscTensor *> operator()();
 private:
  void Init();
  std::vector<AscTensor> tensors_;
  Node *node_;
};

struct AscNodeInputs {
  explicit AscNodeInputs(ge::Node *node) : node_(node) {
    Init();
  }
  AscTensor &operator[](uint32_t index);
  std::vector<AscTensor *> operator()();
  uint32_t Size();
 private:
  void Init();
  std::vector<AscTensor> tensors_;
  Node *node_;
};

class AscNode : public Node {
 public:
  AscNode(const OpDescPtr &op_desc, const ComputeGraphPtr &compute_graph);
  AscNodeInputs inputs;
  AscNodeOutputs outputs;
  AscNodeAttr &attr;
};
using AscNodePtr = std::shared_ptr<AscNode>;

class AscNodeIter {
 public:
  explicit AscNodeIter(ge::ComputeGraph::Vistor<ge::NodePtr>::Iterator &&iter);
  AscNodeIter &operator++();
  AscNodePtr operator*();
  bool operator!=(const AscNodeIter &other) const;
 private:
  ge::ComputeGraph::Vistor<ge::NodePtr>::Iterator impl_;
};

class AscNodeVisitor {
 public:
  using Iterator = ge::ComputeGraph::Vistor<ge::NodePtr>::Iterator;
  AscNodeIter begin();
  AscNodeIter end();
  explicit AscNodeVisitor(ge::ComputeGraph::Vistor<ge::NodePtr> &&visitor);
 private:
  ge::ComputeGraph::Vistor<ge::NodePtr> impl_;
};


template<uint32_t INPUT_INDEX>
class AscOpDynamicInput {
 public:
  explicit AscOpDynamicInput(ge::Operator *op) : op_(op) {}
  AscOpDynamicInput<INPUT_INDEX> &operator=(const std::initializer_list<AscOpOutput> &outputs) {
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(op_);
    CHECK_BOOL_WITH_THROW_EXCEPTION(PARAM_INVALID, !inited_,
                                    "It is not allowed to set the dynamic input repeatedly, node:[%s], type:[%s].",
                                    op_->GetName().c_str(), op_->GetOpType().c_str());
    const size_t input_nums = outputs.size();
    SetDynamicInputNumByIrIndex(*this->op_, INPUT_INDEX, input_nums);
    size_t idx = 0UL;
    for (const auto &output : outputs) {
      CHECK_NOTNULL_WITH_THROW_EXCEPTION(output.op_);
      LinkByIrIndex(*output.op_, output.output_index, *this->op_, INPUT_INDEX, idx++);
    }
    inited_ = true;
    return *this;
  }

 private:
  ge::Operator *op_{nullptr};
  bool inited_{false};
};

class AscGraphImpl;
namespace ascir {
namespace cg {
class CodeGenUtils;
}
}
class AscGraph {
  friend class ascir::cg::CodeGenUtils;
  friend class AscGraphUtils;
 public:
  explicit AscGraph(const char *name);
  ~AscGraph();
  void SetTilingKey(const uint32_t tiling_key);
  int64_t GetTilingKey() const;
  void CreateSizeVar(const Expression &expression);
  Expression CreateSizeVar(const int64_t value);
  Expression CreateSizeVar(const std::string &name);
  Axis &CreateAxis(const std::string &name, const Expression &size);
  Axis &CreateAxis(const std::string &name, Axis::Type type, const Expression &size, const std::vector<AxisId> &from,
                   AxisId split_peer);
  Axis *FindAxis(const int64_t axis_id);
  AscNodePtr AddNode(ge::Operator &op);
  AscNodePtr FindNode(const char *name) const;
  AscNode &Node(const char *name) const;
  std::pair<AxisPtr, AxisPtr> BlockSplit(const int64_t axis_id, const std::string &outer_axis_name = "",
                                         const std::string &inner_axis_name = "");
  std::pair<AxisPtr, AxisPtr> TileSplit(const int64_t axis_id, const std::string &outer_axis_name = "",
                                        const std::string &inner_axis_name = "");
  AxisPtr MergeAxis(const std::vector<int64_t> &axis_ids, const std::string &merge_axis_name = "");
  void ApplySplit(const AscNodePtr &node, const int64_t outter_id, const int64_t inner_id);
  void ApplyMerge(const AscNodePtr &node, const int64_t merged_axis_id);
  void ApplyReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis);
  void ApplySchedAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id, const std::vector<int64_t> &original);
  void ApplySchedAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id);
  void ApplyTensorAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id, const std::vector<int64_t> &original);
  void ApplyTensorAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id);
  void ApplySchedAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis);
  void ApplyTensorAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis);
  bool TryApplyAxisReplace(const AscNodePtr &node, const Axis &src, const Axis &dst);
  AscNodeVisitor GetAllNodes() const;
  AscNodeVisitor GetInputNodes() const;
  std::vector<SizeVarPtr> GetAllSizeVar() const;
  std::vector<AxisPtr> GetAllAxis() const;
  TransInfoRoadOfGraph GetAllAxisTransInfo() const;
  std::string GetName() const;
  bool CheckValid() const;
  AscOpOutput CreateContiguousData(const char *name,
                                   const ge::DataType &dt,
                                   const std::vector<ge::Axis> &axes,
                                   const std::vector<std::vector<int64_t>> &axis_continuous_map,
                                   const ge::Format &format = ge::FORMAT_ND);

  AscOpOutput CreateContiguousData(const char *name,
                                   const ge::DataType &dt,
                                   const std::vector<ge::Axis> &axes,
                                   const ge::Format &format = ge::FORMAT_ND);

  AscOpOutput CreateContiguousOut(const char *name,
                                  const ge::DataType &dt,
                                  const std::vector<ge::Axis> &axes,
                                  const ge::Format &format = ge::FORMAT_ND);
  void SortByExecOrder();
  void CopyFrom(const ge::AscGraph &graph);
 private:
  bool CheckExprValid() const;
  bool CheckAxisValid() const;
  bool CheckExecOrderValid() const;
  bool CheckTensorValid() const;
  bool CheckNodeConnectionValid() const;
  std::shared_ptr<AscGraphImpl> impl_;
};
}  // namespace ge

#endif  // GRAPH_ASCEND_IR_H
