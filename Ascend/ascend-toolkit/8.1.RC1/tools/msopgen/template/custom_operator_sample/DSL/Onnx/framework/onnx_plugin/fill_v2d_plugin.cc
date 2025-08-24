/* Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include <string>
#include <vector>
#include "graph.h"
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "all_ops.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
enum OnnxDataType {
  UNDEFINED = 0,
  FLOAT = 1,
  UINT_8 = 2,
  INT_8 = 3,
  UINT_16 = 4,
  INT_16 = 5,
  INT_32 = 6,
  INT_64 = 7,
  STRING = 8,
  BOOL = 9,
  FLOAT_16 = 10,
  DOUBLE = 11,
  UINT_32 = 12,
  UINT_64 = 13
};

DataType GetGeDataType(int32_t data_type)
{
  static DataType onnxToGeDataType[UINT_64 + 1] = {
    DT_UNDEFINED,
    DT_FLOAT,
    DT_UINT8,
    DT_INT8,
    DT_UINT16,
    DT_INT16,
    DT_INT32,
    DT_INT64,
    DT_STRING,
    DT_BOOL,
    DT_FLOAT16,
    DT_DOUBLE,
    DT_UINT32,
    DT_UINT64
  };
  if ((data_type > UINT_64) || (data_type < 0)) {
    return DT_UNDEFINED;
  }
  return onnxToGeDataType[data_type];
}

static uint8_t* ParseTensorValue(const ge::onnx::TensorProto &tp)
{
  const uint8_t *data = nullptr;
  auto data_type = tp.data_type();
  OP_LOGI("ConstantOfShape", "Datatype[%ld.]", data_type);
  switch (data_type) {
    case ge::onnx::TensorProto::DataType::TensorProto_DataType_INT64:
      data = reinterpret_cast<const uint8_t *>(tp.int64_data().data());
      break;
    case ge::onnx::TensorProto::DataType::TensorProto_DataType_INT32:
      data = reinterpret_cast<const uint8_t *>(tp.int32_data().data());
      break;
    case ge::onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:
      data = reinterpret_cast<const uint8_t *>(tp.float_data().data());
      break;
    case ge::onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:
      data = reinterpret_cast<const uint8_t *>(tp.double_data().data());
      break;
    default:
      OP_LOGE("ConstantOfShape", "Datatype[%ld] don't support.", data_type);
  }
  return const_cast<uint8_t *>(data);
}

Status ParseParamsConstantOfShape(const Message *op_src, ge::Operator &op_dest)
{
  const ge::onnx::NodeProto *node = dynamic_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE("ConstantOfShape", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  // 1.add dynamic input and out
  opDesc->AddDynamicInputDesc("args", 1);
  opDesc->AddDynamicOutputDesc("output", 1);
  // 2.set original_type
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::11::ConstantOfShape");
  // 3.set attr if needed
  ge::TensorDesc tensorDesc;
  vector<int64_t> dims = {};
  ge::Shape shape(dims);
  tensorDesc.SetShape(shape);
  tensorDesc.SetDataType(DT_FLOAT);
  tensorDesc.SetFormat(ge::FORMAT_NCHW);
  tensorDesc.SetOriginShape(shape);
  tensorDesc.SetOriginFormat(ge::FORMAT_NCHW);
  size_t size = sizeof(float);
  uint8_t *data = nullptr;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "value" && attr.type() == ge::onnx::AttributeProto::TENSOR) {
      if (attr.t().raw_data() != "") {
        auto value = const_cast<char *>(attr.t().raw_data().data());
        data = reinterpret_cast<uint8_t *>(value);
      } else {
        data = ParseTensorValue(attr.t());
      }
      DataType datatype0 = GetGeDataType(attr.t().data_type());
      size = GetSizeByDataType(datatype0);
      tensorDesc.SetDataType(datatype0);
    }
  }
  const ge::Tensor valueTensor(tensorDesc, data, size);
  op_dest.SetAttr("value", valueTensor);
  return SUCCESS;
}

static Status ParseOpToGraphConstantOfShape(const Operator &op, Graph &graph)
{
  auto data0 = op::Data("data0").set_attr_index(0);
  ge::Tensor value;
  if (op.GetAttr("value", value) != SUCCESS) {
    OP_LOGE("ConstantOfShape", "get value from op failed");
    return FAILED;
  }
  auto data1 = op::Const("data1").set_attr_value(value);
  auto fill = op::Fill().set_input_dims(data0).set_input_value(data1);

  std::vector<Operator> inputs { data0 };
  std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
  output_indexs.emplace_back(fill, vector<std::size_t> { 0 });
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::ConstantOfShape",
                 "ai.onnx::9::ConstantOfShape",
                 "ai.onnx::10::ConstantOfShape",
                 "ai.onnx::11::ConstantOfShape",
                 "ai.onnx::12::ConstantOfShape",
                 "ai.onnx::13::ConstantOfShape"})
  .ParseParamsFn(ParseParamsConstantOfShape)
  .ParseOpToGraphFn(ParseOpToGraphConstantOfShape)
  .ImplyType(ImplyType::TVM);
}
