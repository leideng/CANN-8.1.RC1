#include "gtest/gtest.h"
#include <math.h>
#include <stdint.h>
#include <Eigen/Dense>
#ifndef private
#define private public
#define protected public
#endif
#include "cpu_nodedef_builder.h"
#include "cpu_kernel_utils.h"
#undef private
#undef protected
#include "cast.h"

using namespace std;
using namespace aicpu;
using namespace Eigen;

namespace {
const char *Test = "Cast";
}

class TEST_CAST_UT : public testing::Test {};

template <typename Tin, typename Tout>
void CalcExpectFunc(const NodeDef &node_def, Tin input_type, Tout expect_out[]) {
  auto input = node_def.MutableInputs(0);
  auto output = node_def.MutableOutputs(0);
  Tin *input_data = (Tin *)input->GetData();
  Tout *output_data = (Tout *)output->GetData();

  int64_t input_num = input->NumElements();

  for (int i = 0; i < input_num; i++) {
    expect_out[i] = (Tout)input_data[i];
  }
}

template <typename T>
bool CompareResult(T output[], T expectOutput[], int num) {
  bool result = true;
  for (int i = 0; i < num; ++i) {
    if (output[i] != expectOutput[i]) {
      cout << "output[" << i << "] = ";
      cout << output[i];
      cout << "expectOutput[" << i << "] =";
      cout << expectOutput[i];
      result = false;
    }
  }
  return result;
}

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = NodeDefBuilder::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Cast", "Cast")                   \
      .Input({"x", data_types[0], shapes[0], datas[0]})            \
      .Output({"y", data_types[1], shapes[1], datas[1]})           \
      .Attr("SrcT", data_types[0])                                 \
      .Attr("DstT", data_types[1]);

TEST_F(TEST_CAST_UT, TestCast_DT_FLOAT_To_DT_INT8) {
  if (true) {
    vector<DataType> data_types = {DT_FLOAT, DT_INT8};
    float input[6] = {(float)22, (float)32.3,
                            (float)-78.0, (float)-28.5,
                            (float)77, (float)0};
    int8_t output[6] = {(int8_t)0};
    vector<void *> datas = {(void *)input, (void *)output};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    CREATE_NODEDEF(shapes, data_types, datas);
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(node_def.get()), 0);
    CastCpuKernel cast;
    cast.Compute(ctx);
    int8_t expect_out[6] = {(int8_t)0};
    float input_type = (float)0;
    CalcExpectFunc(*node_def.get(), input_type, expect_out);
    CompareResult<int8_t>(output, expect_out, 6);
  } else {
    vector<void *> datas = {nullptr, nullptr};
    vector<vector<int64_t>> shapes = {{}, {}};
    vector<DataType> data_types = {DT_FLOAT, DT_INT8};
    CREATE_NODEDEF(shapes, data_types, datas);
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(node_def.get()), 0);
    CastCpuKernel cast;
    cast.Compute(ctx);
  }
}
