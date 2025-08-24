#include <math.h>
#include <vector>
#include <random>
#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "cpu_kernel_utils.h"
#include "cpu_nodedef_builder.h"
#undef private
#undef protected
#include "Eigen/Core"
#include "less.h"

using namespace std;
using namespace aicpu;

class TEST_LESS_UT : public testing::Test {};

class LessCpuKernelTest : LessCpuKernel {
  public :
  using LessCpuKernel::Compute;
};

template <typename T>
void CalcExpectWithSameShape(const NodeDef &node_def, bool expect_out[]) {
  auto input0 = node_def.MutableInputs(0);
  T *input0_data = (T *)input0->GetData();
  auto input1 = node_def.MutableInputs(1);
  T *input1_data = (T *)input1->GetData();
  int64_t input0_num = input0->NumElements();
  int64_t input1_num = input1->NumElements();
  if (input0_num == input1_num) {
    for (int64_t j = 0; j < input0_num; ++j) {
      expect_out[j] = input0_data[j] < input1_data[j] ? true : false;
    }
  }
}

uint64_t CalTotalElements(std::vector<std::vector<int64_t>> &shapes,
uint32_t index) {
  if(index < 0) {
    return 0;
  }
  uint64_t nums = 1;
  for(auto shape : shapes[index]) {
    nums = nums * shape;
  }
  return nums;
}

template <typename T>
void SetRandomValue(T input[], uint64_t num, float min = 0.0,
float max = 10.0) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(min, max);
  for (uint64_t i = 0; i < num; ++i) {
    input[i] = static_cast<T>(dis(gen));
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

template <typename T>
void CalcExpectWithDiffShape(const NodeDef &node_def, bool expect_out[]) {
  auto input0 = node_def.MutableInputs(0);
  T *input0_data = (T *)input0->GetData();
  auto input1 = node_def.MutableInputs(1);
  T *input1_data = (T *)input1->GetData();
  int64_t input0_num = input0->NumElements();
  int64_t input1_num = input1->NumElements();
  if (input0_num > input1_num) {
    for (int64_t j = 0; j < input0_num; ++j) {
      int64_t i = j % input1_num;
      expect_out[j] = input0_data[j] < input1_data[i] ? true : false;
    }
  }
}

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = NodeDefBuilder::CreateNodeDef();                 \
  NodeDefBuilder(node_def.get(), "Less", "Less")                   \
      .Input({"x1", data_types[0], shapes[0], datas[0]})           \
      .Input({"x2", data_types[1], shapes[1], datas[1]})           \
      .Output({"y", data_types[2], shapes[2], datas[2]})

#define RUN_KERNEL(node_def, HOST)                  \
  CpuKernelContext ctx(DEVICE);                     \
  EXPECT_EQ(ctx.Init(node_def.get()), 0);           \
  LessCpuKernelTest less;                           \
  less.Compute(ctx);


// only generate input data by SetRandomValue,
// and calculate output by youself function
template<typename T1, typename T2, typename T3>
void RunLessKernel2(vector<DataType> data_types,
                    vector<vector<int64_t>> &shapes) {
  // gen data use SetRandomValue for input1
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 *input1 = new T1[input1_size];
  SetRandomValue<T1>(input1, input1_size);

  // gen data use SetRandomValue for input2
  uint64_t input2_size = CalTotalElements(shapes, 1);
  T2 *input2 = new T2[input2_size];
  SetRandomValue<T2>(input2, input2_size);

  uint64_t output_size = CalTotalElements(shapes, 2);
  T3 *output = new T3[output_size];
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST);

  // calculate output_exp
  T3 *output_exp = new T3[output_size];
  if(input1_size == input2_size) {
    CalcExpectWithSameShape<T1>(*node_def.get(), output_exp);
  } else {
    CalcExpectWithDiffShape<T1>(*node_def.get(), output_exp);
  }

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete [] input1;
  delete [] input2;
  delete [] output;
  delete [] output_exp;
}

TEST_F(TEST_LESS_UT, DATA_TYPE_UINT16_SUCC) {
  vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
  RunLessKernel2<uint16_t, uint16_t, bool>(data_types, shapes);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_UINT32_SUCC) {
  vector<DataType> data_types = {DT_UINT32, DT_UINT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
  RunLessKernel2<uint32_t, uint32_t, bool>(data_types, shapes);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_UINT64_SUCC) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
  RunLessKernel2<uint64_t, uint64_t, bool>(data_types, shapes);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_UINT64_SAME_SHAPE_SUCC) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
  uint64_t input1[6] = {100, 3, 9, 4, 6, 8};
  uint64_t input2[6] = {3, 5, 9, 6, 0, 4};
  bool output[6] = {false};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST);

  bool output_exp[6] = {false};
  CalcExpectWithSameShape<uint64_t>(*node_def.get(), output_exp);

  bool compare = CompareResult(output, output_exp, 6);
  EXPECT_EQ(compare, true);
}

// exception instance
TEST_F(TEST_LESS_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2, 3}, {2, 2, 4}};
  int32_t input1[12] = {(int32_t)1};
  int32_t input2[16] = {(int32_t)0};
  bool output[16] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST);
}

TEST_F(TEST_LESS_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT64, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  int32_t input1[22] = {(int32_t)1};
  int64_t input2[22] = {(int64_t)0};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST);
}

TEST_F(TEST_LESS_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST);
}

TEST_F(TEST_LESS_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  bool input1[22] = {(bool)1};
  bool input2[22] = {(bool)0};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST);
}
