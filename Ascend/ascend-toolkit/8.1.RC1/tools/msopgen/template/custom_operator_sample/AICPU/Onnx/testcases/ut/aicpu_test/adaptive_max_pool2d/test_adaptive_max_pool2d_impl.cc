#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "cpu_nodedef_builder.h"
#include "adaptive_max_pool2d.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_ADAPTER_MAX_POOL2D_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, input, output0, output1, list_out)                  \
  auto node_def = NodeDefBuilder::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "AdaptiveMaxPool2d", "AdaptiveMaxPool2d")                     \
      .Attr("output_size", list_out)           \
      .Input({"x", data_types[0], shapes[0], input, FORMAT_NCHW})           \
      .Output({"y", data_types[1], shapes[1], output0})           \
      .Output({"argmax", data_types[2], shapes[1], output1});

#define CREATE_NODEDEF_NHWC(shapes, data_types, input, output0, output1, list_out)                  \
  auto node_def = NodeDefBuilder::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "AdaptiveMaxPool2d", "AdaptiveMaxPool2d")                     \
      .Attr("output_size", list_out)           \
      .Input({"x", data_types[0], shapes[0], input, FORMAT_NHWC})           \
      .Output({"y", data_types[1], shapes[1], output0})           \
      .Output({"argmax", data_types[2], shapes[1], output1});

#define CREATE_NODEDEF2(shapes, data_types, input, output0, output1, list_out)                  \
  auto node_def = NodeDefBuilder::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "AdaptiveMaxPool2d", "AdaptiveMaxPool2d")                     \
      .Attr("output_size", list_out)           \
      .Input({"x", data_types[0], shapes[0], input, FORMAT_NCHW})           \
      .Output({"y", data_types[1], shapes[1], output0})           \
      .Output({"argmax", data_types[2], shapes[2], output1});

#define CREATE_NODEDEF3(shapes, data_types, input, output0, output1, list_out, format)                  \
  auto node_def = NodeDefBuilder::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "AdaptiveMaxPool2d", "AdaptiveMaxPool2d")                     \
      .Attr("output_size", list_out)           \
      .Input({"x", data_types[0], shapes[0], input, format})           \
      .Output({"y", data_types[1], shapes[1], output0})           \
      .Output({"argmax", data_types[2], shapes[1], output1});

#define RUN_KERNEL(node_def, HOST)                  \
  CpuKernelContext ctx(DEVICE);                     \
  EXPECT_EQ(ctx.Init(node_def.get()), 0);           \
  AdaptiveMaxPool2d adaptiveMaxPool2d;                           \
  adaptiveMaxPool2d.Compute(ctx);

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

vector<int64_t> list_out_1 = {1, 2};
vector<vector<int64_t>> shapes_1 = {{2, 2, 3}, {2, 1, 2}};
float_t input_1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
float_t expect_output0_1[] = {5, 6, 11, 12};
int32_t expect_output1_1[] = {4, 5, 4, 5};

TEST_F(TEST_ADAPTER_MAX_POOL2D_UT, TestAdaptiveMaxPool2d_dapter_max_pool2d_float_succ_1) {
    int32_t out_data_num = sizeof(expect_output0_1)/sizeof(expect_output0_1[0]);
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32};
    float_t output0[out_data_num] = {(float_t)0};
    int32_t output1[out_data_num] = {(int32_t)0};
    CREATE_NODEDEF(shapes_1, data_types, input_1, output0, output1, list_out_1);
    RUN_KERNEL(node_def, HOST);
    EXPECT_EQ(CompareResult<float_t>(output0, expect_output0_1, out_data_num), true);
    EXPECT_EQ(CompareResult<int32_t>(output1, expect_output1_1, out_data_num), true);
}


vector<int64_t> list_out_nhwc_1 = {1, 2};
vector<vector<int64_t>> shapes_nhwc_1 = {{2, 3, 2}, {1, 2, 2}};
float_t input_nhwc_1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
float_t expect_output0_nhwc_1[] = {5, 6, 11, 12};
int32_t expect_output1_nhwc_1[] = {4, 5, 4, 5};
TEST_F(TEST_ADAPTER_MAX_POOL2D_UT, TestAdaptiveMaxPool2d_dapter_max_pool2d_float_succ_NHWC_1) {
    int32_t out_data_num = sizeof(expect_output0_nhwc_1)/sizeof(expect_output0_nhwc_1[0]);
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32};
    float_t output0[out_data_num] = {(float_t)0};
    int32_t output1[out_data_num] = {(int32_t)0};
    CREATE_NODEDEF_NHWC(shapes_nhwc_1, data_types, input_nhwc_1, output0, output1, list_out_nhwc_1);
    RUN_KERNEL(node_def, HOST);
    EXPECT_EQ(CompareResult<float_t>(output0, expect_output0_nhwc_1, out_data_num), true);
    EXPECT_EQ(CompareResult<int32_t>(output1, expect_output1_nhwc_1, out_data_num), true);
}


vector<int64_t> list_out_2 = {2, 2};
vector<vector<int64_t>> shapes_2 = {{1, 4, 4}, {1, 2, 2}};
float_t input_2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
float_t expect_output0_2[] = {6, 8, 14, 16};
int32_t expect_output1_2[] = {5, 7, 13, 15};
TEST_F(TEST_ADAPTER_MAX_POOL2D_UT, TestAdaptiveMaxPool2d_dapter_max_pool2d_float_succ_2) {
    int32_t out_data_num = sizeof(expect_output0_2)/sizeof(expect_output0_2[0]);
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32};
    float_t output0[out_data_num] = {(float_t)0};
    int32_t output1[out_data_num] = {(int32_t)0};
    CREATE_NODEDEF(shapes_2, data_types, input_2, output0, output1, list_out_2);
    RUN_KERNEL(node_def, HOST);
    EXPECT_EQ(CompareResult<float_t>(output0, expect_output0_2, out_data_num), true);
    EXPECT_EQ(CompareResult<int32_t>(output1, expect_output1_2, out_data_num), true);
}


vector<int64_t> list_out_3 = {2, 1};
vector<vector<int64_t>> shapes_3 = {{4, 2, 3}, {4, 2, 1}};
float_t input_3[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
float_t expect_output0_3[] = {3,6,9,12,15,18,21,24};
int64_t expect_output1_3[] = {2,5,2,5,2,5,2,5};
TEST_F(TEST_ADAPTER_MAX_POOL2D_UT, TestAdaptiveMaxPool2d_dapter_max_pool2d_float_succ_3) {
    int64_t out_data_num = sizeof(expect_output0_3)/sizeof(expect_output0_3[0]);
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT64};
    float_t output0[out_data_num] = {(float_t)0};
    int64_t output1[out_data_num] = {(int64_t)0};
    CREATE_NODEDEF(shapes_3, data_types, input_3, output0, output1, list_out_3);
    RUN_KERNEL(node_def, HOST);
    EXPECT_EQ(CompareResult<float_t>(output0, expect_output0_3, out_data_num), true);
    EXPECT_EQ(CompareResult<int64_t>(output1, expect_output1_3, out_data_num), true);
}


vector<int64_t> list_out_4 = {3, 4};
vector<vector<int64_t>> shapes_4 = {{1, 1, 4}, {1, 3, 4}};
float_t input_4[] = {1, 2, 3, 4};
float_t expect_output0_4[] = {1,2,3,4,1,2,3,4,1,2,3,4};
int64_t expect_output1_4[] = {0,1,2,3,0,1,2,3,0,1,2,3};
TEST_F(TEST_ADAPTER_MAX_POOL2D_UT, TestAdaptiveMaxPool2d_dapter_max_pool2d_float_succ_4) {
    int64_t out_data_num = sizeof(expect_output0_4)/sizeof(expect_output0_4[0]);
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT64};
    float_t output0[out_data_num] = {(float_t)0};
    int64_t output1[out_data_num] = {(int64_t)0};
    CREATE_NODEDEF(shapes_4, data_types, input_4, output0, output1, list_out_4);
    RUN_KERNEL(node_def, HOST);
    EXPECT_EQ(CompareResult<float_t>(output0, expect_output0_4, out_data_num), true);
    EXPECT_EQ(CompareResult<int64_t>(output1, expect_output1_4, out_data_num), true);
}


vector<int64_t> list_out_5 = {3, 4};
vector<vector<int64_t>> shapes_5 = {{1, 2, 2}, {1, 3, 4}};
float_t input_5[] = {1, 2, 3, 4};
float_t expect_output0_5[] = {1,1,2,2,3,3,4,4,3,3,4,4};
int64_t expect_output1_5[] = {0,0,1,1,2,2,3,3,2,2,3,3};
TEST_F(TEST_ADAPTER_MAX_POOL2D_UT, TestAdaptiveMaxPool2d_dapter_max_pool2d_float_succ_5) {
    int64_t out_data_num = sizeof(expect_output0_5)/sizeof(expect_output0_5[0]);
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT64};
    float_t output0[out_data_num] = {(float_t)0};
    int64_t output1[out_data_num] = {(int64_t)0};
    CREATE_NODEDEF(shapes_5, data_types, input_5, output0, output1, list_out_5);
    RUN_KERNEL(node_def, HOST);
    EXPECT_EQ(CompareResult<float_t>(output0, expect_output0_5, out_data_num), true);
    EXPECT_EQ(CompareResult<int64_t>(output1, expect_output1_5, out_data_num), true);
}


vector<int64_t> list_out_6 = {2};
vector<vector<int64_t>> shapes_6 = {{1, 4, 4}, {1, 2, 2}};
float_t input_6[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
float_t expect_output0_6[] = {6, 8, 14, 16};
int32_t expect_output1_6[] = {5, 7, 13, 15};
TEST_F(TEST_ADAPTER_MAX_POOL2D_UT, TestAdaptiveMaxPool2d_dapter_max_pool2d_float_succ_6) {
    int32_t out_data_num = sizeof(expect_output0_6)/sizeof(expect_output0_6[0]);
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32};
    float_t output0[out_data_num] = {(float_t)0};
    int32_t output1[out_data_num] = {(int32_t)0};
    CREATE_NODEDEF(shapes_6, data_types, input_6, output0, output1, list_out_6);
    RUN_KERNEL(node_def, HOST);
    EXPECT_EQ(CompareResult<float_t>(output0, expect_output0_6, out_data_num), true);
    EXPECT_EQ(CompareResult<int32_t>(output1, expect_output1_6, out_data_num), true);
}


vector<int64_t> list_out_7 = {1,2};
vector<vector<int64_t>> shapes_7 = {{2,1,2,4}, {2,1,1,2}};
float_t input_7[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
float_t expect_output0_7[] = {6, 8, 14, 16};
int32_t expect_output1_7[] = {5, 7, 5, 7};
TEST_F(TEST_ADAPTER_MAX_POOL2D_UT, TestAdaptiveMaxPool2d_dapter_max_pool2d_float_succ_7) {
    int32_t out_data_num = sizeof(expect_output0_7)/sizeof(expect_output0_7[0]);
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32};
    float_t output0[out_data_num] = {(float_t)0};
    int32_t output1[out_data_num] = {(int32_t)0};
    CREATE_NODEDEF(shapes_7, data_types, input_7, output0, output1, list_out_7);
    RUN_KERNEL(node_def, HOST);
    EXPECT_EQ(CompareResult<float_t>(output0, expect_output0_7, out_data_num), true);
    EXPECT_EQ(CompareResult<int32_t>(output1, expect_output1_7, out_data_num), true);
}


vector<int64_t> list_out_8 = {2};
vector<vector<int64_t>> shapes_8 = {{1, 4, 4}, {1, 2, 2}};
float_t input_8[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
float_t expect_output0_8[] = {6, 8, 14, 16};
int32_t expect_output1_8[] = {5, 7, 13, 15};
TEST_F(TEST_ADAPTER_MAX_POOL2D_UT, TestAdaptiveMaxPool2d_dapter_max_pool2d_float_failed_1) {
    int32_t out_data_num = sizeof(expect_output0_8)/sizeof(expect_output0_8[0]);
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT8};
    float_t output0[out_data_num] = {(float_t)0};
    int8_t output1[out_data_num] = {(int8_t)0};
    CREATE_NODEDEF(shapes_8, data_types, input_8, output0, output1, list_out_8);
    RUN_KERNEL(node_def, HOST);
}


vector<int64_t> list_out_9 = {2};
vector<vector<int64_t>> shapes_9 = {{1, 4, 4}, {1, 2, 2}};
float_t expect_output0_9[] = {6, 8, 14, 16};
int32_t expect_output1_9[] = {5, 7, 13, 15};
TEST_F(TEST_ADAPTER_MAX_POOL2D_UT, TestAdaptiveMaxPool2d_dapter_max_pool2d_float_failed_2) {
    int32_t out_data_num = sizeof(expect_output0_9)/sizeof(expect_output0_9[0]);
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32};
    float_t output0[out_data_num] = {(float_t)0};
    int32_t output1[out_data_num] = {(int32_t)0};
    CREATE_NODEDEF(shapes_9, data_types, nullptr, output0, output1, list_out_9);
    RUN_KERNEL(node_def, HOST);
}


vector<vector<int64_t>> shapes_10 = {{1, 4, 4}, {1, 2, 2}};
float_t input_10[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
float_t expect_output0_10[] = {6, 8, 14, 16};
int32_t expect_output1_10[] = {5, 7, 13, 15};
TEST_F(TEST_ADAPTER_MAX_POOL2D_UT, TestAdaptiveMaxPool2d_dapter_max_pool2d_float_failed_3) {
    int32_t out_data_num = sizeof(expect_output0_10)/sizeof(expect_output0_10[0]);
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32};
    float_t output0[out_data_num] = {(float_t)0};
    int32_t output1[out_data_num] = {(int32_t)0};
    CREATE_NODEDEF(shapes_10, data_types, input_10, output0, output1, nullptr);
    RUN_KERNEL(node_def, HOST);
}

vector<int64_t> list_out_11 = {2};
vector<vector<int64_t>> shapes_11 = {{1,1,1, 4, 4}, {1, 2, 2}};
float_t input_11[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
float_t expect_output0_11[] = {6, 8, 14, 16};
int32_t expect_output1_11[] = {5, 7, 13, 15};
TEST_F(TEST_ADAPTER_MAX_POOL2D_UT, TestAdaptiveMaxPool2d_dapter_max_pool2d_float_failed_4) {
    int32_t out_data_num = sizeof(expect_output0_11)/sizeof(expect_output0_11[0]);
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32};
    float_t output0[out_data_num] = {(float_t)0};
    int32_t output1[out_data_num] = {(int32_t)0};
    CREATE_NODEDEF(shapes_11, data_types, input_11, output0, output1, list_out_11);
    RUN_KERNEL(node_def, HOST);
}
