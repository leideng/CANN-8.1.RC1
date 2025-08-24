/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
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

#include "kernel_util.h"

#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <algorithm>
#include "securec.h"

namespace aicpu {
namespace {
const std::map<Format, std::string> kFormatToStringMap = {
    {FORMAT_NCHW, "NCHW"},
    {FORMAT_NHWC, "NHWC"},
    {FORMAT_ND, "ND"},
    {FORMAT_NC1HWC0, "NC1HWC0"},
    {FORMAT_FRACTAL_Z, "FRACTAL_Z"},
    {FORMAT_NC1C0HWPAD, "NC1C0HWPAD"},
    {FORMAT_NHWC1C0, "NHWC1C0"},
    {FORMAT_FSR_NCHW, "FSR_NCHW"},
    {FORMAT_FRACTAL_DECONV, "FRACTAL_DECONV"},
    {FORMAT_C1HWNC0, "C1HWNC0"},
    {FORMAT_FRACTAL_DECONV_TRANSPOSE, "FRACTAL_DECONV_TRANSPOSE"},
    {FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS, "FRACTAL_DECONV_SP_STRIDE_TRANS"},
    {FORMAT_NC1HWC0_C04, "NC1HWC0_C04"},
    {FORMAT_FRACTAL_Z_C04, "FRACTAL_Z_C04"},
    {FORMAT_CHWN, "CHWN"},
    {FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS, "DECONV_SP_STRIDE8_TRANS"},
    {FORMAT_NC1KHKWHWC0, "NC1KHKWHWC0"},
    {FORMAT_BN_WEIGHT, "BN_WEIGHT"},
    {FORMAT_FILTER_HWCK, "FILTER_HWCK"},
    {FORMAT_HWCN, "HWCN"},
    {FORMAT_HASHTABLE_LOOKUP_LOOKUPS, "LOOKUP_LOOKUPS"},
    {FORMAT_HASHTABLE_LOOKUP_KEYS, "LOOKUP_KEYS"},
    {FORMAT_HASHTABLE_LOOKUP_VALUE, "LOOKUP_VALUE"},
    {FORMAT_HASHTABLE_LOOKUP_OUTPUT, "LOOKUP_OUTPUT"},
    {FORMAT_HASHTABLE_LOOKUP_HITS, "LOOKUP_HITS"},
    {FORMAT_MD, "MD"},
    {FORMAT_NDHWC, "NDHWC"},
    {FORMAT_NCDHW, "NCDHW"},
    {FORMAT_DHWCN, "DHWCN"},
    {FORMAT_DHWNC, "DHWNC"},
    {FORMAT_NDC1HWC0, "NDC1HWC0"},
    {FORMAT_FRACTAL_Z_3D, "FRACTAL_Z_3D"},
    {FORMAT_FRACTAL_Z_3D_TRANSPOSE, "FRACTAL_Z_3D_TRANSPOSE"},
    {FORMAT_C1HWNCoC0, "C1HWNCoC0"},
    {FORMAT_FRACTAL_NZ, "FRACTAL_NZ"},
    {FORMAT_CN, "CN"},
    {FORMAT_NC, "NC"},
    {FORMAT_FRACTAL_ZN_LSTM, "FRACTAL_ZN_LSTM"},
    {FORMAT_FRACTAL_Z_G, "FRACTAL_Z_G"},
    {FORMAT_RESERVED, "FORMAT_RESERVED"},
    {FORMAT_ALL, "ALL"},
    {FORMAT_NULL, "NULL"},
    {FORMAT_FRACTAL_Z_WINO, "FORMAT_FRACTAL_Z_WINO"},
    {FORMAT_C1HWC0, "C1HWC0"}};
}

std::string FormatToSerialString(Format format) {
  auto it =
      kFormatToStringMap.find(static_cast<Format>(GetPrimaryFormat(static_cast<int32_t>(format))));
  if (it != kFormatToStringMap.end()) {
    if (HasSubFormat(static_cast<int32_t>(format))) {
      return it->second + ":" + std::to_string(GetSubFormat(static_cast<int32_t>(format)));
    }
    return it->second;
  } else {
    KERNEL_LOG_ERROR("Format not support [%u]", format);
    return "UNDEFINED";
  }
}

const std::map<std::string, DataType> dtype_maps {
    {"DT_FLOAT", DT_FLOAT},
    {"DT_FLOAT16", DT_FLOAT16},
    {"DT_BFLOAT16", DT_BFLOAT16},
    {"DT_INT8", DT_INT8},
    {"DT_INT16", DT_INT16},
    {"DT_UINT16", DT_UINT16},
    {"DT_UINT8", DT_UINT8},
    {"DT_INT32", DT_INT32},
    {"DT_INT64", DT_INT64},
    {"DT_UINT32", DT_UINT32},
    {"DT_UINT64", DT_UINT64},
    {"DT_BOOL", DT_BOOL},
    {"DT_DOUBLE", DT_DOUBLE},
    {"DT_STRING", DT_STRING},
    {"DT_DUAL_SUB_INT8", DT_DUAL_SUB_INT8},
    {"DT_DUAL_SUB_UINT8", DT_DUAL_SUB_UINT8},
    {"DT_COMPLEX32", DT_COMPLEX32},
    {"DT_COMPLEX64", DT_COMPLEX64},
    {"DT_COMPLEX128", DT_COMPLEX128},
    {"DT_QINT8", DT_QINT8},
    {"DT_QINT16", DT_QINT16},
    {"DT_QINT32", DT_QINT32},
    {"DT_QUINT8", DT_QUINT8},
    {"DT_QUINT16", DT_QUINT16},
    {"DT_RESOURCE", DT_RESOURCE},
    {"DT_STRING_REF", DT_STRING_REF},
    {"DT_DUAL", DT_DUAL},
    {"DT_UINT1", DT_UINT1},
    {"DT_HIFLOAT8", DT_HIFLOAT8},
    {"DT_UNDEFINED", DT_UNDEFINED}};

bool IsEmptyTensor(Tensor *tensor) {
  auto dims = tensor->GetTensorShape()->GetDimSizes();
  if (tensor->GetData() == nullptr) {
    for (uint32_t i = 0; i < dims.size(); i++) {
      if (dims[i] == 0) {
        return true;
      }
    }
  }
  return false;
}

uint32_t NormalMathCheck(CpuKernelContext &ctx) {
  const uint32_t kInputNum = 2;
  const uint32_t kOutputNum = 1;

  if ((ctx.GetInputsSize() != kInputNum) ||
      (ctx.GetOutputsSize() != kOutputNum)) {
    KERNEL_LOG_ERROR("[%s] Input size or Output size is unexpected,"
                     "expected input size [%u], real input size [%u],"
                     "expected output size [%u], real output size [%u]",
                     ctx.GetOpType().c_str(),
                     kInputNum, ctx.GetInputsSize(),
                     kOutputNum, ctx.GetOutputsSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  Tensor *input_0 = ctx.Input(kFirstInputIndex);
  KERNEL_CHECK_NULLPTR(input_0, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] Get input[0] failed", ctx.GetOpType().c_str());
  Tensor *input_1 = ctx.Input(kSecondInputIndex);
  KERNEL_CHECK_NULLPTR(input_1, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] Get input[1] failed", ctx.GetOpType().c_str());

  if (input_0->GetDataType() != input_1->GetDataType()) {
    KERNEL_LOG_WARN(
        "[%s] dtype of inputs not matched, input[0] data_type is [%s], "
        "input[1] data_type is [%s]",
        ctx.GetOpType().c_str(), DTypeStr(input_0->GetDataType()).c_str(),
        DTypeStr(input_1->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  Tensor *output = ctx.Output(kFirstOutputIndex);
  KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get output failed", ctx.GetOpType().c_str());
  return KERNEL_STATUS_OK;
}

uint32_t NormalCheck(CpuKernelContext &ctx, const uint32_t inputs_num,
                     const uint32_t outputs_num) {
  if (static_cast<int32_t>(inputs_num) != kDynamicInput) {
    KERNEL_CHECK_FALSE(
        (ctx.GetInputsSize() >= inputs_num), KERNEL_STATUS_PARAM_INVALID,
        "[%s] need [%u] inputs, but got [%u].", ctx.GetOpType().c_str(),
        inputs_num, ctx.GetInputsSize());
    for (uint32_t i = 0; i < inputs_num; ++i) {
      Tensor *input = ctx.Input(i);
      KERNEL_CHECK_NULLPTR(input, KERNEL_STATUS_INNER_ERROR,
                           "[%s] get input[%u] failed.",
                           ctx.GetOpType().c_str(), i);
      auto input_shape = input->GetTensorShape();
      KERNEL_CHECK_NULLPTR(input_shape, KERNEL_STATUS_PARAM_INVALID,
                           "%s input[%u] tensor shape is nullptr.",
                           ctx.GetOpType().c_str(), i);
      if (!IsEmptyTensor(input)) {
        auto input_data = input->GetData();
        KERNEL_CHECK_NULLPTR(input_data, KERNEL_STATUS_PARAM_INVALID,
                             "%s input[%u] tensor data is nullptr.",
                             ctx.GetOpType().c_str(), i);
      }
    }
  }

  if (static_cast<int32_t>(outputs_num) != kDynamicOutput) {
    KERNEL_CHECK_FALSE(
        (ctx.GetOutputsSize() == outputs_num), KERNEL_STATUS_PARAM_INVALID,
        "[%s] need [%u] outputs, but got [%u].", ctx.GetOpType().c_str(),
        outputs_num, ctx.GetOutputsSize());
    for (uint32_t i = 0; i < outputs_num; ++i) {
      Tensor *output = ctx.Output(i);
      KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_INNER_ERROR,
                           "[%s] get output[%u] failed.",
                           ctx.GetOpType().c_str(), i);
      auto output_shape = output->GetTensorShape();
      KERNEL_CHECK_NULLPTR(output_shape, KERNEL_STATUS_PARAM_INVALID,
                           "%s output[%u] tensor shape is nullptr.",
                           ctx.GetOpType().c_str(), i);
      if (!IsEmptyTensor(output)) {
        auto output_data = output->GetData();
        KERNEL_CHECK_NULLPTR(output_data, KERNEL_STATUS_PARAM_INVALID,
                             "%s output[%u] tensor data is nullptr.",
                             ctx.GetOpType().c_str(), i);
      }
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t NormalCheck(CpuKernelContext &ctx, const uint32_t inputs_num,
                     const uint32_t outputs_num,
                     const std::vector<std::string> &attr_names) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, inputs_num, outputs_num),
                      "Check Greater params failed.");
  for (auto const &attr_name : attr_names) {
    auto attr = ctx.GetAttr(attr_name);
    KERNEL_CHECK_NULLPTR(attr, KERNEL_STATUS_PARAM_INVALID,
                         "%s get attr[%s] is nullptr.",
                         ctx.GetOpType().c_str(), attr_name.c_str());
  }
  return KERNEL_STATUS_OK;
}

bool IsScalar(const std::vector<int64_t> &shape) {
  return (shape.size() == 0);
}

bool IsVector(const std::vector<int64_t> &shape) {
  return (shape.size() == 1);
}

bool IsMatrix(const std::vector<int64_t> &shape) {
  return (shape.size() == 2);
}

bool IsSquareMatrix(const std::vector<int64_t> &shape) {
  return ((shape.size() == 2) && (shape[0] == shape[1]));
}

bool AddrAlignedCheck(const void *addr, uint64_t alignment) {
  return reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(addr)) %
             alignment ==
         0;
}

bool IsVectorOrHigher(const std::vector<int64_t> &shape) {
  return (shape.size() >= 1);
}

DataType DType(std::string dtype_str) {
  auto iter = dtype_maps.find(dtype_str);
  if (iter != dtype_maps.end()) {
    return iter->second;
  } else {
    return DT_UNDEFINED;
  }
}

std::string DTypeStr(DataType dtype) {
  auto iter = std::find_if(
      dtype_maps.begin(), dtype_maps.end(),
      [dtype](const std::map<std::string, DataType>::value_type &kv) {
        return (kv.second == dtype);
      });
  if (iter != dtype_maps.end()) {
    return iter->first;
  } else {
    return std::string("DT_UNDEFINED");
  }
}

inline uint64_t PtrToValue(const void *const ptr) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
}

inline void *ValueToPtr(const uint64_t value) {
  return reinterpret_cast<void *>(static_cast<uintptr_t>(value));
}

bool BiggerMemCpy(void *dst_addr, const std::size_t dst_len,
                  const void *src_addr, const std::size_t src_len) {
  if ((dst_addr == nullptr) || (src_addr == nullptr)) {
    KERNEL_LOG_ERROR("BiggerMemCpy input param is null.");
    return false;
  }

  if (dst_len < src_len) {
    KERNEL_LOG_ERROR("BiggerMemCpy dst_len is[%zu] less than src_len[%zu].",
                     dst_len, src_len);
    return false;
  }

  std::size_t remain_size = src_len;
  while (remain_size > SECUREC_MEM_MAX_LEN) {
    if (memcpy_s(dst_addr, SECUREC_MEM_MAX_LEN, src_addr,
                 SECUREC_MEM_MAX_LEN) != EOK) {
      KERNEL_LOG_ERROR("BiggerMemCpy memcpy_s failed.");
      return false;
    }
    remain_size -= SECUREC_MEM_MAX_LEN;
    src_addr = ValueToPtr(PtrToValue(src_addr) + SECUREC_MEM_MAX_LEN);
    dst_addr = ValueToPtr(PtrToValue(dst_addr) + SECUREC_MEM_MAX_LEN);
  }
  if ((remain_size != 0U) &&
      (memcpy_s(dst_addr, remain_size, src_addr, remain_size) != EOK)) {
    KERNEL_LOG_ERROR("BiggerMemCpy memcpy_s remain size failed.");
    return false;
  }
  return true;
}

bool BiggerMemSet(void *dst_addr, const std::size_t dst_len, const int c,
                  const std::size_t count) {
  if (dst_addr == nullptr) {
    KERNEL_LOG_ERROR("BiggerMemSet dst_addr is null.");
    return false;
  }

  if (dst_len < count) {
    KERNEL_LOG_ERROR("BiggerMemSet dst_len is[%zu] less than count[%zu].",
                     dst_len, count);
    return false;
  }

  std::size_t remain_size = count;
  while (remain_size > SECUREC_MEM_MAX_LEN) {
    if (memset_s(dst_addr, SECUREC_MEM_MAX_LEN, c, SECUREC_MEM_MAX_LEN) !=
        EOK) {
      KERNEL_LOG_ERROR("BiggerMemSet memset_s failed.");
      return false;
    }
    remain_size -= SECUREC_MEM_MAX_LEN;
    dst_addr = ValueToPtr(PtrToValue(dst_addr) + SECUREC_MEM_MAX_LEN);
  }
  if ((remain_size != 0U) &&
      (memset_s(dst_addr, remain_size, c, remain_size) != EOK)) {
    KERNEL_LOG_ERROR("BiggerMemSet memset_s remain size failed.");
    return false;
  }
  return true;
}

int64_t CeilMultiple(const int64_t x, const int64_t base) {
  int64_t ret = x / base;
  if ((x % base) != 0) {
    ret++;
  }
  return ret;
}

int64_t GetMaxCompilerCoreNum() {
  int64_t max_compiler_core_number = 1;
  const char* kMaxCompilerCoreNum = "MAX_COMPILE_CORE_NUMBER";
  const char *value = getenv(kMaxCompilerCoreNum);
  if ((value != nullptr) && (value[0U] != '\0')) {
    max_compiler_core_number =
        std::strtol(&(value[0U]), nullptr, 10);  // 10 is for 10进制
    max_compiler_core_number = max_compiler_core_number > get_nprocs() ? get_nprocs() : max_compiler_core_number;
  }
  return max_compiler_core_number;
}

uint64_t CalFileSize(const std::string &file_path) {
  struct stat st = {};
  auto ret = stat(file_path.c_str(), &st);
  if (ret != 0) {
    return 0UL;
  }
  return st.st_size;
}

// Align size to the nearest multiple of alignment
uint64_t AlignedTo(const uint64_t size, const uint64_t alignment) {
  return (size + alignment -1) & (~(alignment - 1));
}
}  // namespace aicpu
