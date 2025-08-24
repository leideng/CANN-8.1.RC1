/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#ifndef AICPU_CONTEXT_CUST_OP_CUST_DLOG_RECORD_H_
#define AICPU_CONTEXT_CUST_OP_CUST_DLOG_RECORD_H_

#include <dlfcn.h>

#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

#include "cpu_context.h"
#include "cust_cpu_utils.h"
#include "toolchain/slog.h"

namespace aicpu {
class CustCpuKernelDlogUtils {
 public:
  static CustCpuKernelDlogUtils &GetInstance();
  int32_t CustSetCpuKernelContext(uint64_t workspace_size, uint64_t workspace_addr);
  int64_t GetTid();
  std::shared_ptr<CpuKernelContext> GetCpuKernelContext();
  const std::unordered_map<int, std::string> module_to_string_map_;

 private:
  CustCpuKernelDlogUtils()
      : module_to_string_map_({{SLOG, "SLOG"},
                               {IDEDD, "IDEDD"},
                               {HCCL, "HCCL"},
                               {FMK, "FMK"},
                               {DVPP, "DVPP"},
                               {RUNTIME, "RUNTIME"},
                               {CCE, "CCE"},
                               {HDC, "HDC"},
                               {DRV, "DRV"},
                               {DEVMM, "DEVMM"},
                               {KERNEL, "KERNEL"},
                               {LIBMEDIA, "LIBMEDIA"},
                               {CCECPU, "CCECPU"},
                               {ROS, "ROS"},
                               {HCCP, "HCCP"},
                               {ROCE, "ROCE"},
                               {TEFUSION, "TEFUSION"},
                               {PROFILING, "PROFILING"},
                               {DP, "DP"},
                               {APP, "APP"},
                               {TS, "TS"},
                               {TSDUMP, "TSDUMP"},
                               {AICPU, "AICPU"},
                               {LP, "LP"},
                               {TDT, "TDT"},
                               {FE, "FE"},
                               {MD, "MD"},
                               {MB, "MB"},
                               {ME, "ME"},
                               {IMU, "IMU"},
                               {IMP, "IMP"},
                               {GE, "GE"},
                               {CAMERA, "CAMERA"},
                               {ASCENDCL, "ASCENDCL"},
                               {TEEOS, "TEEOS"},
                               {ISP, "ISP"},
                               {SIS, "SIS"},
                               {HSM, "HSM"},
                               {DSS, "DSS"},
                               {PROCMGR, "PROCMGR"},
                               {BBOX, "BBOX"},
                               {AIVECTOR, "AIVECTOR"},
                               {TBE, "TBE"},
                               {FV, "FV"},
                               {TUNE, "TUNE"},
                               {HSS, "HSS"},
                               {FFTS, "FFTS"},
                               {OP, "OP"},
                               {UDF, "UDF"},
                               {HICAID, "HICAID"},
                               {TSYNC, "TSYNC"},
                               {AUDIO, "AUDIO"},
                               {TPRT, "TPRT"},
                               {ASCENDCKERNEL, "ASCENDCKERNEL"},
                               {ASYS, "ASYS"},
                               {ATRACE, "ATRACE"},
                               {RTC, "RTC"},
                               {SYSMONITOR, "SYSMONITOR"},
                               {AML, "AML"},
                               {ADETECT, "ADETECT"},
                               {INVLID_MOUDLE_ID, "INVLID_MOUDLE_ID"}}) {};
  ~CustCpuKernelDlogUtils();
  CustCpuKernelDlogUtils(const CustCpuKernelDlogUtils &) = delete;
  CustCpuKernelDlogUtils(CustCpuKernelDlogUtils &&) = delete;
  CustCpuKernelDlogUtils &operator=(const CustCpuKernelDlogUtils &) = delete;
  CustCpuKernelDlogUtils &operator=(CustCpuKernelDlogUtils &&) = delete;
  std::unordered_map<int64_t, std::shared_ptr<CpuKernelContext>> ctx_map_;
  std::mutex ctx_mutex_;
};
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) int32_t CustSetCpuKernelContext(uint64_t workspace_size,
                                                                       uint64_t workspace_addr);
}

#endif  // AICPU_CONTEXT_CUST_OP_CUST_DLOG_RECORD_H_