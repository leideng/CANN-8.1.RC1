/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
 * \file inner_kernel_operator_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_INTERFACE_H

#include "inner_kernel_operator_data_copy_intf.cppm"
#include "inner_kernel_operator_dump_tensor_intf.cppm"
#include "inner_kernel_operator_mm_intf.cppm"
#include "inner_kernel_operator_gemm_intf.cppm"
#include "inner_kernel_operator_fixpipe_intf.cppm"
#include "inner_kernel_operator_conv2d_intf.cppm"
#include "inner_kernel_operator_common_intf.cppm"
#include "inner_kernel_operator_proposal_intf.cppm"
#include "inner_kernel_operator_vec_bilinearinterpalation_intf.cppm"
#include "inner_kernel_operator_vec_createvecindex_intf.cppm"
#include "inner_kernel_operator_vec_mulcast_intf.cppm"
#include "inner_kernel_operator_determine_compute_sync_intf.cppm"
#include "inner_kernel_operator_vec_transpose_intf.cppm"
#include "inner_kernel_operator_vec_gather_intf.cppm"
#include "inner_kernel_operator_vec_scatter_intf.cppm"
#include "inner_kernel_operator_vec_brcb_intf.cppm"
#include "inner_kernel_operator_vec_binary_intf.cppm"
#include "inner_kernel_operator_vec_binary_scalar_intf.cppm"
#include "inner_kernel_operator_vec_cmpsel_intf.cppm"
#include "inner_kernel_operator_vec_duplicate_intf.cppm"
#include "inner_kernel_operator_vec_reduce_intf.cppm"
#include "inner_kernel_operator_vec_gather_mask_intf.cppm"
#include "inner_kernel_operator_vec_ternary_scalar_intf.cppm"
#include "inner_kernel_operator_vec_unary_intf.cppm"
#include "inner_kernel_operator_vec_vconv_intf.cppm"
#include "inner_kernel_operator_vec_vpadding_intf.cppm"
#include "inner_kernel_operator_scalar_intf.cppm"
#include "inner_kernel_operator_sys_var_intf.cppm"
#include "inner_kernel_operator_set_atomic_intf.cppm"
#include "inner_kernel_prof_trace_intf.cppm"

#endif // ASCENDC_MODULE_INNER_OPERATOR_INTERFACE_H
