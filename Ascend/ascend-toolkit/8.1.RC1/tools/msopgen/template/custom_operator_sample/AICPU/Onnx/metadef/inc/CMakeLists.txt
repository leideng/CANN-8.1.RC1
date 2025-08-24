# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


add_library(metadef_headers INTERFACE)
target_include_directories(metadef_headers INTERFACE
    $<BUILD_INTERFACE:${METADEF_DIR}>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/common>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/exe_graph>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/external>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/external/exe_graph>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/external/op_common>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/graph>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/graph/utils>
    $<BUILD_INTERFACE:${METADEF_DIR}/inc/register>

    # 下列头文件包含路径是非法的，需要在后续整改中删掉
    # --------------------start------------------------
    $<BUILD_INTERFACE:${METADEF_DIR}/third_party/transformer>
    $<BUILD_INTERFACE:${METADEF_DIR}/third_party/transformer/inc>
    $<BUILD_INTERFACE:${METADEF_DIR}/exe_graph>
    $<BUILD_INTERFACE:${METADEF_DIR}/graph>
    $<BUILD_INTERFACE:${METADEF_DIR}/graph/debug>
    $<BUILD_INTERFACE:${METADEF_DIR}/register>
    $<BUILD_INTERFACE:${METADEF_DIR}/register/op_tiling>
    # ---------------------end-----------------------
    
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/metadef>
    $<INSTALL_INTERFACE:include/metadef/common>
    $<INSTALL_INTERFACE:include/metadef/exe_graph>
    $<INSTALL_INTERFACE:include/metadef/external>
    $<INSTALL_INTERFACE:include/metadef/external/exe_graph>
    $<INSTALL_INTERFACE:include/metadef/external/op_common>
    $<INSTALL_INTERFACE:include/metadef/graph>
    $<INSTALL_INTERFACE:include/metadef/graph/utils>
    $<INSTALL_INTERFACE:include/metadef/register>
    $<INSTALL_INTERFACE:include/metadef/transformer>
)
