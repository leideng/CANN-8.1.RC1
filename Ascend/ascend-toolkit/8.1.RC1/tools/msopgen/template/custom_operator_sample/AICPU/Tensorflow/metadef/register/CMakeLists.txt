# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

include(${METADEF_DIR}/cmake/build_type.cmake)
set(SRC_LIST
    "register.cpp"
    "register_custom_pass.cpp"
    "prototype_pass_registry.cc"
    "ops_kernel_builder_registry.cc"
    "graph_optimizer/fusion_common/op_slice_info.cc"
    "graph_optimizer/fusion_common/fusion_pass_desc.cc"
    "graph_optimizer/fusion_common/fusion_turbo_utils.cc"
    "graph_optimizer/fusion_common/fusion_turbo.cc"
    "graph_optimizer/fusion_common/unknown_shape_utils.cc"
    "graph_optimizer/fusion_common/fusion_config_info.cc"
    "graph_optimizer/graph_fusion/graph_fusion_pass_base.cc"
    "graph_optimizer/graph_fusion/connection_matrix.cc"
    "graph_optimizer/graph_fusion/fusion_pass_registry.cc"
    "graph_optimizer/graph_fusion/fusion_pattern.cc"
    "graph_optimizer/graph_fusion/pattern_fusion_base_pass.cc"
    "graph_optimizer/graph_fusion/graph_pass_util.cc"
    "graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.cc"
    "graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.h"
    "graph_optimizer/graph_fusion/fusion_quant_util.cc"
    "graph_optimizer/graph_fusion/fusion_quant_util_impl.cc"
    "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.cc"
    "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.cc"
    "graph_optimizer/buffer_fusion/buffer_fusion_pattern.cc"
    "graph_optimizer/fusion_statistic/fusion_statistic_recorder.cc"
    "op_kernel_registry.cpp"
    "auto_mapping_util.cpp"
    "ffts_plus_update_manager.cc"
    "ffts_node_converter_registry.cc"
    "ffts_node_calculater_registry.cc"
    "op_ext_calc_param_registry.cc"
    "op_ext_gentask_registry.cc"
    "host_cpu_context.cc"
    "hidden_input_func_registry.cc"
    "hidden_inputs_func_registry.cc"
    "tensor_assign.cpp"
    "infer_data_slice_registry.cc"
    "infer_axis_slice_registry.cc"
    "kernel_register_data.cc"
    "kernel_registry_impl.cc"
    "node_converter_registry.cc"
    "op_binary_resource_manager.cc"
    "exe_res_generation_context.cc"
    "op_impl_registry.cc"
    "op_lib_register.cc"
    "register_base.cc"
    "op_ct_impl_registry.cc"
    "op_impl_space_registry.cc"
    "op_impl_registry_holder_manager.cc"
    "scope/scope_graph.cc"
    "scope/scope_pass.cc"
    "scope/scope_pattern.cc"
    "scope/scope_util.cc"
    "scope/scope_pass_registry.cc"
    "shape_inference.cc"
    "ascendc/ascendc_py.cc"
    "ascendc/op_check.cc"
    "ascendc/tilingdata_base.cc"
    "opdef/op_def.cc"
    "opdef/op_def_attr.cc"
    "opdef/op_def_param.cc"
    "opdef/op_def_aicore.cc"
    "opdef/op_def_factory.cc"
    "opdef/op_def_mc2.cc"
    "device_op_impl_registry.cc"
    "tuning_tiling_registry.cc"
    "tuning_bank_key_registry.cc"
    "stream_manage_func_registry.cc"
    "optimization_option_registry.cc"
    "pass_option_utils.cc"
    "opp_so_manager.cc"
    "${METADEF_DIR}/exe_graph/lowering/kernel_run_context_builder.cc"
    "${METADEF_DIR}/exe_graph/lowering/bg_kernel_context_extend.cc"
    "${METADEF_DIR}/exe_graph/lowering/buffer_pool.cc"
    "${METADEF_DIR}/exe_graph/runtime/compute_node_info.cc"
    "${METADEF_DIR}/exe_graph/lowering/bg_ir_attrs.cc"
)

############ libregister.so ############
add_library(register SHARED
    ${SRC_LIST}
    $<TARGET_OBJECTS:metadef_tensorflow_protos_obj>
    $<TARGET_OBJECTS:metadef_aicpu_protos_obj>
)

target_compile_options(register PRIVATE
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:/utf-8>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

target_compile_definitions(register PRIVATE
    google=ascend_private
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_include_directories(register PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${METADEF_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
)

target_link_options(register PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(register
    PRIVATE
        intf_pub
        msprof_headers
        cce_headers
        runtime_headers
        -Wl,--whole-archive
        op_tiling_o2
        -Wl,--no-whole-archive
        -Wl,--no-as-needed
        ascend_protobuf
        c_sec
        slog
        platform
        graph
        static_mmpa
        -Wl,--as-needed
        json
    PUBLIC
        metadef_headers
)

############ libregister.a ############
add_library(register_static STATIC
    ${SRC_LIST}
    $<TARGET_OBJECTS:metadef_tensorflow_protos_obj>
    $<TARGET_OBJECTS:metadef_aicpu_protos_obj>
    "op_tiling/op_tiling.cc"
    "op_tiling/op_tiling_info.cc"
    "op_tiling/op_tiling_utils.cc"
    "op_tiling/op_tiling_attr_utils.cc"
    "op_tiling/op_compile_info_manager.cc"
    "op_tiling/op_tiling_registry.cc"
    "op_tiling/op_tiling_py.cc"
)

target_compile_options(register_static PRIVATE
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:/utf-8>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

target_compile_definitions(register_static PRIVATE
    google=ascend_private
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    LOG_CPP
)

target_include_directories(register_static PUBLIC
    ${METADEF_DIR}/graph
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/register
)

target_include_directories(register_static PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${METADEF_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
)

target_link_libraries(register_static PRIVATE
    c_sec
    json
    intf_pub
    slog_headers
    msprof_headers
    mmpa_headers
    cce_headers
    runtime_headers
    metadef_headers
)

target_link_libraries(register_static PRIVATE
        ascend_protobuf_static
        )

set_target_properties(register_static PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    OUTPUT_NAME $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,libregister,register>
)

############ libop_tiling_o2.a ############
add_library(op_tiling_o2 STATIC
    "op_tiling/op_tiling.cc"
    "op_tiling/op_tiling_info.cc"
    "op_tiling/op_tiling_utils.cc"
    "op_tiling/op_tiling_attr_utils.cc"
    "op_tiling/op_compile_info_manager.cc"
    "op_tiling/op_tiling_registry.cc"
    "op_tiling/op_tiling_py.cc")

add_dependencies(op_tiling_o2
    metadef_protos
)

target_include_directories(op_tiling_o2 PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${METADEF_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
)

target_compile_options(op_tiling_o2 PRIVATE
    -O2
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
)

target_compile_definitions(op_tiling_o2 PRIVATE
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    LOG_CPP
)

target_link_libraries(op_tiling_o2 PRIVATE
    intf_pub
    slog_headers
    mmpa_headers
    metadef_headers
    ascend_protobuf
    json
    c_sec
)

############ librt2_registry.a ############
add_library(rt2_registry_objects OBJECT
    "op_impl_registry.cc"
    "op_ct_impl_registry.cc"
    "op_impl_functions.cc"
)

target_compile_options(rt2_registry_objects PRIVATE
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fvisibility=hidden -fno-common -fPIC -O2 -Werror -Wextra -Wfloat-equal>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:/utf-8>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

target_compile_definitions(rt2_registry_objects PRIVATE
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_include_directories(rt2_registry_objects PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/third_party/graphengine/inc
    ${METADEF_DIR}/third_party/graphengine/inc/external
    ${METADEF_DIR}/third_party/fwkacllib/inc
)

set_target_properties(rt2_registry_objects PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    OUTPUT_NAME $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,librt2_registry,rt2_registry>
)

target_link_libraries(rt2_registry_objects
    PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    c_sec
    slog
    PUBLIC
    metadef_headers
)

############ librt2_registry.a ############
add_library(rt2_registry_static STATIC
    $<TARGET_OBJECTS:rt2_registry_objects>
)

set_target_properties(rt2_registry_static PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    OUTPUT_NAME $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,librt2_registry,rt2_registry>
)

target_link_libraries(rt2_registry_static
    PUBLIC
    metadef_headers
)

##############################################################
set(STUB_HEADER_LIST
    ${METADEF_DIR}/inc/external/register/op_def.h
    ${METADEF_DIR}/inc/external/register/op_def_factory.h
    ${METADEF_DIR}/inc/external/register/op_impl_registry.h
    ${METADEF_DIR}/inc/external/register/op_lib_register.h
    ${METADEF_DIR}/inc/external/register/register_base.h
    ${METADEF_DIR}/inc/external/register/op_tiling_info.h
    ${METADEF_DIR}/inc/external/register/op_tiling_registry.h
    ${METADEF_DIR}/inc/external/register/register.h
    ${METADEF_DIR}/inc/external/register/tilingdata_base.h
    ${METADEF_DIR}/inc/external/register/tuning_bank_key_registry.h
    ${METADEF_DIR}/inc/external/register/tuning_tiling_registry.h
    ${METADEF_DIR}/inc/external/register/hidden_input_func_registry.h
    ${METADEF_DIR}/inc/external/register/hidden_inputs_func_registry.h
    ${METADEF_DIR}/inc/register/op_tiling.h
    ${METADEF_DIR}/inc/register/op_impl_registry_base.h
    ${METADEF_DIR}/inc/register/op_impl_space_registry.h
    ${METADEF_DIR}/inc/register/stream_manage_func_registry.h
    ${METADEF_DIR}/inc/register/op_impl_registry_holder_manager.h
    ${METADEF_DIR}/inc/register/optimization_option_registry.h
    ${METADEF_DIR}/inc/register/opp_so_manager.h
    ${METADEF_DIR}/inc/register/op_lib_register_impl.h
)

list(TRANSFORM STUB_HEADER_LIST
    REPLACE "^.*/([^/]+)\\.h$" "${CMAKE_CURRENT_BINARY_DIR}/stub_\\1.cc"
    OUTPUT_VARIABLE STUB_SRC_LIST
)

add_custom_command(
    OUTPUT ${STUB_SRC_LIST}
    COMMAND echo "Generating stub files."
            && ${HI_PYTHON} ${METADEF_DIR}/tests/stub/gen_stubapi.py ${CMAKE_CURRENT_BINARY_DIR} ${STUB_HEADER_LIST}
            && echo "Generating stub files end."
)

add_custom_target(register_stub DEPENDS ${STUB_SRC_LIST})

############ stub/libregister.so ############
add_library(stub_register SHARED ${STUB_SRC_LIST})

add_dependencies(stub_register metadef_protos register_stub)

target_include_directories(stub_register PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
)

target_compile_options(stub_register PRIVATE
    -Wfloat-equal
    -fno-common
    -Os
)

target_link_libraries(stub_register
    PRIVATE
        intf_pub
        slog_headers
        msprof_headers
        runtime_headers
        c_sec_headers
        ascend_protobuf
        c_sec
        slog
        json
    PUBLIC
        metadef_headers
)

set_target_properties(stub_register PROPERTIES
    OUTPUT_NAME register
    LIBRARY_OUTPUT_DIRECTORY stub
)


############ stub/libregister.a ############
if (NOT ENABLE_OPEN_SRC)
    target_clone(stub_register stub_register_static STATIC)

    add_dependencies(stub_register_static metadef_protos register_stub)

    target_compile_options(stub_register_static PRIVATE -ffunction-sections -fdata-sections)

    set_target_properties(stub_register_static PROPERTIES
        OUTPUT_NAME register
        ARCHIVE_OUTPUT_DIRECTORY stub
    )
endif ()


############ install ############
install(TARGETS register_static
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
)

install(TARGETS rt2_registry_static
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR}/${CMAKE_SYSTEM_PROCESSOR} OPTIONAL
)

install(TARGETS stub_register OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}/stub
)
