# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
aux_source_directory(. SRCS)
message(STATUS "SRCS = ${SRCS}")

if("x${SRCS}" STREQUAL "x")
    add_custom_target(${ONNX_PLUGIN_TARGET}
            COMMAND mkdir -p ${ONNX_PLUGIN_TARGET_OUT_DIR}
            COMMAND echo "no source to make lib ${ONNX_PLUGIN_TARGET}.so")
    return(0)
endif()

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/metadef/inc)
include_directories(${CMAKE_SOURCE_DIR}/metadef)
include_directories(${CMAKE_SOURCE_DIR}/metadef/inc/external/graph)
include_directories(${CMAKE_SOURCE_DIR}/log)
include_directories(${CMAKE_SOURCE_DIR}/op_proto/inc)

set(LIBRARY_OUTPUT_PATH ${ONNX_PLUGIN_TARGET_OUT_DIR})

add_library(${ONNX_PLUGIN_TARGET} SHARED ${SRCS})

target_compile_definitions(${ONNX_PLUGIN_TARGET} PRIVATE
    google=ascend_private
)

target_link_libraries(${ONNX_PLUGIN_TARGET} ${ASCEND_INC}/../lib64/libgraph.so)