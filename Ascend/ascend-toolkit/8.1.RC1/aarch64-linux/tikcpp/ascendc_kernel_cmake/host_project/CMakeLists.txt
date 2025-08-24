cmake_minimum_required(VERSION 3.16.0)
project(host_bisheng)

include(${ASCENDC_KERNEL_CMAKE_DIR}/bisheng_config.cmake)
include(${ASCENDC_KERNEL_CMAKE_DIR}/bisheng_intf.cmake)

include(${BUILD_CFG})

add_library(host_bisheng_obj OBJECT
    ${SOURCES}
)

target_compile_options(host_bisheng_obj PRIVATE
    "SHELL:-include ${INCLUDE_DIR}/aclrtlaunch_triple_chevrons_func.h"
)

target_link_libraries(host_bisheng_obj PRIVATE
    $<BUILD_INTERFACE:device_intf_pub>
    $<BUILD_INTERFACE:intf_host_bisheng>
)

install(TARGETS host_bisheng_obj
    DESTINATION .
)

# Public compile rules of device
add_library(intf_host_bisheng INTERFACE)

target_compile_options(intf_host_bisheng INTERFACE
    ${ASCENDC_HOST_COMPILE_OPTIONS}
    -fPIC
    --cce-host-only
    -fcce-kernel-launch-custom
    ${OPTIONS}
)

target_include_directories(intf_host_bisheng INTERFACE 
    ${INCLUDES}
)

target_compile_definitions(intf_host_bisheng INTERFACE
    ${DEFINITIONS}
)

