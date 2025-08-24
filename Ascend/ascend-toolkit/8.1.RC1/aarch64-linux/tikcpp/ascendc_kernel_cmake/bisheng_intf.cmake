add_library(device_intf_pub INTERFACE)

target_compile_options(device_intf_pub INTERFACE
    -O3
    -std=c++17
)

set(CANN_VERSION_HEADER ${ASCENDC_DEVKIT_PATH}/../include/version/cann_version.h)
if(EXISTS ${CANN_VERSION_HEADER})
    target_compile_options(device_intf_pub INTERFACE
        "SHELL:-include ${CANN_VERSION_HEADER}"
    )
endif()

target_compile_definitions(device_intf_pub INTERFACE
    TILING_KEY_VAR=0
)

target_include_directories(device_intf_pub INTERFACE
    ${ASCENDC_DEVKIT_PATH}/tikcpp/tikcfw
    ${ASCENDC_DEVKIT_PATH}/tikcpp/tikcfw/interface
    ${ASCENDC_DEVKIT_PATH}/tikcpp/tikcfw/impl
)

add_library(m300_intf_pub INTERFACE)

target_compile_options(m300_intf_pub INTERFACE
    --cce-aicore-arch=dav-m300
    --cce-aicore-only
    --cce-auto-sync
    --cce-mask-opt
    "SHELL:-mllvm -cce-aicore-function-stack-size=16000"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-or-combine=false"
    "SHELL:-mllvm -instcombine-code-sinking=false"
    "SHELL:-mllvm -cce-aicore-jump-expand=false"
    "SHELL:-mllvm -cce-aicore-mask-opt=false"
)

target_link_libraries(m300_intf_pub INTERFACE
    $<BUILD_INTERFACE:device_intf_pub>
)

add_library(aiv_intf_pub INTERFACE)

target_compile_options(aiv_intf_pub INTERFACE
    --cce-aicore-arch=dav-c220-vec
    --cce-aicore-only
    --cce-auto-sync
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

target_link_libraries(aiv_intf_pub INTERFACE
    $<BUILD_INTERFACE:device_intf_pub>
)

add_library(aic_intf_pub INTERFACE)

target_compile_options(aic_intf_pub INTERFACE
    --cce-aicore-arch=dav-c220-cube
    --cce-aicore-only
    --cce-auto-sync
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

target_link_libraries(aic_intf_pub INTERFACE
    $<BUILD_INTERFACE:device_intf_pub>
)

add_library(m200_intf_pub INTERFACE)

target_compile_options(m200_intf_pub INTERFACE
    --cce-aicore-arch=dav-m200
    --cce-aicore-only
    --cce-auto-sync
    --cce-mask-opt
    "SHELL:-mllvm -cce-aicore-fp-ceiling=2"
    "SHELL:-mllvm -cce-aicore-record-overflow=false"
    "SHELL:-mllvm -cce-aicore-mask-opt=false"
)

target_link_libraries(m200_intf_pub INTERFACE
    $<BUILD_INTERFACE:device_intf_pub>
)

add_library(m200_vec_intf_pub INTERFACE)

target_compile_options(m200_vec_intf_pub INTERFACE
    --cce-aicore-arch=dav-m200-vec
    --cce-aicore-only
    --cce-auto-sync
    --cce-mask-opt
    "SHELL:-mllvm -cce-aicore-fp-ceiling=2"
    "SHELL:-mllvm -cce-aicore-record-overflow=false"
    "SHELL:-mllvm -cce-aicore-mask-opt=false"
    "SHELL:-D__ENABLE_VECTOR_CORE__"
)

target_link_libraries(m200_vec_intf_pub INTERFACE
    $<BUILD_INTERFACE:device_intf_pub>
)

add_library(c100_intf_pub INTERFACE)

target_compile_options(c100_intf_pub INTERFACE
    --cce-aicore-arch=dav-c100
    --cce-aicore-only
    --cce-auto-sync
    --cce-mask-opt
    "SHELL:-mllvm -cce-aicore-function-stack-size=16000"
    "SHELL:-mllvm -cce-aicore-record-overflow=false"
    "SHELL:-mllvm -cce-aicore-jump-expand=false"
    "SHELL:-mllvm -cce-aicore-mask-opt=false"
)

target_link_libraries(c100_intf_pub INTERFACE
    $<BUILD_INTERFACE:device_intf_pub>
)
