add_library(host_intf_pub INTERFACE)

target_compile_options(host_intf_pub INTERFACE
    -fPIC
    #-fvisibility=hidden
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-O0>
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++11 -fvisibility-inlines-hidden>
    $<$<COMPILE_LANGUAGE:C>:-pthread -Wfloat-equal -Wshadow -Wformat=2 -Wno-deprecated -Wextra>
    $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
)

target_compile_definitions(host_intf_pub INTERFACE
    _GLIBCXX_USE_CXX11_ABI=0
    $<$<CONFIG:Release>:_FORTIFY_SOURCE=2>
)

target_include_directories(host_intf_pub INTERFACE
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

target_link_options(host_intf_pub INTERFACE
    $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-pie>
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    $<$<CONFIG:Release>:-s>
)

target_link_directories(host_intf_pub INTERFACE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

execute_process(
    COMMAND ${ASCEND_PYTHON_EXECUTABLE} extract_static_library.py -a ${ASCENDC_RUNTIME} -t ${ASCENDC_RUNTIME_OBJ_TARGET} -d ${CMAKE_BINARY_DIR}  -o ${ASCENDC_RUNTIME_CONFIG}
    WORKING_DIRECTORY ${ASCENDC_KERNEL_CMAKE_DIR}/util
    RESULT_VARIABLE result
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )

include(${CMAKE_BINARY_DIR}/${ASCENDC_RUNTIME_CONFIG})

add_library(${ASCENDC_RUNTIME_STATIC_TARGET} STATIC IMPORTED)
set_target_properties(${ASCENDC_RUNTIME_STATIC_TARGET} PROPERTIES
    IMPORTED_LOCATION    "${ASCENDC_RUNTIME}"
    )



