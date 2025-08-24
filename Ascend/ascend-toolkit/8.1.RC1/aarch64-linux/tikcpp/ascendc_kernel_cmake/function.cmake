function(merge_device_obj)
    cmake_parse_arguments(MERGE "FATBIN" "TARGET_NAME;NORMAL_DIR;AIC_DIR;AIV_DIR;OUT_DIR;OUT_NAME" "DEPENDS" ${ARGN})

    if(MERGE_FATBIN)
        set(FATBIN_OPTION --fatbin)
        set(_TARGET_NAME ${MERGE_TARGET_NAME}_fatbin_obj)
    else()
        set(FATBIN_OPTION)
        set(_TARGET_NAME ${MERGE_TARGET_NAME}_device_obj)
    endif()

    add_custom_target(${_TARGET_NAME} ALL
        COMMAND ${ASCEND_PYTHON_EXECUTABLE} merge_device_obj.py
            -l ${CCEC_LINKER}
            -o ${MERGE_OUT_DIR}
            -n ${MERGE_OUT_NAME}
            --normal-dir ${MERGE_NORMAL_DIR}
            --aic-dir ${MERGE_AIC_DIR}
            --aiv-dir ${MERGE_AIV_DIR}
            --build-type ${CMAKE_BUILD_TYPE}
            --script ${ASCENDC_KERNEL_CMAKE_DIR}/util/merge_obj.sh
            ${FATBIN_OPTION}
        WORKING_DIRECTORY ${ASCENDC_KERNEL_CMAKE_DIR}/util
        COMMAND_EXPAND_LISTS
    )

    if(MERGE_DEPENDS)
        add_dependencies(${_TARGET_NAME} ${MERGE_DEPENDS})
    endif()
endfunction()

function(ascendc_library target_name target_type)
    string(TOLOWER ${target_type} _lower_target_type)
    string(TOUPPER ${target_type} _upper_target_type)

    set(support_types SHARED STATIC OBJECT)
    if(NOT _upper_target_type IN_LIST support_types)
        message(FATAL_ERROR "target_type ${target_type} does not support, the support list is ${support_types}")
    endif()

    set(device_target ${target_name})
    set(${device_target}_auto_gen_dir     ${CMAKE_BINARY_DIR}/auto_gen/${target_name})
    set(${device_target}_include_dir      ${CMAKE_BINARY_DIR}/include/${target_name})
    set(${device_target}_merge_obj_dir    ${CMAKE_CURRENT_BINARY_DIR}/${device_target}_merge_obj_dir)
    set(${target_name}_host_dir   ${CMAKE_CURRENT_BINARY_DIR}/${target_name}_host_dir)
    set(${device_target}_obj_install_dir  ${CMAKE_INSTALL_PREFIX}/single_objects/${target_name})

    set(SOURCES)
    foreach(_source ${ARGN})
        get_filename_component(absolute_source "${_source}" ABSOLUTE)
        list(APPEND SOURCES ${absolute_source})
    endforeach()

    string(REPLACE ";" "::" EP_SOURCES "${SOURCES}")

    # Transfer attribute information of target_compile_options/target_compile_definitions/target_include_directories
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${target_name}_stub.cpp PROPERTIES GENERATED TRUE)
    add_library(${target_name}_interface STATIC EXCLUDE_FROM_ALL
        ${CMAKE_CURRENT_BINARY_DIR}/${target_name}_stub.cpp
    )

    ExternalProject_Add(${device_target}_precompile
                        SOURCE_DIR ${ASCENDC_KERNEL_CMAKE_DIR}/device_precompile_project
                        CONFIGURE_COMMAND  ${CMAKE_COMMAND}
                            -G ${CMAKE_GENERATOR}
                            -DASCENDC_KERNEL_CMAKE_DIR=${ASCENDC_KERNEL_CMAKE_DIR}
                            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                            -DSOURCES=${EP_SOURCES}
                            -DOPTIONS=$<TARGET_PROPERTY:${target_name}_interface,OPTIONS>
                            -DDEFINITIONS=$<TARGET_PROPERTY:${target_name}_interface,DEFINITIONS>
                            -DINCLUDES=$<TARGET_PROPERTY:${target_name}_interface,INCLUDES>
                            -DDYNAMIC_MODE=${DYNAMIC_MODE}
                            -DBUILD_MODE=${BUILD_MODE}
                            -DDST_DIR=${${device_target}_auto_gen_dir}
                            -DINCLUDE_DIR=${${device_target}_include_dir}
                            -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
                            -DASCEND_PYTHON_EXECUTABLE=${ASCEND_PYTHON_EXECUTABLE}
                            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                            -DTARGET_TYPE=${_upper_target_type}
                            <SOURCE_DIR>
                        INSTALL_COMMAND ""
                        LIST_SEPARATOR ::
                        BUILD_ALWAYS TRUE
                        )

    ExternalProject_Add(${device_target}_preprocess
                        SOURCE_DIR ${ASCENDC_KERNEL_CMAKE_DIR}/device_preprocess_project
                        CONFIGURE_COMMAND  ${CMAKE_COMMAND}
                            -G ${CMAKE_GENERATOR}
                            -DASCENDC_KERNEL_CMAKE_DIR=${ASCENDC_KERNEL_CMAKE_DIR}
                            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                            -DSOURCES=${EP_SOURCES}
                            -DOPTIONS=$<TARGET_PROPERTY:${target_name}_interface,OPTIONS>
                            -DDEFINITIONS=$<TARGET_PROPERTY:${target_name}_interface,DEFINITIONS>
                            -DINCLUDES=$<TARGET_PROPERTY:${target_name}_interface,INCLUDES>
                            -DDYNAMIC_MODE=${DYNAMIC_MODE}
                            -DBUILD_MODE=${BUILD_MODE}
                            -DDST_DIR=${${device_target}_auto_gen_dir}
                            -DINCLUDE_DIR=${${device_target}_include_dir}
                            -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
                            -DASCEND_PYTHON_EXECUTABLE=${ASCEND_PYTHON_EXECUTABLE}
                            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                            -DTARGET_TYPE=${_upper_target_type}
                            <SOURCE_DIR>
                        INSTALL_COMMAND ""
                        LIST_SEPARATOR ::
                        BUILD_ALWAYS TRUE
                        )
    add_dependencies(${device_target}_preprocess ${device_target}_precompile)

    # Merge the device-side obj files to generate device.o
    if(DYNAMIC_MODE)
        set(${device_target}_aic_device_dir ${CMAKE_CURRENT_BINARY_DIR}/${device_target}_aic_device_dir)
        set(AIC_DIR ${${device_target}_aic_device_dir} PARENT_SCOPE)
        ExternalProject_Add(${device_target}_aic_device
                            SOURCE_DIR ${ASCENDC_KERNEL_CMAKE_DIR}/device_project
                            CONFIGURE_COMMAND  ${CMAKE_COMMAND}
                                -G ${CMAKE_GENERATOR}
                                -DBUILD_MODE=aic
                                -DASCENDC_KERNEL_CMAKE_DIR=${ASCENDC_KERNEL_CMAKE_DIR}
                                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                                -DOPTIONS=$<TARGET_PROPERTY:${target_name}_interface,OPTIONS>
                                -DDEFINITIONS=$<TARGET_PROPERTY:${target_name}_interface,DEFINITIONS>
                                -DINCLUDES=$<TARGET_PROPERTY:${target_name}_interface,INCLUDES>
                                -DDST_DIR=${${device_target}_aic_device_dir}
                                -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
                                -DBUILD_CFG=${${device_target}_auto_gen_dir}/aic_config.cmake
                                -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                                -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                                <SOURCE_DIR>
                            INSTALL_COMMAND ""
                            LIST_SEPARATOR ::
                            BUILD_ALWAYS TRUE
                            DEPENDS ${device_target}_preprocess
                            )

        set(${device_target}_aiv_device_dir ${CMAKE_CURRENT_BINARY_DIR}/${device_target}_aiv_device_dir)
        set(AIV_DIR ${${device_target}_aiv_device_dir} PARENT_SCOPE)
        ExternalProject_Add(${device_target}_aiv_device
                            SOURCE_DIR ${ASCENDC_KERNEL_CMAKE_DIR}/device_project
                            CONFIGURE_COMMAND  ${CMAKE_COMMAND}
                                -G ${CMAKE_GENERATOR}
                                -DBUILD_MODE=aiv
                                -DASCENDC_KERNEL_CMAKE_DIR=${ASCENDC_KERNEL_CMAKE_DIR}
                                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                                -DOPTIONS=$<TARGET_PROPERTY:${target_name}_interface,OPTIONS>
                                -DDEFINITIONS=$<TARGET_PROPERTY:${target_name}_interface,DEFINITIONS>
                                -DINCLUDES=$<TARGET_PROPERTY:${target_name}_interface,INCLUDES>
                                -DDST_DIR=${${device_target}_aiv_device_dir}
                                -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
                                -DBUILD_CFG=${${device_target}_auto_gen_dir}/aiv_config.cmake
                                -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                                -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                                <SOURCE_DIR>
                            INSTALL_COMMAND ""
                            LIST_SEPARATOR ::
                            BUILD_ALWAYS TRUE
                            DEPENDS ${device_target}_preprocess
                            )

        if("${_upper_target_type}" STREQUAL "OBJECT")
            merge_device_obj(
                TARGET_NAME ${device_target}
                AIC_DIR     ${${device_target}_aic_device_dir}
                AIV_DIR     ${${device_target}_aiv_device_dir}
                OUT_DIR     ${${device_target}_obj_install_dir}
                DEPENDS
                    ${device_target}_aic_device
                    ${device_target}_aiv_device
            )
        else()
            add_custom_target(${device_target}_merge_obj ALL
                COMMAND bash merge_mix_obj.sh -l ${CCEC_LINKER} -o ${${device_target}_merge_obj_dir} --aiv-dir ${${device_target}_aiv_device_dir} --aic-dir ${${device_target}_aic_device_dir} --build-type ${CMAKE_BUILD_TYPE}
                COMMAND ${ASCEND_PYTHON_EXECUTABLE} update_host_stub.py ${${device_target}_auto_gen_dir}  ${${device_target}_merge_obj_dir} ${_LOWER_SOC_VERSION} ${device_target}
                WORKING_DIRECTORY ${ASCENDC_KERNEL_CMAKE_DIR}/util
                DEPENDS ${device_target}_aic_device ${device_target}_aiv_device
                COMMAND_EXPAND_LISTS
            )
        endif()
    elseif(BUILD_MODE STREQUAL "m200" AND NOT "${_upper_target_type}" STREQUAL "OBJECT")
        set(${device_target}_aic_device_dir ${CMAKE_CURRENT_BINARY_DIR}/${device_target}_aic_device_dir)
        set(AIC_DIR ${${device_target}_aic_device_dir} PARENT_SCOPE)
        ExternalProject_Add(${device_target}_aic_device
                            SOURCE_DIR ${ASCENDC_KERNEL_CMAKE_DIR}/device_project
                            CONFIGURE_COMMAND  ${CMAKE_COMMAND}
                                -G ${CMAKE_GENERATOR}
                                -DBUILD_MODE=${BUILD_MODE}
                                -DASCENDC_KERNEL_CMAKE_DIR=${ASCENDC_KERNEL_CMAKE_DIR}
                                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                                -DOPTIONS=$<TARGET_PROPERTY:${target_name}_interface,OPTIONS>
                                -DDEFINITIONS=$<TARGET_PROPERTY:${target_name}_interface,DEFINITIONS>
                                -DINCLUDES=$<TARGET_PROPERTY:${target_name}_interface,INCLUDES>
                                -DDST_DIR=${${device_target}_aic_device_dir}
                                -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
                                -DBUILD_CFG=${${device_target}_auto_gen_dir}/aic_config.cmake
                                -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                                -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                                <SOURCE_DIR>
                            INSTALL_COMMAND ""
                            LIST_SEPARATOR ::
                            BUILD_ALWAYS TRUE
                            DEPENDS ${device_target}_preprocess
                            )

        set(${device_target}_aiv_device_dir ${CMAKE_CURRENT_BINARY_DIR}/${device_target}_aiv_device_dir)
        set(AIV_DIR ${${device_target}_aiv_device_dir} PARENT_SCOPE)
        ExternalProject_Add(${device_target}_aiv_device
                            SOURCE_DIR ${ASCENDC_KERNEL_CMAKE_DIR}/device_project
                            CONFIGURE_COMMAND  ${CMAKE_COMMAND}
                                -G ${CMAKE_GENERATOR}
                                -DBUILD_MODE=${BUILD_MODE}_vec
                                -DASCENDC_KERNEL_CMAKE_DIR=${ASCENDC_KERNEL_CMAKE_DIR}
                                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                                -DOPTIONS=$<TARGET_PROPERTY:${target_name}_interface,OPTIONS>
                                -DDEFINITIONS=$<TARGET_PROPERTY:${target_name}_interface,DEFINITIONS>
                                -DINCLUDES=$<TARGET_PROPERTY:${target_name}_interface,INCLUDES>
                                -DDST_DIR=${${device_target}_aiv_device_dir}
                                -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
                                -DBUILD_CFG=${${device_target}_auto_gen_dir}/aiv_config.cmake
                                -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                                -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                                <SOURCE_DIR>
                            INSTALL_COMMAND ""
                            LIST_SEPARATOR ::
                            BUILD_ALWAYS TRUE
                            DEPENDS ${device_target}_preprocess
                            )
        add_custom_target(${device_target}_merge_obj ALL
            COMMAND bash merge_mix_obj.sh -l ${CCEC_LINKER} -o ${${device_target}_merge_obj_dir} --aiv-dir ${${device_target}_aiv_device_dir} --aic-dir ${${device_target}_aic_device_dir} --build-type ${CMAKE_BUILD_TYPE}
            COMMAND ${ASCEND_PYTHON_EXECUTABLE} update_host_stub.py ${${device_target}_auto_gen_dir}  ${${device_target}_merge_obj_dir} ${_LOWER_SOC_VERSION} ${device_target}
            WORKING_DIRECTORY ${ASCENDC_KERNEL_CMAKE_DIR}/util
            DEPENDS ${device_target}_aic_device ${device_target}_aiv_device
            COMMAND_EXPAND_LISTS
        )
    else()
        set(NORMAL_DIR ${${device_target}_merge_obj_dir} PARENT_SCOPE)
        ExternalProject_Add(${device_target}_device
                            SOURCE_DIR ${ASCENDC_KERNEL_CMAKE_DIR}/device_project
                            CONFIGURE_COMMAND  ${CMAKE_COMMAND}
                                -G ${CMAKE_GENERATOR}
                                -DBUILD_MODE=${BUILD_MODE}
                                -DASCENDC_KERNEL_CMAKE_DIR=${ASCENDC_KERNEL_CMAKE_DIR}
                                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                                -DOPTIONS=$<TARGET_PROPERTY:${target_name}_interface,OPTIONS>
                                -DDEFINITIONS=$<TARGET_PROPERTY:${target_name}_interface,DEFINITIONS>
                                -DINCLUDES=$<TARGET_PROPERTY:${target_name}_interface,INCLUDES>
                                -DDST_DIR=${${device_target}_merge_obj_dir}
                                -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
                                -DBUILD_CFG=${${device_target}_auto_gen_dir}/normal_config.cmake
                                -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                                -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                                -DENABLE_MERGE=ON
                                <SOURCE_DIR>
                            INSTALL_COMMAND ""
                            LIST_SEPARATOR ::
                            BUILD_ALWAYS TRUE
                            DEPENDS ${device_target}_preprocess
                            )

        if("${_upper_target_type}" STREQUAL "OBJECT")
            merge_device_obj(
                TARGET_NAME ${device_target}
                NORMAL_DIR  ${${device_target}_merge_obj_dir}
                OUT_DIR     ${${device_target}_obj_install_dir}
                DEPENDS     ${device_target}_device
            )
        else()
            add_custom_target(${device_target}_merge_obj ALL
                COMMAND ${ASCEND_PYTHON_EXECUTABLE} update_host_stub.py ${${device_target}_auto_gen_dir}  ${${device_target}_merge_obj_dir} ${_LOWER_SOC_VERSION} ${device_target}
                WORKING_DIRECTORY ${ASCENDC_KERNEL_CMAKE_DIR}/util
                DEPENDS ${device_target}_device
            )
        endif()
    endif()

    if("${_upper_target_type}" STREQUAL "OBJECT")
        return()
    endif()

    # compile host bisheng
    if(NOT ASCEND_KERNEL_LAUNCH_ONLY)
        ExternalProject_Add(${target_name}_host
                            SOURCE_DIR ${ASCENDC_KERNEL_CMAKE_DIR}/host_project
                            CONFIGURE_COMMAND  ${CMAKE_COMMAND}
                                -G ${CMAKE_GENERATOR}
                                -DASCENDC_KERNEL_CMAKE_DIR=${ASCENDC_KERNEL_CMAKE_DIR}
                                -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
                                -DASCEND_PYTHON_EXECUTABLE=${ASCEND_PYTHON_EXECUTABLE}
                                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                                -DSOURCES=${EP_SOURCES}
                                -DINCLUDES=$<TARGET_PROPERTY:${target_name}_interface,INCLUDES>
                                -DCMAKE_INSTALL_PREFIX=${${target_name}_host_dir}
                                -DINCLUDE_DIR=${${device_target}_include_dir}
                                -DASCENDC_HOST_COMPILE_OPTIONS=$<TARGET_PROPERTY:${target_name}_interface,HOST_COMPILE_OPTIONS>
                                -DBUILD_CFG=${${device_target}_auto_gen_dir}/host_config.cmake
                                -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                                -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                                <SOURCE_DIR>
                            LIST_SEPARATOR ::
                            BUILD_ALWAYS TRUE
                            )
        add_dependencies(${target_name}_host ${device_target}_preprocess)
    endif()

    # Generate host-side bin file based on device.o and host_stub.cpp
    set_source_files_properties(${${device_target}_auto_gen_dir}/host_stub.cpp PROPERTIES GENERATED TRUE)
    add_library(${device_target}_host_stub_obj OBJECT
        ${${device_target}_auto_gen_dir}/host_stub.cpp
    )

    target_link_libraries(${device_target}_host_stub_obj PRIVATE
        $<BUILD_INTERFACE:host_intf_pub>
    )

    target_compile_definitions(${device_target}_host_stub_obj PRIVATE
        $<$<BOOL:$<IN_LIST:ASCENDC_DUMP=0,$<TARGET_PROPERTY:${target_name}_interface,DEFINITIONS>>>:ASCENDC_DUMP=0>
        $<$<BOOL:$<IN_LIST:-DASCENDC_DUMP=0,$<TARGET_PROPERTY:${target_name}_interface,DEFINITIONS>>>:ASCENDC_DUMP=0>
    )

    add_dependencies(${device_target}_host_stub_obj ${device_target}_preprocess ${device_target}_merge_obj)

    add_library(${target_name} ${_upper_target_type}
        $<TARGET_OBJECTS:${device_target}_host_stub_obj>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,STATIC_LIBRARY>:$<TARGET_OBJECTS:${ASCENDC_RUNTIME_OBJ_TARGET}>>
    )

    target_include_directories(${target_name}
        INTERFACE
            ${${device_target}_include_dir}
            ${ASCEND_CANN_PACKAGE_PATH}/include
            ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/tikcfw/
    )

    target_link_directories(${target_name} PUBLIC
        ${ASCEND_CANN_PACKAGE_PATH}/lib64
        ${ASCEND_CANN_PACKAGE_PATH}/tools/simulator/${SOC_VERSION}/lib
    )

    target_link_libraries(${target_name}
        PRIVATE
            $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${ASCENDC_RUNTIME_STATIC_TARGET}>
            $<BUILD_INTERFACE:host_intf_pub>
        INTERFACE
            $<BUILD_INTERFACE:$<$<AND:$<STREQUAL:${RUN_MODE},sim>,$<STREQUAL:${BUILD_MODE},c100>>:pem_davinci>>
            $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
            $<BUILD_INTERFACE:$<$<AND:$<STREQUAL:${RUN_MODE},sim>,$<STREQUAL:${DYNAMIC_MODE},ON>>:npu_drv>>
            ascendcl
            $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
            register
            error_manager
            profapi
            ge_common_base
            ascendalog
            mmpa
            dl
        PUBLIC
            ascend_dump
            c_sec
    )

    set_target_properties(${target_name} PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )

    add_custom_command(TARGET ${target_name}
        PRE_LINK
        COMMAND bash ${ASCENDC_KERNEL_CMAKE_DIR}/util/ascendc_pack_kernel.sh --pack_tool ${ASCENDC_PACK_KERNEL} --elf_in $<TARGET_OBJECTS:${device_target}_host_stub_obj> --add_dir ${${device_target}_merge_obj_dir}
    )

    if(NOT ASCEND_KERNEL_LAUNCH_ONLY)
        add_dependencies(${target_name} ${target_name}_host)
        
        add_custom_command(TARGET ${target_name}
            POST_BUILD
            COMMAND rm -f $<TARGET_FILE:${target_name}>
            COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${ASCENDC_KERNEL_CMAKE_DIR}/util/recompile_binary.py --root-dir ${CMAKE_CURRENT_BINARY_DIR} --target-name ${target_name} --add-dir ${${target_name}_host_dir}
        )
    endif()

    include(GNUInstallDirs)
    install(TARGETS ${target_name}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(DIRECTORY ${CMAKE_BINARY_DIR}/include/
        DESTINATION include
    )
endfunction()

function(ascendc_fatbin_library target_name)
    set(${target_name}_fatbin_install_dir ${CMAKE_INSTALL_PREFIX}/fatbin/${target_name})

    # NORMAL_DIR/AIC_DIR/AIV_DIR Definition comes from ascendc_library function
    ascendc_library(${target_name} OBJECT
        ${ARGN}
    )

    if(DYNAMIC_MODE)
        merge_device_obj(
            TARGET_NAME ${target_name}
            AIC_DIR     ${AIC_DIR}
            AIV_DIR     ${AIV_DIR}
            OUT_DIR     ${${target_name}_fatbin_install_dir}
            OUT_NAME    ${target_name}.o
            DEPENDS     ${target_name}_device_obj
            FATBIN
        )
    else()
        merge_device_obj(
            TARGET_NAME ${target_name}
            NORMAL_DIR  ${NORMAL_DIR}
            OUT_DIR     ${${target_name}_fatbin_install_dir}
            OUT_NAME    ${target_name}.o
            DEPENDS     ${target_name}_device_obj
            FATBIN
        )
    endif()
endfunction()

function(ascendc_compile_options target_name target_scope)
    if(ARGN)
        set(CUSTOM_KERNEL_OPTIONS_LIST)
        set(HOST_COMPILE_OPTIONS_LIST)
        set(FIND_HOST_COMPILE_OPTIONS_FLAG OFF)
        # split compile options by "-forward-options-to-host-compiler",
        # which after this options are belong to host compile options
        foreach(arg ${ARGN})
            if (FIND_HOST_COMPILE_OPTIONS_FLAG)
                list(APPEND HOST_COMPILE_OPTIONS_LIST ${arg})
            else()
                if (${arg} STREQUAL "-forward-options-to-host-compiler")
                    set(FIND_HOST_COMPILE_OPTIONS_FLAG ON)
                else()
                    list(APPEND CUSTOM_KERNEL_OPTIONS_LIST ${arg})
                endif()
            endif()
        endforeach()
        get_target_property(CUSTOM_OPTIONS ${target_name}_interface OPTIONS)
        if(CUSTOM_OPTIONS)
            list(APPEND CUSTOM_OPTIONS ${CUSTOM_KERNEL_OPTIONS_LIST})
        else()
            set(CUSTOM_OPTIONS ${CUSTOM_KERNEL_OPTIONS_LIST})
        endif()
        set_target_properties(${target_name}_interface PROPERTIES OPTIONS "${CUSTOM_OPTIONS}")
        set_target_properties(${target_name}_interface PROPERTIES HOST_COMPILE_OPTIONS "${HOST_COMPILE_OPTIONS_LIST}")
        target_compile_options(host_intf_pub INTERFACE "${HOST_COMPILE_OPTIONS_LIST}")
    endif()
endfunction()

function(ascendc_compile_definitions target_name target_scope)
    if(ARGN)
        get_target_property(CUSTOM_DEFINITIONS ${target_name}_interface DEFINITIONS)
        if(CUSTOM_DEFINITIONS)
            list(APPEND CUSTOM_DEFINITIONS ${ARGN})
        else()
            set(CUSTOM_DEFINITIONS ${ARGN})
        endif()

        set_target_properties(${target_name}_interface PROPERTIES DEFINITIONS "${CUSTOM_DEFINITIONS}")
    endif()
endfunction()

function(ascendc_include_directories target_name target_scope)
    if(ARGN)
        set(ALL_INCLUDES)
        foreach(_include ${ARGN})
            get_filename_component(absolute_include "${_include}" ABSOLUTE)
            list(APPEND ALL_INCLUDES ${absolute_include})
        endforeach()

        get_target_property(CUSTOM_INCLUDES ${target_name}_interface INCLUDES)
        if(CUSTOM_INCLUDES)
            list(APPEND CUSTOM_INCLUDES ${ALL_INCLUDES})
        else()
            set(CUSTOM_INCLUDES ${ALL_INCLUDES})
        endif()

        set_target_properties(${target_name}_interface PROPERTIES INCLUDES "${CUSTOM_INCLUDES}")
    endif()
endfunction()
