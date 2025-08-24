# 递归收集目标的所有动态库依赖路径
function(get_all_dynamic_dirs target result_var)
    set(dirs "")
    # 获取目标的直接依赖库
    get_target_property(libs ${target} LINK_LIBRARIES)
    foreach (lib IN LISTS libs)
        # 仅处理 CMake 目标（排除系统库和绝对路径）
        if (TARGET ${lib})
            get_target_property(type ${lib} TYPE)
            # 处理动态库（SHARED_LIBRARY）
            if (type STREQUAL "SHARED_LIBRARY")
                # 获取动态库的输出目录
                get_target_property(lib_output_dir ${lib} LIBRARY_OUTPUT_DIRECTORY)
                if (NOT lib_output_dir)
                    set(lib_output_dir $<TARGET_FILE_DIR:${lib}>)
                endif ()
                list(APPEND dirs ${lib_output_dir})
                # 递归处理依赖
                get_all_dynamic_dirs(${lib} child_dirs)
                list(APPEND dirs ${child_dirs})
            endif ()
        endif ()
    endforeach ()
    list(REMOVE_DUPLICATES dirs)
    set(${result_var} ${dirs} PARENT_SCOPE)
endfunction()

function(ascir_generate depend_so_target bin_dir so_var h_var)
    # 1. 收集生成器可执行程序的所有动态库依赖路径
    get_all_dynamic_dirs(ascir_ops_header_generator lib_dirs)

    # 2. 合并依赖库路径（避免重复）
    list(REMOVE_DUPLICATES lib_dirs)

    # 3. 转换为冒号分隔的路径字符串
    string(REPLACE ";" ":" lib_path_str "${lib_dirs}")

    # 4. 添加自定义命令，设置 LD_LIBRARY_PATH
    add_custom_command(
            OUTPUT ${h_var}
            DEPENDS ${depend_so_target} ascir_ops_header_generator
            COMMAND ${CMAKE_COMMAND} -E env
            "LD_LIBRARY_PATH=${lib_path_str}:$ENV{LD_LIBRARY_PATH}"
            ${bin_dir}/ascir_ops_header_generator ${so_var} ${h_var}
            VERBATIM
            COMMENT "Generating header ${h_var} with all dependencies"
    )
endfunction()