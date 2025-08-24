#!/usr/bin/env fish
# Perform setenv for compiler package
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

set -g curpath (realpath (dirname (status --current-filename)))
set -g curfile (realpath (status --current-filename))
set -g param_mult_ver $argv[1]

function get_install_param
    set -l _key $argv[1]
    set -l _file $argv[2]
    if not test -f "$_file"
        exit 1
    end
    set -l install_info_key_array Compiler_Install_Type Compiler_Feature_Type Compiler_UserName Compiler_UserGroup Compiler_Install_Path_Param Compiler_Arch_Linux_Path Compiler_Hetero_Arch_Flag
    for key_param in $install_info_key_array
        if test "$key_param" = "$_key"
            grep -i "$_key=" "$_file" | cut --only-delimited -d"=" -f2-
            break
        end
    end
end

function get_install_dir
    set -l install_info "$curpath/../ascend_install.info"
    set -l hetero_arch (get_install_param "Compiler_Hetero_Arch_Flag" "$install_info")
    if test "$param_mult_ver" = "multi_version"
        if test "$hetero_arch" = "y"
            echo (realpath $curpath/../../../../../latest)/compiler
        else
            echo (realpath $curpath/../../../latest)/compiler
        end
    else
        echo (realpath $curpath/..)
    end
end

set -l INSTALL_DIR (get_install_dir)
set -l lib_path "$INSTALL_DIR/python/site-packages/"
if test -d "$lib_path"
    set -l python_path "$PYTHONPATH"
    set -l num (echo ":$python_path:" | grep ":$lib_path:" | wc -l)
    if test "$num" -eq 0
        if test "-$python_path" = "-"
            set -gx PYTHONPATH "$lib_path"
        else
            set -gx PYTHONPATH "$lib_path:$python_path"
        end
    end
end

if test -d "$INSTALL_DIR/ccec_compiler/bin" -a -d "$INSTALL_DIR/bin"
    for p_path in $INSTALL_DIR/bin $INSTALL_DIR/ccec_compiler/bin
        set -l num (echo :$PATH: | grep :$p_path: | wc -l)
        if test $num = 0
            set -gx PATH $p_path $PATH
        end
    end
end

set -l library_path "$INSTALL_DIR/lib64"
if test -d "$library_path"
    set -l ld_library_path "$LD_LIBRARY_PATH"
    set -l num (echo ":$ld_library_path:" | grep ":$library_path:" | wc -l)
    if test "$num" -eq 0
        if test "-$ld_library_path" = "-"
            set -gx LD_LIBRARY_PATH "$library_path:$library_path/plugin/opskernel:$library_path/plugin/nnengine:$library_path/stub"
        else
            set -gx LD_LIBRARY_PATH "$library_path:$library_path/plugin/opskernel:$library_path/plugin/nnengine:$ld_library_path:$library_path/stub"
        end
    end
end

set -l custom_path_file "$INSTALL_DIR/../conf/path.cfg"
set -l common_interface "$INSTALL_DIR/script/common_interface.fish"
set -l owner (stat -c \%U "$curfile")
if test (id -u) -ne 0 -a (id -un) != "$owner" -a -f "$custom_path_file" -a -f "$common_interface"
    . "$common_interface"
    mk_custom_path "$custom_path_file"
    for dir_name in "conf" "data"
        set -l dst_dir (grep -w "$dir_name" "$custom_path_file" | cut --only-delimited -d"=" -f2-)
        set -l dst_dir (eval echo "$dst_dir")
        if test -d "$INSTALL_DIR/$dir_name" -a -d "$dst_dir"
            chmod -R u+w $dst_dir/* > /dev/null 2>&1
            cp -rfL $INSTALL_DIR/$dir_name/* "$dst_dir"
        end
    end
end
