#!/usr/bin/env fish
set -x PACKAGE toolkit
set -x CUR_DIR (realpath (dirname (status --current-filename)))

function get_install_dir
    set -lx version_dir (cat "$CUR_DIR/../version.info" | grep "version_dir" | cut -d"=" -f2)
    if test "-$version_dir" = "-"
        echo (realpath $CUR_DIR/../..)
    else
        echo (realpath $CUR_DIR/../../../latest)
    end
end
set -x INSTALL_DIR (get_install_dir)

set -x toolchain_path "$INSTALL_DIR/$PACKAGE"
if test -d {$toolchain_path}
    set -lx toolchain_home (echo {$TOOLCHAIN_HOME})
    set -lx num (echo ":$toolchain_home:" | grep ":$toolchain_path:" | wc -l)
    if test $num -eq 0
        if test "-$toolchain_home" = "-"
            set -gx TOOLCHAIN_HOME {$toolchain_path}
        else
            set -gx TOOLCHAIN_HOME {$toolchain_path}:{$toolchain_home}
        end
    end
end
set -e toolchain_path

set -x lib_path "$INSTALL_DIR/$PACKAGE/python/site-packages/"
if test -d $lib_path
    set -lx python_path (echo {$PYTHONPATH})
    set -lx num (echo ":$python_path:" | grep ":$lib_path:" | wc -l)
    if test $num -eq 0
        if test "-$python_path" = "-"
            set -gx PYTHONPATH {$lib_path}
        else
            set -gx PYTHONPATH {$lib_path}:{$python_path}
        end
    end
end
set -e lib_path

set -x op_tools_path "$INSTALL_DIR/$PACKAGE/python/site-packages/bin/"
if test -d $op_tools_path
    set -lx num (echo "$PATH" | grep "$op_tools_path" | wc -l)
    if test $num -eq 0
        if test "-$PATH" = "-"
            set -gx PATH {$op_tools_path}
        else
            set -gx PATH {$op_tools_path} {$PATH}
        end
    end
end
set -e op_tools_path

set -x msprof_path "$INSTALL_DIR/$PACKAGE/tools/profiler/bin/"
if test -d $msprof_path
    set -lx num (echo "$PATH" | grep "$msprof_path" | wc -l)
    if test $num -eq 0
        if test "-$PATH" = "-"
            set -gx PATH {$msprof_path}
        else
            set -gx PATH {$msprof_path} {$PATH}
        end
    end
end
set -e msprof_path

set -x ascend_ml_path "$INSTALL_DIR/$PACKAGE/tools/aml/lib64"
if test -d $ascend_ml_path
    set -lx num=`echo ":$LD_LIBRARY_PATH:" | grep ":${ascend_ml_path}:" | wc -l`
    if test $num -eq 0
        if test "-$LD_LIBRARY_PATH" = "-"
            set -gx LD_LIBRARY_PATH ${ascend_ml_path}
        else
            set -gx LD_LIBRARY_PATH ${ascend_ml_path} ${ascend_ml_path}/plugin ${LD_LIBRARY_PATH}
        fi
    fi
fi
set -e ascend_ml_path

set -x asys_path "$INSTALL_DIR/$PACKAGE/tools/ascend_system_advisor/asys"
if test -d asys_path
    set -lx num (echo "$PATH" | grep "asys_path" | wc -l)
    if test $num -eq 0
        if test "-$PATH" = "-"
            set -gx PATH {asys_path}
        else
            set -gx PATH {asys_path} {$PATH}
        end
    end
end
set -e asys_path

ccec_compiler_path="$INSTALL_DIR/$PACKAGE/tools/ccec_compiler/bin"
if test -d $ccec_compiler_path
    set -lx num (echo "$PATH" | grep "$ccec_compiler_path" | wc -l)
    if test $num -eq 0
        if test "-$PATH" = "-"
            set -gx PATH {$ccec_compiler_path}
        else
            set -gx PATH {$ccec_compiler_path} {$PATH}
        end
    end
end
set -e ccec_compiler_path

set -x msobjdump_path "$INSTALL_DIR/$PACKAGE/tools/msobjdump/"
if test -d $msobjdump_path
    set -lx num (echo "$PATH" | grep "$msobjdump_path" | wc -l)
    if test $num -eq 0
        if test "-$PATH" = "-"
            set -gx PATH {$msobjdump_path}
        else
            set -gx PATH {$msobjdump_path} {$PATH}
        end
    end
end
set -e msobjdump_path

set -x ascendump_path "$INSTALL_DIR/$PACKAGE/tools/show_kernel_debug_data/"
if test -d $ascendump_path
    set -lx num (echo "$PATH" | grep "$ascendump_path" | wc -l)
    if test $num -eq 0
        if test "-$PATH" = "-"
            set -gx PATH {$ascendump_path}
        else
            set -gx PATH {$ascendump_path} {$PATH}
        end
    end
end
set -e ascendump_path