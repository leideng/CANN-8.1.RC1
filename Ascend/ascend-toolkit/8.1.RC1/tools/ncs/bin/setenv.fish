#!/usr/bin/env fish
# Perform setenv for ncs package
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

set -x PACKAGE ncs
set -lx CUR_DIR (cd (dirname (status --current-filename)); pwd)
set -lx INSTALL_PATH (realpath {$CUR_DIR}/../..)

set -x ncs_path "$INSTALL_DIR/$PACKAGE/bin"
if test -d {$ncs_path}
    set -lx ncs_home (echo {$PATH})
    set -lx num (echo ":{$ncs_home}:" | grep ":{$ncs_path}:" | wc -l)
    if test $num -eq 0
        if test "-$ncs_home" = "-"
            set -gx PATH {$ncs_path}
        else
            set -gx PATH {$ncs_path}:{$ncs_home}
        end
    end
end
set -e ncs_path

set -x ld "$INSTALL_PATH/$PACKAGE/lib64"
if test -d {$ld}
    set -lx ld_library_path (echo {$LD_LIBRARY_PATH})
    set -lx num (echo ":{$ld_library_path}:" | grep ":{$ld}:" | wc -l)
    if test $num -eq 0
        if test "-$ld_library_path" = "-"
            set -gx LD_LIBRARY_PATH {$ld}
        else
            set -gx LD_LIBRARY_PATH {$ld}:{$ld_library_path}
        end
    end
end
set -e ld
