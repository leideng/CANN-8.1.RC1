#!/bin/sh
# Perform uninstall for runtime package
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

username="$(id -un)"
usergroup="$(id -gn)"
is_quiet=n
curpath=$(dirname $(readlink -f "$0"))
common_func_path="${curpath}/common_func.inc"
pkg_version_path="${curpath}/../version.info"

. "${common_func_path}"

if [ "$1" ]; then
    input_install_dir="${2}"
    common_parse_type="${3}"
    is_quiet="${4}"
    is_docker_install="${5}"  # 兼容跨版本调用保留参数
    docker_root="${6}"
    is_recreate_softlink="${7}"
fi

if [ "${is_recreate_softlink}" = "y" ]; then
    recreate_softlink_option="--recreate-softlink"
else
    recreate_softlink_option=""
fi

if [ "x${docker_root}" != "x" ]; then
    common_parse_dir="${docker_root}${input_install_dir}"
else
    common_parse_dir="${input_install_dir}"
fi

get_version "pkg_version" "$pkg_version_path"
is_multi_version_pkg "pkg_is_multi_version" "$pkg_version_path"
if [ "$pkg_is_multi_version" = "true" ] && [ "$hetero_arch" != "y" ]; then
    get_version_dir "pkg_version_dir" "$pkg_version_path"
    common_parse_dir="$common_parse_dir/$pkg_version_dir"
fi

if [ $(id -u) -ne 0 ]; then
    log_dir="${HOME}/var/log/ascend_seclog"
else
    log_dir="/var/log/ascend_seclog"
fi
logfile="${log_dir}/ascend_install.log"

SOURCE_INSTALL_COMMON_PARSER_FILE="${common_parse_dir}/runtime/script/install_common_parser.sh"
SOURCE_FILELIST_FILE="${common_parse_dir}/runtime/script/filelist.csv"

# 写日志
log() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    local log_type="$1"
    local log_msg="$2"
    local log_format="[Runtime] [$cur_date] [$log_type]: $log_msg"
    if [ "$log_type" = "INFO" ]; then
        echo "$log_format"
    elif [ "$log_type" = "WARNING" ]; then
        echo "$log_format"
    elif [ "$log_type" = "ERROR" ]; then
        echo "$log_format"
    elif [ "$log_type" = "DEBUG" ]; then
        echo "$log_format" 1> /dev/null
    fi
    echo "$log_format" >> "$logfile"
}

##########################################################################
log "INFO" "step into run_runtime_uninstall.sh ......"
log "INFO" "uninstall target dir $common_parse_dir, type $common_parse_type."

if [ ! -d "$common_parse_dir/runtime" ]; then
    log "ERROR" "ERR_NO:0x0001;ERR_DES:path $common_parse_dir/runtime is not exist."
    exit 1
fi

new_uninstall() {
    if [ ! -d "${common_parse_dir}/runtime" ]; then
        log "INFO" "no need to uninstall runtime files."
        return 0
    fi

    # 赋可写权限
    chmod +w -R "${SOURCE_INSTALL_COMMON_PARSER_FILE}"

    if [ "$pkg_is_multi_version" = "true" ] && [ "$hetero_arch" = "y" ]; then
        local package_db_info="$common_parse_dir/var/ascend_package_db.info"
        if [ -e "$package_db_info" ]; then
            local linux_path="$(realpath $common_parse_dir/..)"
            local arch_path="$(basename $linux_path)"
            local latest_path="$(realpath $linux_path/../..)/latest"
            local pkgs="$(cut -d'|' -f2 $package_db_info | sort -u)"
            if [ "$pkgs" = "runtime" ]; then
                if [ -L "$latest_path/$arch_path" ] && [ "$(realpath $linux_path)" = "$(realpath $latest_path/$arch_path)" ]; then
                    rm -f "$latest_path/$arch_path"
                fi
            elif [ -n "$pkgs" ] && [ -d "$latest_path" ]; then
                if [ ! -e "$latest_path/$arch_path" ] || [ -L "$latest_path/$arch_path" ]; then
                    ln -srfn "$linux_path" "$latest_path"
                fi
            fi
        fi
    fi

    # 执行卸载
    custom_options="--custom-options=--common-parse-dir=$common_parse_dir,--logfile=$logfile,--stage=uninstall,--quiet=$is_quiet,--hetero-arch=$hetero_arch"
    sh "$SOURCE_INSTALL_COMMON_PARSER_FILE" --package="runtime" --uninstall --username="$username" --usergroup="$usergroup" ${recreate_softlink_option} \
        --version=$pkg_version --version-dir=$pkg_version_dir \
        --docker-root="$docker_root" $custom_options "$common_parse_type" "$input_install_dir" "$SOURCE_FILELIST_FILE"
    if [ $? -ne 0 ]; then
        log "ERROR" "ERR_NO:0x0090;ERR_DES:failed to uninstall package."
        return 1
    fi

    if [ -n "$latest_path" ] && [ -d "$latest_path" ] && [ "x$(ls -A $latest_path 2>&1)" = "x" ]; then
        rm -rf "$latest_path"
    fi

    return 0
}

new_uninstall
if [ $? -ne 0 ]; then
    exit 1
fi

exit 0
