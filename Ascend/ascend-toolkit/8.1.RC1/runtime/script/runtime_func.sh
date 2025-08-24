#!/bin/sh
# Perform common functions for runtime package
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

create_acllib_softlink() {
    local install_path="$1"
    local pkgname="acllib"
    local acllib_installed=""

    does_pkg_installed "${install_path}" "${pkgname}" "acllib_installed"

    if [ "${acllib_installed}" = "true" ]; then
        run_uninstall_script "${install_path}" "${pkgname}"
    fi

    remove_dir "${install_path}/${pkgname}"
    create_softlink "${install_path}" "${install_path}" "runtime" "${pkgname}"
}

remove_acllib_softlink() {
    local install_path="$1"
    remove_softlink "${install_path}/acllib"
}

create_modeldeployer_softlink() {
    local install_path="$1"
    local pkgname="modeldeployer"

    if [ -d "${install_path}/hsheadfwk" ] || [ -d "${install_path}/hsdevre" ]; then
        remove_dir "${install_path}/${pkgname}"
        create_softlink "${install_path}" "${install_path}" "runtime" "${pkgname}"
    fi
}

remove_modeldeployer_softlink() {
    local install_path="$1"
    remove_softlink "${install_path}/modeldeployer"
}

get_pkg_arch_name() {
    local scene_info="$curpath/../scene.info"
    if [ ! -f "$scene_info" ]; then
        local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
        echo "[Runtime] [$cur_date] [ERROR]: $scene_info file cannot be found!"
        exit 1
    fi
    local arch="$(grep -iw arch "$scene_info" | cut -d"=" -f2- | awk '{print tolower($0)}')"
    if [ -z "$arch" ]; then
        local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
        echo "[Runtime] [$cur_date] [ERROR]: var arch cannot be found in file $scene_info!"
        exit 1
    fi
    echo $arch
}

get_installed_pkg_arch_name() {
    if [ "$pkg_is_multi_version" = "true" ]; then
        local scene_info="$docker_root$input_install_path/$pkg_version_dir/runtime/scene.info"
    else
        local scene_info="$docker_root$input_install_path/runtime/scene.info"
    fi
    if [ ! -f "$scene_info" ]; then
        return
    fi
    local arch="$(grep -iw arch "$scene_info" | cut -d"=" -f2- | awk '{print tolower($0)}')"
    echo $arch
}

get_os_arch_name() {
    uname -m 2>&1 | awk '{print tolower($0)}'
}

get_hetero_arch_name() {
    if [ "x$(get_pkg_arch_name)" = "xx86_64" ]; then
        echo aarch64
    else
        echo x86_64
    fi
}

if [ "$(get_pkg_arch_name)" = "$(get_os_arch_name)" ]; then
    is_pkg_hetero_arch="n"
else
    is_pkg_hetero_arch="y"
fi

arch_linux_path="$(get_pkg_arch_name)-linux"
arch_scripts_path="$arch_linux_path/hetero-arch-scripts"
arch_linux_path_os="$(get_os_arch_name)-linux"
arch_linux_path_hetero="$(get_hetero_arch_name)-linux"
arch_scripts_path_hetero="$arch_linux_path_hetero/hetero-arch-scripts"

is_same_arch_pkg_installed() {
    if [ "$pkg_is_multi_version" = "true" ]; then
        local linux_path="$docker_root$input_install_path/$pkg_version_dir/$arch_linux_path_os"
    else
        local linux_path="$docker_root$input_install_path/$arch_linux_path_os"
    fi
    if [ "$is_pkg_hetero_arch" = "y" ] && [ -d "$linux_path" ]; then
        echo "y"
    else
        echo "n"
    fi
}

get_package_upgrade_install_info_hetero() {
    local _outvar="$1"
    local path_latest="$docker_root$input_install_path/latest"
    local path_hetero="$docker_root$input_install_path/latest/$arch_scripts_path"
    if [ -d "$path_latest" ] && [ -d "$path_hetero" ]; then
        local install_info=$(find "$path_hetero" -type f -name 'ascend_install.info' | grep 'runtime/ascend_install.info')
        if [ -n "$install_info" ]; then
            install_info="$(realpath $install_info)"
            eval "${_outvar}=\"$install_info\""
            return
        fi
    fi
    eval "${_outvar}=\"\""
}

is_hetero_arch_pkg_installed() {
    if [ "$is_pkg_hetero_arch" = "y" ]; then
        if [ "$pkg_is_multi_version" = "true" ]; then
            local path_version="$docker_root$input_install_path/$pkg_version_dir/$arch_scripts_path/runtime/version.info"
            local path_install="$docker_root$input_install_path/$pkg_version_dir/$arch_scripts_path/runtime/ascend_install.info"
        else
            local path_version="$docker_root$input_install_path/$arch_scripts_path/runtime/version.info"
            local path_install="$docker_root$input_install_path/$arch_scripts_path/runtime/ascend_install.info"
        fi
        if [ -f "$path_version" ] && [ -f "$path_install" ]; then
            echo "installed-hetero"
            return
        fi

        if [ "$pkg_is_multi_version" = "true" ]; then
            local path_version="$docker_root$input_install_path/$pkg_version_dir/runtime/version.info"
            local path_install="$docker_root$input_install_path/$pkg_version_dir/runtime/ascend_install.info"
        else
            local path_version="$docker_root$input_install_path/runtime/version.info"
            local path_install="$docker_root$input_install_path/runtime/ascend_install.info"
        fi
        if [ -f "$path_version" ] && [ -f "$path_install" ] && [ "$(get_installed_pkg_arch_name)" = "$(get_pkg_arch_name)" ]; then
            echo "installed-normal"
            return
        fi

        if [ "$pkg_is_multi_version" = "true" ]; then
            get_package_upgrade_install_info_hetero "ascend_install_info"
            if [ -f "${ascend_install_info}" ]; then
                echo "installed-hetero-to-be-upgraded"
                return
            else
                get_package_upgrade_install_info "ascend_install_info" "$docker_root$input_install_path" "runtime"
                if [ -f "${ascend_install_info}" ]; then
                    echo "installed-normal-to-be-upgraded"
                    return
                fi
            fi
        fi

        echo "no"
    else
        echo "NA"
    fi
}

update_install_info_hetero() {
    local install_info=$1
    local pkg_version_dir=$2
    if [ -f "$install_info" ]; then
        chmod u+w "$(dirname $install_info)" "$install_info"
        local new_install_path="$(grep -iw Runtime_Install_Path_Param $install_info)"
        local new_install_path="$(echo $new_install_path | awk -v version=$pkg_version_dir 'BEGIN{FS=OFS="/"} {$(NF-2)=version; print}')"
        sed -i -e "s:Runtime_Install_Path_Param=.*:$new_install_path:" "$install_info"
    fi
}

replace_filelist() {
    local curpath=$(dirname $(readlink -f "$0"))
    cp -p $curpath/filelist.csv $curpath/filelist.csv.bak && awk -v arch_prefix="$arch_linux_path" 'BEGIN {
        FS=OFS=","
        pat="^"arch_prefix
    }

    function in_pkg_black_list() {
        relative_install_path = $4
        if (relative_install_path ~ /^runtime\/python/) { return 1 }
        return 0
    }

    function format_softlink(links) {
        split(links, arr, ";")
        ret = ""
        for (i in arr) {
            if (arr[i] ~ pat) { arr[i] = "../../" arr[i] }
            ret = ret ";" arr[i]
        }
        sub("^;", "", ret)
        return ret
    }

    {
        if (in_pkg_black_list()) { next }
        if ($4 ~ pat) {
            if ($4 ~ pat"/python") { $15 = "NA" }
            $4 = "../../" $4;
        }
        $9 = format_softlink($9)
        print
    }' $curpath/filelist.csv.bak > $curpath/filelist.csv
}

get_dir_mod() {
    local path="$1"
    stat -c %a "$path"
}

remove_dir_recursive() {
    local dir_start="$1"
    local dir_end="$2"
    if [ "$dir_end" = "$dir_start" ]; then
        return 0
    fi
    if [ ! -e "$dir_end" ]; then
        return 0
    fi
    if [ "x$(ls -A $dir_end 2>&1)" != "x" ]; then
        return 0
    fi
    local up_dir="$(dirname $dir_end)"
    local oldmod="$(get_dir_mod $up_dir)"
    chmod u+w "$up_dir"
    [ -n "$dir_end" ] && rm -rf "$dir_end"
    if [ $? -ne 0 ]; then
        chmod "$oldmod" "$up_dir"
        return 1
    fi
    chmod "$oldmod" "$up_dir"
    remove_dir_recursive "$dir_start" "$up_dir"
}

get_pkg_toolchain() {
    local scene_info="$curpath/../scene.info"
    grep -iw ^toolchain "$scene_info" 2> /dev/null | cut -d"=" -f2-
}
