#!/usr/bin/env fish

set PYTHON_VERSION 3.7.5
set -lx SHELL_DIR (cd (dirname (status --current-filename)); pwd)

set -l python_version (python3 --version | cut -d" " -f2)
for num in (seq 1 3)
    set -l version_num (echo {$python_version} | cut -d"." -f{$num})
    set -l min_version_num (echo {$PYTHON_VERSION} | cut -d"." -f{$num})
    if test {$version_num} -lt {$min_version_num}
        echo "python version $python_version low than $PYTHON_VERSION."
        break
    end
    if test {$version_num} -gt {$min_version_num}
        break
    end
end

set python_module_infos sqlite3:
set install_info "$SHELL_DIR/../ascend_install.info"
if test -f "$install_info"
    set -l feature_type (cat $install_info | grep "Feature_Type")
    set -l feature_type_matched (echo $feature_type | grep -Eo "op|all|model")
    if test "-$feature_type_matched" != "-"
        set python_module_infos $python_module_infos numpy:1.13.3
    end
    set feature_type_matched (echo $feature_type | grep -Eo "op|all")
    if test "-$feature_type_matched" != "-"
        set python_module_infos $python_module_infos scipy:1.4.1 psutil:5.7.0 protobuf:3.13.0
    end
end

for module_info in $python_module_infos
    set -l module (echo {$module_info} | cut -d":" -f1)
    if test $module = "protobuf"
        python3 -c "import google.protobuf" >/dev/null 2>&1
    else
        python3 -c "import $module" >/dev/null 2>&1
    end
    if test $status -gt 0
        echo "python module $module doesn't exist."
        continue
    end

    set -l min_version (echo {$module_info} | cut -d":" -f2)
    if test "-$min_version" = "-"
        continue
    end

    set -l installed_version (pip3 list 2>&1 | grep "^$module " | cut -d" " -f2- | grep -Eo "[0-9.]+")
    if test "-$installed_version" = "-"
        echo "search $module version failed."
        continue
    end
    set temp_installed_version {$installed_version}
    set -l min_version_arr (echo $min_version | string split .)
    for min_version_num in $min_version_arr
        set -l installed_version_num (echo {$temp_installed_version} | cut -d"." -f1)
        set -l temp_installed_version (echo {$temp_installed_version} | cut -d"." -f2-)
        if test "-$installed_version_num" = "-"
            break
        end
        if test {$installed_version_num} -lt {$min_version_num}
            echo "python module $module version $installed_version low than $min_version."
            break
        end
        if test {$installed_version_num} -gt {$min_version_num}
            break
        end
    end
end
