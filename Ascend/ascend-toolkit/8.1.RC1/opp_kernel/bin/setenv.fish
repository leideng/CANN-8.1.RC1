#!/usr/bin/fish
set param_mult_ver $argv[1]
set REAL_SHELL_PATH (realpath (command -v $argv[0]))
set CANN_PATH (cd (dirname $REAL_SHELL_PATH)/../../ && pwd)
if test -d "$CANN_PATH/opp" -a test -d "$CANN_PATH/../latest"
    set INSATLL_PATH (cd (dirname $REAL_SHELL_PATH)/../../../ && pwd)
    if test -L "$INSATLL_PATH/latest/opp"
        set _ASCEND_OPP_PATH (cd $CANN_PATH/opp && pwd)
        if test "$param_mult_ver" = "multi_version"
            set _ASCEND_OPP_PATH (cd $INSATLL_PATH/latest/opp && pwd)
        end
    elseif not test -L "$INSATLL_PATH/latest/opp" -a test -L "$INSATLL_PATH/latest/opp_kernel"
        set _ASCEND_OPP_PATH (cd $CANN_PATH/opp && pwd)
    end
elseif test -d "$CANN_PATH/opp"
    set _ASCEND_OPP_PATH (cd $CANN_PATH/opp && pwd)
end

set -x ASCEND_OPP_PATH $_ASCEND_OPP_PATH
set -x PYTHONPATH $PYTHONPATH:$ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/
cd $CURR_DIR
set -l custom_path_file "$_ASCEND_OPP_PATH/../conf/path.cfg"
set -l common_interface "$_ASCEND_OPP_PATH/script/common_interface.fish"
set -l owner (stat -c \%U "$curfile")
set -l dst_dir (grep -w "data" "$custom_path_file" | cut --only-delimited -d"=" -f2-)
set -l dst_dir (eval echo "$dst_dir")
set -l vendors_dir "$_ASCEND_OPP_PATH/vendors"
set -l vendor_dir (ls "$vendors_dir" 2> /dev/null)

if test (id -u) -ne 0 -a (id -un) != "$owner" -a -f "$custom_path_file" -a -f "$common_interface"
    . "$common_interface"
    mk_custom_path "$custom_path_file"
    for dir_name in "data"
        if test -d "$_ASCEND_OPP_PATH/built-in/$dir_name" -a -d "$dst_dir"
            chmod -R u+w $dst_dir > /dev/null 2>&1
            cp -rfL $_ASCEND_OPP_PATH/built-in/$dir_name/* "$dst_dir"
        end
        if test -d "$vendors_dir/$vendor_dir/$dir_name" -a -d "$dst_dir"
            chmod -R u+w $dst_dir > /dev/null 2>&1
            cp -rfL $vendors_dir/$vendor_dir/$dir_name/* "$dst_dir"
        else if test ! -d "$vendors_dir/$vendor_dir/$dir_name" -a -d "$dst_dir" -a -d "$_ASCEND_OPP_PATH/$dir_name/custom"
            chmod -R u+w $dst_dir > /dev/null 2>&1
            cp -rfL $_ASCEND_OPP_PATH/$dir_name/* "$dst_dir"
        end
    end
end
if test (id -u) -ne 0 -a (id -un) != "$owner"
    set -l opp_custom_list op_impl op_proto fusion_pass fusion_rules framework
    for i in $opp_custom_list
        set -l dst_file $dst_dir$i/custom
        mkdir -p "$dst_file"
        chmod -R u+w $dst_file > /dev/null 2>&1
        set -l custom_file (find $_ASCEND_OPP_PATH/ -name "custom" |grep $i)
        if test "$custom_file" != "" -a ! -d "$vendors_dir/$vendor_dir/$i"
            set -l opp_custom_file (ls "$custom_file" 2> /dev/null)
        else if test -d "$vendors_dir/$vendor_dir/$i"
            set -l opp_custom_file (ls "$vendors_dir/$vendor_dir/$i" 2> /dev/null)
        end
        if test "$opp_custom_file" != ""
            cp -rfL $custom_file/*  $dst_file
        else
            echo "[INFO]: $custom_file/ is empty"
        end
    end
end

