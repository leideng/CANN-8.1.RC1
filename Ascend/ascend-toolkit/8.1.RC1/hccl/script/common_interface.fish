#!/usr/bin/env fish

function mk_custom_path
    set -l custom_file_path $argv[1]
    if test (id -u) -eq 0
        return 0
    end
    while read line
        set -l _custom_path (echo "$line" | cut --only-delimited -d= -f2)
        if test -z $_custom_path
            continue
        end
        set -l _custom_path (eval echo "$_custom_path")
        if not test -d $_custom_path
            mkdir -p "$_custom_path"
            if not test $status -eq 0
                set -l cur_date (date +"%Y-%m-%d %H:%M:%S")
                echo "[Common] [$cur_date] [ERROR]: create $_custom_path failed."
                return 1
            end
        end
    end < $custom_file_path
    return 0
end
