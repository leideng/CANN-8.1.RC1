#!/usr/bin/fish

set ILE_NOT_EXIST "0x0080"

set cur_date (date +"%Y-%m-%d %H:%M:%S")
echo "[Opp][$cur_date][INFO]: Start pre installation check of opp module."
which python3 >/dev/null
if test ! $status -eq 0
    set cur_date (date +"%Y-%m-%d %H:%M:%S")
    exit 0
end
set python_version (python3 --version 2>/dev/null)
set val (echo $python_version|grep -i '3.7.5')
if test ! $status -eq 0
    set cur_date (date +"%Y-%m-%d %H:%M:%S")
    exit 0
else
    set cur_date (date +"%Y-%m-%d %H:%M:%S")
    exit 0
end
