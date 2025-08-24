#!/usr/bin/env bash

curpath=`readlink -f  $(dirname $BASH_SOURCE[0])`
param_mult_ver=$1

get_install_dir() {
    if [ "$param_mult_ver" = "multi_version" ]; then
        echo "$(realpath $curpath/../../../../latest)/tools/aoe"
    else
        echo "$(realpath $curpath/..)"
    fi
}

set_path_env() {
BIN=$1
WORDS=`echo $2 | sed 's/:/ /g'`
for bin in $WORDS
do
    if [ "$bin" != "$1" ]
    then
        BIN=$BIN:$bin
    fi
done
export PATH="$BIN"
}

set_ld_env() {
LD=$1:$2
LDWORDS=`echo $3 | sed 's/:/ /g'`
for ld in $LDWORDS
do
    if [ "$ld" != "$1" ] && [ "$ld" != "$2" ]
    then
        LD=$LD:$ld
    fi
done
export LD_LIBRARY_PATH="$LD"
}

INSTALL_DIR="$(get_install_dir)"
main() {
    BIN_PATH=$INSTALL_DIR/bin
    LD_PATH=$INSTALL_DIR/lib64
    set_path_env $BIN_PATH $PATH
    set_ld_env $LD_PATH $LD_LIBRARY_PATH
}

main
