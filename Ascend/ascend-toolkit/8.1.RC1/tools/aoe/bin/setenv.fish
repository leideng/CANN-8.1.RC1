#!/usr/bin/env fish

set -g curpath (realpath (dirname (status --current-filename)))
set -g curfile (realpath (status --current-filename))
set -g param_mult_ver $argv[1]

function get_install_dir
    if test "$param_mult_ver" = "multi_version"
        echo (realpath $curpath/../../../../latest)/tools/aoe
    else
        echo (realpath $curpath/..)
    end
end

set -l INSTALL_DIR (get_install_dir)
set -x BIN $INSTALL_DIR/bin
# traverse PATH list
for bin in $PATH
    if test $bin != $BIN
        set -x BIN $BIN {$bin}
    end
end

# set PATH
set -gx PATH $BIN
set -e BIN

# set LD_LIBRARY_PATH
# get lib64 realpath save local variable
set -lx REAL $INSTALL_DIR/lib64
set -x LD $REAL

# split by '\n'
set -lx WORDS (echo $LD_LIBRARY_PATH | sed 's/:/\n/g')

# traverse LD_LIBRARY_PATH
for bin in $WORDS
    if test $bin != $REAL
        set -x LD $LD":"{$bin}
    end
end

# set LD_LIBRARY_PATH
set -gx LD_LIBRARY_PATH $LD
set -e LD
