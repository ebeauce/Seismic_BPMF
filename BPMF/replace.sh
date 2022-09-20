#!/bin/sh
# replace $1 with $2 in $*
# usage: replace "old-pattern" "new-pattern" file [file...]

OLD=$1          # first parameter of the script
NEW=$2          # second parameter
shift ; shift   # discard the first 2 parameters: the next are the file names
for file in $*  # for all files given as parameters
do
    # replace every occurrence of OLD with NEW, save on a temporary file
      sed "s/$OLD/$NEW/g" ${file} > ${file}.new
      # rename the temporary file as the original file
        /bin/mv ${file}.new ${file}
    done
