#!/bin/bash

cur_path=$(dirname "$0")
new_version="$1"

init_py="$cur_path/../xalgorithm/__init__.py"

if [ ! -f "$init_py" ]; then
    echo "File $init_py does not exist."
    exit 1
fi

grep -q "^__version__" "$init_py" # check if there's a line starting with __version__

if [ $? -eq 0 ]; then # Line starting with __version__ exists, so overwrite it
    sed -i "s/^__version__.*/__version__ = \"$new_version\"/" "$init_py"
else  # Line starting with __version__ does not exist, so add it at the end
    echo -e "\n\n__version__ = \"$new_version\"" >> "$init_py"
fi