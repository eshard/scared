#!/bin/bash
set -x
cd $SRC_DIR
$PYTHON -m pip install --no-deps --ignore-installed -v .
