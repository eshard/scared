#!/bin/bash
set -e -x

PLAT=manylinux1_x86_64
PYBINS=(
    /opt/python/cp36-cp36m/bin
    /opt/python/cp37-cp37m/bin
    /opt/python/cp38-cp38/bin
)
# Compile wheels
for PYBIN in "${PYBINS[@]}"; do
    "${PYBIN}/pip" install pip setuptools wheel numpy
    "${PYBIN}/python" setup.py bdist_wheel 
done

# Bundle external shared libraries into the wheels
for whl in dist/*.whl; do
    auditwheel repair "$whl" --plat $PLAT
done
