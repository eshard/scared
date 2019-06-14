#vim : set ft=sh et:

cd $SRC_DIR

$PYTHON setup.py sdist
$PYTHON setup.py bdist_wheel

$PYTHON setup.py install --single-version-externally-managed --record=record.txt

