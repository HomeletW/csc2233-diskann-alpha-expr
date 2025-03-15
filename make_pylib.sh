#!/usr/bin/zsh

BUILD_DIR="$(pwd)/build"

mkdir -p $BUILD_DIR
pushd $BUILD_DIR
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
popd

pip install build
python -m build
pip install --force dist/*.whl
pip install -r alpha_expr/requirements.txt
