#!/bin/bash
set -xe
export PATH=/opt/homebrew/bin:/usr/local/bin:$PATH

MACHINE_TYPE=$(uname -m)
if [ ${MACHINE_TYPE} != "arm64" ]; then
    echo "Unsupported architecture: ${MACHINE_TYPE}"
    echo "Expected arm64"
    exit 1
fi

SYSTEM_PYTHON=/opt/homebrew/opt/python@3.12/bin/python3.12
SYSTEM_PIP=/opt/homebrew/opt/python@3.12/bin/pip3.12

# Verify system Python exists before proceeding
if [ ! -f "$SYSTEM_PYTHON" ]; then
    echo "Homebrew Python not found at $SYSTEM_PYTHON. Check install.sh."
    exit 1
fi

# Create and activate virtual environment
echo "Creating Python virtual environment..."
$SYSTEM_PYTHON -m venv /tmp/ml-cpp-build-venv
source /tmp/ml-cpp-build-venv/bin/activate

# Update paths to use venv
PYTHON_EXE=/tmp/ml-cpp-build-venv/bin/python
PIP_EXE=/tmp/ml-cpp-build-venv/bin/pip

export CPP='clang -E'
export CC=clang
export CFLAGS="-O3"
export CXX='clang++ -std=c++17 -stdlib=libc++'
export CXXFLAGS="-O3"
export CXXCPP='clang++ -std=c++17 -E'
export LDFLAGS=-Wl,-headerpad_max_install_names
unset CPATH
unset C_INCLUDE_PATH
unset CPLUS_INCLUDE_PATH
unset LIBRARY_PATH

# 1. Download to a physical file first
echo "Downloading Boost 1.86.0..."
curl -L -f https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.bz2 -o boost_1_86_0.tar.bz2

# 2. Check if the file is valid (not empty/missing)
if [ ! -s boost_1_86_0.tar.bz2 ]; then
    echo "ERROR: Boost download failed. Check network or URL."
    exit 1
fi

# 3. Extract the physical file
echo "Extracting Boost..."
tar xjf boost_1_86_0.tar.bz2
cd boost_1_86_0

# 4. Run the bootstrap
./bootstrap.sh --with-toolset=clang --without-libraries=python --without-icu

# 5. Install WITHOUT the sed patch
echo "Starting Boost build (this may take a few minutes)..."
sudo ./b2 -j4 install \
  --layout=versioned \
  --disable-icu \
  architecture=arm \
  address-model=64 \
  threading=multi \
  link=static,shared \
  variant=release \
  cxxflags="-std=c++17 -stdlib=libc++" \
  linkflags="-stdlib=libc++" \
  define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS \
  --without-python

# 6. Cleanup
cd ..
sudo rm -rf boost_1_86_0 boost_1_86_0.tar.bz2

# Install python modules in virtual environment
echo "Installing Python packages in virtual environment..."
$PIP_EXE install numpy ninja pyyaml setuptools cffi typing_extensions future six requests dataclasses


# Build and install PyTorch
git clone --depth=1 --branch=v2.7.1 https://github.com/pytorch/pytorch.git && \
cd pytorch && \
git submodule sync && \
git submodule update --init --recursive
sed -i -e 's/system(/strlen(/' torch/csrc/jit/codegen/fuser/cpu/fused_kernel.cpp && \
sed -i -e '/CUDNN_ROOT/a\
           "DNNL_TARGET_ARCH",
' tools/setup_helpers/cmake.py && \
sed -i -e 's/CMAKE_LINK_WHAT_YOU_USE TRUE/CMAKE_LINK_WHAT_YOU_USE FALSE/' CMakeLists.txt && \
sed -i -e '189i\
inline void to_bytes(bytestring& bytes, const unsigned long arg) {\
  auto as_cstring = reinterpret_cast<const char*>(&arg);\
  bytes.append(as_cstring, sizeof(unsigned long));\
}\
' third_party/ideep/include/ideep/utils.hpp

export BLAS=vecLib
export BUILD_TEST=OFF
export BUILD_CAFFE2=OFF
export USE_NUMPY=OFF
export USE_DISTRIBUTED=OFF
export DNNL_TARGET_ARCH=AARCH64
export USE_MKLDNN=ON
export USE_QNNPACK=OFF
export USE_PYTORCH_QNNPACK=OFF
export PYTORCH_BUILD_VERSION=2.7.1
export PYTORCH_BUILD_NUMBER=1
echo "Installing PyTorch in virtual environment..."
$PYTHON_EXE setup.py install

sudo mkdir -p /usr/local/lib && \
sudo mkdir -p /usr/local/include/pytorch && \
sudo cp -r torch/include/* /usr/local/include/pytorch/ && \
sudo cp torch/lib/libtorch_cpu.dylib /usr/local/lib/ && \
sudo cp torch/lib/libc10.dylib /usr/local/lib/ && \
cd .. && \
rm -rf pytorch

# Make sure all changes are written to disk
sync
