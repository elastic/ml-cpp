#!/bin/bash
set -xe
export PATH=/opt/homebrew/bin:/usr/local/bin:$PATH

MACHINE_TYPE=$(uname -m)
if [ ${MACHINE_TYPE} != "arm64" ]; then
    echo "Unsupported architecture: ${MACHINE_TYPE}"
    echo "Expected arm64"
    exit 1
fi

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

# Build and install boost 1.86.0
curl -L https://boostorg.jfrog.io/artifactory/main/release/1.86.0/source/boost_1_86_0.tar.bz2 | tar xvjf - && \
cd boost_1_86_0 && \
./bootstrap.sh --with-toolset=clang --without-libraries=context --without-libraries=coroutine --without-libraries=graph_parallel --without-libraries=mpi --without-libraries=python --without-icu && \
sed -i -e 's|constexpr static std::size_t const sizes[] = {13ul, 29ul, 53ul, 97ul,|constexpr static std::size_t const sizes[] = {3ul, 13ul, 29ul, 53ul, 97ul,|' boost/unordered/detail/prime_fmod.hpp
sudo ./b2 -j8 install --layout=versioned --disable-icu cxxflags="-std=c++17 -stdlib=libc++" linkflags="-std=c++17 -stdlib=libc++ -Wl,-headerpad_max_install_names" optimization=speed inlining=full define=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS define=BOOST_LOG_WITHOUT_DEBUG_OUTPUT define=BOOST_LOG_WITHOUT_EVENT_LOG define=BOOST_LOG_WITHOUT_SYSLOG define=BOOST_LOG_WITHOUT_IPC && \
cd .. && \
sudo rm -rf boost_1_86_0

# Install python modules required by PyTorch
sudo pip3 install numpy ninja pyyaml setuptools cffi typing_extensions future six requests dataclasses

# Build and install PyTorch
git clone --depth=1 --branch=v2.3.1 https://github.com/pytorch/pytorch.git && \
cd pytorch && \
git submodule sync && \
git submodule update --init --recursive && \
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
export PYTORCH_BUILD_VERSION=2.3.1
export PYTORCH_BUILD_NUMBER=1
python3 setup.py install

sudo mkdir -p /usr/local/lib && \
sudo mkdir -p /usr/local/include/pytorch && \
sudo cp -r torch/include/* /usr/local/include/pytorch/ && \
sudo cp torch/lib/libtorch_cpu.dylib /usr/local/lib/ && \
sudo cp torch/lib/libc10.dylib /usr/local/lib/ && \
cd .. && \
rm -rf pytorch

# Make sure all changes are written to disk
sync
