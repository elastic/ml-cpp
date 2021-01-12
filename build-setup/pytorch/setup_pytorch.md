# Setup for Pytorch

If you wish to build and run the PyTorch inference applications you will need to install Pytorch and build the torch libraries. 

The PyTorch libaries are built with [Anaconda](https://www.anaconda.com). 
Anaconda must be installed first and can be downloaded [here](https://www.anaconda.com/products/individual#download-section). 


## Install the Python Dependencies 
```
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
```

## Clone the v1.7.1 release of the Pytorch repository
```
git clone --depth=1 --branch=v1.7.1 git@github.com:pytorch/pytorch.git
cd pytorch
# update submodules used by PyTorch
git submodule sync
git submodule update --init --recursive
``` 


## Build macOS
```
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
CC=clang CXX=clang++ python setup.py install
```

Once built copy headers and libraries to system directories
```
sudo mkdir /usr/local/include/pytorch
sudo cp -r torch/include/* /usr/local/include/pytorch/

sudo cp torch/lib/libtorch_cpu.dylib /usr/local/lib/
sudo cp torch/lib/libc10.dylib /usr/local/lib/
```

## Linux
TODO

## Windows
TODO