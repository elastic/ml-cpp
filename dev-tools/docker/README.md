# Summary of Docker images in the ml-dev namespace

## Repository: ml-linux-dependency-build

### Latest version: pytorch_viable_strict

### Comments
A Docker image that can be used to compile the machine learning
C++ code for Linux with the latest build of the PyTorch release branch.

This image is intended to be built regularly in order to flush out any
issues we may have with the latest PyTorch release as early as possible.
To avoid swamping the Docker registry with numerous images we re-use the
same tag - pytorch_viable_strict - for the image.

### Depends on: ml-linux-dependency-builder:3

### Build script: dev-tools/docker/build_pytorch_linux_build_image.sh


## Repository: ml-linux-dependency-builder

### Latest version: 4

### Comments
A Docker image that can be used to compile the dependencies of
the machine learning C++ code for Linux, e.g. PyTorch or Boost, in order
to test the latest versions of them.

This image is not intended to be built regularly. It is based on the latest
`ml-linux-build` image, but with extra dependencies added in order to build
PyTorch.  Therefore, when changing any of the tools or 3rd party components
required to build the machine learning C++ code dependencies:

1. Increment the version
2. Change the Dockerfile and build a new image to be used for subsequent builds on this branch.
3. Update the version to be used for builds in docker files that refer to it.

### Depends on: ml-linux-build:33

### Build script: dev-tools/docker/build_linux_dependency_builder_image.sh



## Repository: ml-linux-build

### Latest version: 33

### Comments
A Docker image that can be used to compile the machine learning
C++ code for Linux x86_64. It is intended to be used for PR builds etc.

This image is not intended to be built regularly.  When changing the tools
or 3rd party components required to build the machine learning C++ code:

 1. increment the version
 2.  change the Dockerfile and build a new image to be
used for subsequent builds on this branch.
 3.  Update the version to be sed for builds in *dev-tools/docker/linux_builder/Dockerfile.*

### Build script: dev-tools/docker/build_linux_build_image.sh


## Repository: ml-linux-aarch64-cross-build

### Latest version: 16

### Comments
A Docker image that can be used to compile the machine learning
C++ code for Linux aarch64 (cross compiled).

This image is not intended to be built regularly.  When changing the tools
or 3rd party components required to build the machine learning C++ code:

 1. Increment the version 
 2. Change the Dockerfile and build a new image to be
used for subsequent builds on this branch.  
 3. Update the version to be used for builds in *dev-tool/docker/linux_aarch64_cross_builder/Dockerfile.*

### Build script: dev-tools/docker/build_linux_aarch64_cross_build_image.sh


## Repository: ml-linux-aarch64-native-build

### Latest version: 16

### Comments
A Docker image that can be used to compile the machine learning
C++ code for Linux aarch64 (native).

This image is not intended to be built regularly.  When changing the tools
or 3rd party components required to build the machine learning C++ code:

 1. Increment the version
 2. Change the Dockerfile and build a new image to be used for subsequent builds on this branch
 3. Update the version to be used for builds in *dev-tool/docker/linux_aarch64_native_builder/Dockerfile.*

### Build script: dev-tools/docker/build_linux_aarch64_native_build_image.sh


## Repository: ml_cpp_linux_x86_64_jdk17

### Latest version: 3

### Comments
A Docker image that can be used to run a subset of the ES
integration tests.

This image is not intended to be built regularly. If changing the *Dockerfile* in any way:

 1. Increment the version
 2. Build a new image

### Build script: dev-tools/docker/build_linux_jdk.sh


## Repository: ml-check-style

### Latest version: 2

### Comments
A Docker image that is used to run `clang-format` over the `ml-cpp` code.
This image is not intended to be built regularly.  When changing the
`clang-format` version:

 1. Increment the image version
 2. Build a new image

 
### Build script: dev-tools/docker/build_check_style_image.sh

