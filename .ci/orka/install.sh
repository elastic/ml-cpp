#!/bin/bash
set -xe
export PATH=/opt/homebrew/bin:/usr/local/bin:$PATH

MACHINE_TYPE=$(uname -m)

if [ ${MACHINE_TYPE} == "x86_64" ]; then
    echo "Running on x86_64"

elif [ ${MACHINE_TYPE} == "arm64" ]; then
    echo "Running on arm64"

else
    echo "Unknown architecture: ${MACHINE_TYPE}"
    exit 1
fi

echo "export PATH=$PATH" >> .zshrc

if ! java --version 2> /dev/null ; then
    echo 'install jdk 11'
    # Don't use brew to install java, it brings in many unnecessary libraries that e.g. Boost will link against if
    # present.
    curl https://cdn.azul.com/zulu/bin/zulu11.76.21-ca-jdk11.0.25-macosx_aarch64.tar.gz | sudo tar xvzf - -C /Library/Java/JavaVirtualMachines && \
    sudo mv /Library/Java/JavaVirtualMachines/zulu11.76.21-ca-jdk11.0.25-macosx_aarch64/zulu-11.jdk /Library/Java/JavaVirtualMachines && \
    sudo rm -rf /Library/Java/JavaVirtualMachines/zulu11.76.21-ca-jdk11.0.25-macosx_aarch64
fi

# Install CMake
echo "Install CMake"
curl -v -L https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-macos-universal.tar.gz | tar xvzf - --strip-components 1 -C /Applications
sudo ln -sf /Applications/CMake.app/Contents/bin/cmake /usr/local/bin/cmake

# Make sure all changes are written to disk
sync
