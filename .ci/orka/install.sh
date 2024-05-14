#!/bin/bash
set -xe
export PATH=/opt/homebrew/bin:/usr/local/bin:$PATH

MACHINE_TYPE=$(uname -m)

if [ ${MACHINE_TYPE} == "x86_64" ]; then
    echo "Running on x86_64"

    PYTHON3_URL=https://www.python.org/ftp/python/3.10.8/python-3.10.8-macos11.pkg
    PYTHON3_FILE=python-3.10.8-macos11.pkg

    GCLOUD_SDK_URL=https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-408.0.1-darwin-x86_64.tar.gz
    GCLOUD_SDK_FILE=google-cloud-cli.tar.gz
elif [ ${MACHINE_TYPE} == "arm64" ]; then
    echo "Running on arm64"

    PYTHON3_URL=https://www.python.org/ftp/python/3.10.8/python-3.10.8-macos11.pkg
    PYTHON3_FILE=python-3.10.8-macos11.pkg

    GCLOUD_SDK_URL=https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-408.0.1-darwin-arm.tar.gz
    GCLOUD_SDK_FILE=google-cloud-cli.tar.gz
else
    echo "Unknown architecture: ${MACHINE_TYPE}"
    exit 1
fi

if ! command -v brew 2> /dev/null ; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

eval "$(brew shellenv)"

if ! java --version 2> /dev/null ; then
  echo 'install jdk 11'
  brew install openjdk@11
  sudo ln -sfn /opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-11.jdk
fi

if ! command -v vault 2> /dev/null ; then
    echo "Install vault"
    brew install vault
fi

if ! command -v jq 2> /dev/null ; then
    echo "Install jq"
    brew install jq
fi

if ! command -v orka-vm-tools 2> /dev/null ; then
    echo "Install orka-vm-tools"
    brew install orka-vm-tools
fi

echo "Install google cloud sdk in home dir"
pushd ~
/bin/bash -c "$(curl -s ${GCLOUD_SDK_URL} --output ${GCLOUD_SDK_FILE})"
tar zxf ${GCLOUD_SDK_FILE}
rm -f ${GCLOUD_SDK_FILE}
popd

echo "Install python 3"
/bin/bash -c "$(curl -s ${PYTHON3_URL} --output ${PYTHON3_FILE})"
sudo installer -pkg ${PYTHON3_FILE} -target /
rm -f ${PYTHON3_FILE}

# Install the gobld bootstrap script
echo "Install gobld bootstrap callout script"
sudo mkdir -p /usr/local/bin
sudo cp /tmp/gobld-bootstrap.sh /usr/local/bin/gobld-bootstrap.sh
sudo chmod +x /usr/local/bin/gobld-bootstrap.sh
sudo cp /tmp/gobld-bootstrap.plist /Library/LaunchDaemons/gobld-bootstrap.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/gobld-bootstrap.plist
sudo cp /tmp/gobld-bootstrap.plist /Users/admin

# Install CMake
echo "Install CMake"
curl -v -L https://github.com/Kitware/CMake/releases/download/v3.23.3/cmake-3.23.3-macos-universal.tar.gz | tar xvzf - --strip-components 1 -C /Applications
sudo ln -sf /Applications/CMake.app/Contents/bin/cmake /usr/local/bin/cmake

# Make sure all changes are written to disk
sync
