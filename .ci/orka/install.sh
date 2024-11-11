#!/bin/bash
set -xe
export PATH=/opt/homebrew/bin:/usr/local/bin:$PATH

MACHINE_TYPE=$(uname -m)

PYTHON3_URL=https://www.python.org/ftp/python/3.10.10/python-3.10.10-macos11.pkg
PYTHON3_FILE=python-3.10.10-macos11.pkg

GCLOUD_SDK_URL=https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-408.0.1-darwin-x86_64.tar.gz
GCLOUD_SDK_FILE=google-cloud-cli.tar.gz

if [ ${MACHINE_TYPE} == "x86_64" ]; then
    echo "Running on x86_64"

elif [ ${MACHINE_TYPE} == "arm64" ]; then
    echo "Running on arm64"

else
    echo "Unknown architecture: ${MACHINE_TYPE}"
    exit 1
fi

if ! command -v brew 2> /dev/null ; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

eval "$(brew shellenv)"
echo 'export PATH="$HOMEBREW_PREFIX/bin:$PATH"' >> ~/.zshrc
echo 'export PATH="$HOMEBREW_PREFIX/bin:$PATH"' >> ~/.bash_profile
export PATH="$HOMEBREW_PREFIX/bin:$PATH"

if ! java --version 2> /dev/null ; then
  echo 'install jdk 11'
  # Don't use brew to install java, it brings in many unnecessary libraries that e.g. Boost will link against if
  # present.
  curl https://cdn.azul.com/zulu/bin/zulu11.76.21-ca-jdk11.0.25-macosx_aarch64.tar.gz | sudo tar xvzf - -C /Library/Java/JavaVirtualMachines && \
  sudo mv /Library/Java/JavaVirtualMachines/zulu11.76.21-ca-jdk11.0.25-macosx_aarch64/zulu-11.jdk /Library/Java/JavaVirtualMachines && \
  sudo rm -rf /Library/Java/JavaVirtualMachines/zulu11.76.21-ca-jdk11.0.25-macosx_aarch64
fi

if ! command -v vault 2> /dev/null ; then
    echo "install vault"
    brew tap hashicorp/tap
    brew install hashicorp/tap/vault
fi

if ! command -v jq 2> /dev/null ; then
    echo "Install jq"
    brew install jq
fi

if ! command -v orka-vm-tools 2> /dev/null ; then
    echo "Install orka-vm-tools"
    brew install orka-vm-tools
fi

if ! command -v buildkite-agent 2> /dev/null ; then
  echo "Downloading buildkite-agent..."
  curl -sL https://raw.githubusercontent.com/elastic/buildkite-agent/main/install.sh -o /tmp/buildkite-install.sh
  echo "Done downloading buildkite-agent..."

  bash -c "$(cat /tmp/buildkite-install.sh)" > /tmp/insty 2>&1 || true
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
curl -v -L https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-macos-universal.tar.gz | tar xvzf - --strip-components 1 -C /Applications
sudo ln -sf /Applications/CMake.app/Contents/bin/cmake /usr/local/bin/cmake

# Make sure all changes are written to disk
sync
