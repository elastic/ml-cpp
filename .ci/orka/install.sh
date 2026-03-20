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

if ! command -v brew 2> /dev/null ; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

eval "$(brew shellenv)"
echo 'export PATH="$HOMEBREW_PREFIX/bin:$PATH"' >> ~/.zshrc
echo 'export PATH="$HOMEBREW_PREFIX/bin:$PATH"' >> ~/.bash_profile
export PATH="$HOMEBREW_PREFIX/bin:$PATH"

if ! command -v vault 2> /dev/null ; then
    echo "install vault"
    brew tap hashicorp/tap
    brew install hashicorp/tap/vault
fi

if [ ! -f "/opt/homebrew/bin/python3" ]; then
    echo "installing homebrew python"
    brew install python
fi

if ! command -v jq 2> /dev/null ; then
    echo "install jq"
    brew install jq
fi

if ! command -v orka-vm-tools 2> /dev/null ; then
    echo "install orka-vm-tools"
    brew install orka-vm-tools
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

# 1. Install CMake (Fixing the sudo and pathing order)
echo "Install CMake"
# Use sudo for the tar command so it has permission to write to /Applications
curl -L https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-macos-universal.tar.gz | sudo tar xvzf - --strip-components 1 -C /Applications

# Ensure /usr/local/bin exists BEFORE creating the symlink
sudo mkdir -p /usr/local/bin
sudo ln -sf /Applications/CMake.app/Contents/bin/cmake /usr/local/bin/cmake

# 2. Install the bootstrap script
sudo cp /tmp/gobld-bootstrap.sh /usr/local/bin/gobld-bootstrap.sh
sudo chmod +x /usr/local/bin/gobld-bootstrap.sh

# 3. FIX: Rename plist to match its internal Label: co.elastic.gobld-bootstrap
# This is mandatory to prevent Error 5
DEST_PLIST="/Library/LaunchDaemons/co.elastic.gobld-bootstrap.plist"
sudo cp /tmp/gobld-bootstrap.plist "$DEST_PLIST"
sudo chown root:wheel "$DEST_PLIST"
sudo chmod 644 "$DEST_PLIST"

# 4. Load the service 
# Using legacy 'load' is more reliable than 'bootstrap' in Packer/Orka SSH sessions
echo "Loading the LaunchDaemon..."
sudo launchctl unload -w "$DEST_PLIST" 2>/dev/null || true
sudo launchctl load -w "$DEST_PLIST"

# Copy to home for reference as requested
cp /tmp/gobld-bootstrap.plist /Users/admin/co.elastic.gobld-bootstrap.plist

# Make sure all changes are written to disk
sync
