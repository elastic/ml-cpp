# Create Orka Images via Packer

## Requirements

- Packer
- Vault
- Orka VPN

## Files

- `install.sh` The script that does the software installs on the image
- `orka-macos-12-arm.pkr.hcl` The packer definition for a MacOS 12 ARM builder image


## Set Up Packer

If you haven't run these before, run the following once so packer downloads the `vault` integration:

```
packer init orka-macos-12-arm.pkr.hcl
```
or
```
packer init orka-macos-12-x86_64.pkr.hcl
```

## Build

Make sure you are connected to the Orka VM.

Packer requires access to secrets in vault, where VAULT_ADDR=https://vault-ci-prod.elastic.dev and VAULT_TOKEN must be set appropriately in the environment.

Run the following to create the image (MacOS 12 ARM in this example):

```
packer build orka-macos-12-arm.pkr.hcl
```

## Versioning

The name of the resulting images are hard-coded (currently), and end in a sequence number (e.g., `001`).  When creating a new image, bump the sequence number, otherwise you won't be able to create an image if it already exists.

## Source Images

The source images used for the MacOS builds are slightly modified copies of the standard Orka images, e.g. 90GBMontereySSH.orkasi:

The source images are named:
 * `ml-macos-12-base-arm-fundamental.orkasi`
 * `ml-macos-12-base-x86_64-fundamental.img`

The source image only has the following changes on it:
    * Adding passwordless `sudo` for the default `admin` user
    * Configured `admin` user to be automatically logged in
    * Installed Xcode command line tools version 13 (by running `clang++ --version` and clicking through the dialogues)

## Packer Install Steps

The packer script does the following:
    * Install Brew `4.2.21`
    * Install JDK `11.0.23`
    * Install python3
    * Install vault
    * Install jq
    * Install Google Cloud SDK into `~admin/google-cloud-sdk/`
    * Install CMake 3.23.3
    * Install `gobld-bootstrap.sh` script to run at system startup
        * This script pulls down and runs another script from a static location to do the following:
            * Unseal one-time vault token from gobld
            * Install and run the latest `buildkite-agent`

## Caveats

* Prior to the dependency on PyTorch 2.3.1 we only needed Orka for ARM builds (CI and dependencies), x86_64 builds were
  performed via cross-compilation. However, PyTorch 2.3.1 now requires a more modern version of `clang` that our cross
  compilation framework provided. As a suitable Orka base image is available for x86_64, it is now simpler to compile
  natively for that architecture.
