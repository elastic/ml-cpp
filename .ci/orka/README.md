# Create Orka Images via Packer

## Requirements

- Packer
- Vault
- Orka VPN

## Files

- `install.sh` The script that does the software installs on the image
- `orka-macos-14-arm.pkr.hcl` The packer definition for a MacOS 14 ARM builder image


## Set Up Packer

If you haven't run these before, run the following once so packer downloads the `vault` integration:

```
packer init orka-macos-14-arm.pkr.hcl
```

## Build

Make sure you are connected to the Orka VM.

Packer requires access to secrets in vault, where VAULT_ADDR=https://vault-ci-prod.elastic.dev and VAULT_TOKEN must be set appropriately in the environment.

Run the following to create the image (MacOS 14 ARM in this example):

```
packer build orka-macos-14-arm.pkr.hcl
```

## Versioning

The name of the resulting images are hard-coded (currently), and end in a sequence number (e.g., `001`).  When creating a new image, bump the sequence number, otherwise you won't be able to create an image if it already exists.

## Source Images

We make use of an image - `generic-14-sonoma-arm-001.orkasi` - that is configured such that it:

    * Adds passwordless `sudo` for the default `admin` user
    * Configures `the admin` user to be automatically logged in
    * Installs Xcode command line tools version 14 (by running `clang++ --version` and clicking through the dialogues)

The generic image has the following packages installed:

 * Google Cloud SDK into `~admin/google-cloud-sdk/`

## Packer Install Steps

The ML packer scripts do the following:
 * brew `4.0.28`
 * vault `1.14.0`
 * python3 `3.10.8`
 * jq `1.6`
 * orka-vm-tools
 * `gobld-bootstrap.sh` script to run at system startup
    * This script pulls down and runs another script from a static location to do the following:
      * Unseal one-time vault token from gobld
      * Install and run the latest `buildkite-agent`
 * Install JDK `11.0.25`
 * Install CMake `3.30.5`
 * Install Boost `1.86.0` from source
 * Install PyTorch `2.7.1` from source

## Caveats

* As of version 9.0.0 support for macos x86_64 builds has been dropped. It is necessary to checkout and work on previous branches in order to maintain x86_64 Orka VMs.
