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

* As of version 8.18 support for macos x86_64 builds has been dropped. It is necessary to checkout and work on previous branches in order to maintain x86_64 Orka VMs.
