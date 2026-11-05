# Adds the distrust of the compromised eMudhra roots (InfoSec incident "Intrepid
# Falcon") to the existing macOS 12 image, as a thin layer over it: only
# distrust_emudhra_cas.sh runs, so the toolchain installed by install.sh is
# unchanged. Homebrew no longer supports macOS 12, so rebuilding from the base
# image is avoided.
packer {
  required_plugins {
    macstadium-orka = {
      version = ">= 3.0.0, < 4.0.0"
      source  = "github.com/macstadium/macstadium-orka"
    }
  }
}

locals {
  orka_endpoint   = vault("secret/ci/elastic-ml-cpp/orka3", "orka_endpoint")
  orka_auth_token = vault("secret/ci/elastic-ml-cpp/orka3", "packer_service_account_token")
  ssh_username    = vault("secret/ci/elastic-ml-cpp/orka3", "ssh_username")
  ssh_password    = vault("secret/ci/elastic-ml-cpp/orka3", "ssh_password")
}

source "macstadium-orka" "image" {
  source_image      = "ml-macos-12-arm-001.orkasi"
  image_name        = "ml-macos-12-arm-003.orkasi"
  orka_endpoint     = local.orka_endpoint
  orka_auth_token   = local.orka_auth_token
  ssh_username      = local.ssh_username
  ssh_password      = local.ssh_password
  orka_vm_cpu_core  = 4
  no_delete_vm      = false
  image_description = "macOS 12 Monterey image for ML CI testing, with the compromised eMudhra roots distrusted"
}

build {
  sources = [
    "macstadium-orka.image"
  ]
  # passwordless sudo is required for the distrust script to work without manual intervention
  provisioner "shell" {
    inline = [
      "echo '${local.ssh_password}' | sudo -S sh -c \"echo '${local.ssh_username} ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/${local.ssh_username}-nopasswd\"",
      "echo '${local.ssh_password}' | sudo -S chmod 0440 /etc/sudoers.d/${local.ssh_username}-nopasswd",
    ]
  }
  provisioner "file" {
    source      = "distrust_emudhra_cas.sh"
    destination = "/tmp/distrust_emudhra_cas.sh"
  }
  # Report what is there, distrust the roots, then fail the build if any store still trusts them.
  provisioner "shell" {
    inline = [
      "chmod u+x /tmp/distrust_emudhra_cas.sh",
      "/tmp/distrust_emudhra_cas.sh report",
      "/tmp/distrust_emudhra_cas.sh distrust",
      "/tmp/distrust_emudhra_cas.sh verify",
    ]
  }
}
