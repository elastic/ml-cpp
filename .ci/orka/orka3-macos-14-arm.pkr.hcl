packer {
  required_plugins {
    macstadium-orka = {
      version = ">= 3.0.0, < 4.0.0"
      source  = "github.com/macstadium/macstadium-orka"
    }
  }
}

locals {
  orka_endpoint = vault("secret/security-sre-team/ci/orka3", "orka_endpoint")
  orka_auth_token = vault("secret/security-sre-team/ci/orka3", "packer_service_account_token")
  ssh_username = vault("secret/security-sre-team/ci/orka", "ssh_username")
  ssh_password = vault("secret/security-sre-team/ci/orka", "ssh_password")
  sensitive  = true
}

variable "admin_password" {
  type      = string
  default   = "dontUseThisDefault"
  sensitive = true
}

source "macstadium-orka" "image" {
  source_image     = "generic-14-sonoma-arm"
  image_name       = "clind-ml-macos-14-arm"
  orka_endpoint    = local.orka_endpoint
  orka_auth_token  = local.orka_auth_token
  ssh_username     = local.ssh_username
  ssh_password     = local.ssh_password
  orka_vm_cpu_core = 4
  no_delete_vm     = false
  image_description = "macOS 14 Sonoma image for ML CI testing"
}

build {
  sources = [
    "macstadium-orka.image"
  ]
  # passwordless sudo is required for the install script to work without manual intervention
  provisioner "shell" {
    inline = [
      "echo '${local.ssh_password}' | sudo -S sh -c \"echo '${local.ssh_username} ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/${local.ssh_username}-nopasswd\"",
      "echo '${local.ssh_password}' | sudo -S chmod 0440 /etc/sudoers.d/${local.ssh_username}-nopasswd",
    ]
  }
  provisioner "file" {
    source = "install.sh"
    destination = "/tmp/install.sh"
  }
  provisioner "file" {
    source = "third_party_deps.sh"
    destination = "/tmp/third_party_deps.sh"
  }
  provisioner "file" {
    source = "gobld-bootstrap.sh"
    destination = "/tmp/gobld-bootstrap.sh"
  }
  provisioner "file" {
    source = "gobld-bootstrap.plist"
    destination = "/tmp/gobld-bootstrap.plist"
  }
  provisioner "shell" {
    inline = [
      "chmod u+x /tmp/install.sh",
      "/tmp/install.sh",
    ]
  }
  provisioner "shell" {
    timeout = "2h"
    inline = [
      "chmod u+x /tmp/third_party_deps.sh",
      "/tmp/third_party_deps.sh",
    ]
  }
}
