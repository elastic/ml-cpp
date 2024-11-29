packer {
  required_plugins {
    macstadium-orka = {
      version = "= 2.3.0"
      source  = "github.com/macstadium/macstadium-orka"
    }
  }
}

locals {
  orka_endpoint = vault("secret/ci/elastic-ml-cpp/orka", "orka_endpoint")
  orka_user     = vault("secret/ci/elastic-ml-cpp/orka", "orka_user")
  orka_password = vault("secret/ci/elastic-ml-cpp/orka", "orka_password")
  ssh_username  = vault("secret/ci/elastic-ml-cpp/orka", "ssh_username")
  ssh_password  = vault("secret/ci/elastic-ml-cpp/orka", "ssh_password")
  sensitive     = true
}

source "macstadium-orka" "image" {
  source_image     = "generic-13-ventura-arm-002.orkasi"
  image_name       = "ml-macos-13-arm-002.orkasi"
  orka_endpoint    = local.orka_endpoint
  orka_user        = local.orka_user
  orka_password    = local.orka_password
  ssh_username     = local.ssh_username
  ssh_password     = local.ssh_password
  orka_vm_cpu_core = 4
  no_delete_vm     = false
}

build {
  sources = [
    "macstadium-orka.image"
  ]
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
    inline = [
      "chmod u+x /tmp/third_party_deps.sh",
      "/tmp/third_party_deps.sh",
    ]
  }
}
