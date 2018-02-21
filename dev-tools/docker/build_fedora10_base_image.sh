#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# Builds a tarball that can be used to create a Fedora 10 base Docker image.
# Designed to be run on a Fedora 10 machine with the installation DVD mounted.
#
# Must be run by root.
#
# This script is a one-off.  The image built using it should be uploaded to
# Docker Hub and then downloaded from there for use.  This script is only
# in source control for reference.

WORK_DIR=/tmp/root.$$
PACKAGE_DIR=/media/Packages

# Create dummy devices
mkdir -m 755 -p $WORK_DIR/dev
mknod -m 600 $WORK_DIR/dev/console c 5 1
mknod -m 600 $WORK_DIR/dev/initctl p
mknod -m 666 $WORK_DIR/dev/full c 1 7
mknod -m 666 $WORK_DIR/dev/null c 1 3
mknod -m 666 $WORK_DIR/dev/ptmx c 5 2
mknod -m 666 $WORK_DIR/dev/random c 1 8
mknod -m 666 $WORK_DIR/dev/tty c 5 0
mknod -m 666 $WORK_DIR/dev/tty0 c 4 0
mknod -m 666 $WORK_DIR/dev/urandom c 1 9
mknod -m 666 $WORK_DIR/dev/zero c 1 5

# Create a new RPM database in the working directory
mkdir -m 755 -p $WORK_DIR/var/lib/rpm
rpm --root $WORK_DIR --initdb

# Install a fairly minimal set of packages.  It's hard to come up with the
# absolute bare minimum because Fedora 10 pre-dates containerisation and some
# important packages have annoying dependencies.
rpm --install --root $WORK_DIR --excludedocs \
    $PACKAGE_DIR/ConsoleKit-0.3.0-2.fc10.x86_64.rpm \
    $PACKAGE_DIR/ConsoleKit-libs-0.3.0-2.fc10.x86_64.rpm \
    $PACKAGE_DIR/MAKEDEV-3.24-1.x86_64.rpm \
    $PACKAGE_DIR/NetworkManager-0.7.0-0.11.svn4229.fc10.x86_64.rpm \
    $PACKAGE_DIR/NetworkManager-glib-0.7.0-0.11.svn4229.fc10.x86_64.rpm \
    $PACKAGE_DIR/PolicyKit-0.9-3.fc10.x86_64.rpm \
    $PACKAGE_DIR/acl-2.2.47-3.fc10.x86_64.rpm \
    $PACKAGE_DIR/at-3.1.10-26.fc10.x86_64.rpm \
    $PACKAGE_DIR/audit-libs-1.7.8-6.fc10.x86_64.rpm \
    $PACKAGE_DIR/authconfig-5.4.4-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/avahi-autoipd-0.6.22-11.fc10.x86_64.rpm \
    $PACKAGE_DIR/basesystem-10.0-1.noarch.rpm \
    $PACKAGE_DIR/bash-3.2-29.fc10.x86_64.rpm \
    $PACKAGE_DIR/bind-libs-9.5.1-0.8.b2.fc10.x86_64.rpm \
    $PACKAGE_DIR/bzip2-1.0.5-3.fc10.x86_64.rpm \
    $PACKAGE_DIR/bzip2-libs-1.0.5-3.fc10.x86_64.rpm \
    $PACKAGE_DIR/ca-certificates-2008-7.noarch.rpm \
    $PACKAGE_DIR/chkconfig-1.3.38-1.x86_64.rpm \
    $PACKAGE_DIR/compat-db45-4.5.20-5.fc10.x86_64.rpm \
    $PACKAGE_DIR/coreutils-6.12-17.fc10.x86_64.rpm \
    $PACKAGE_DIR/cpio-2.9.90-2.fc10.x86_64.rpm \
    $PACKAGE_DIR/cpuspeed-1.5-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/cracklib-2.8.12-2.x86_64.rpm \
    $PACKAGE_DIR/cracklib-dicts-2.8.12-2.x86_64.rpm \
    $PACKAGE_DIR/cronie-1.2-4.fc10.x86_64.rpm \
    $PACKAGE_DIR/crontabs-1.10-23.fc10.noarch.rpm \
    $PACKAGE_DIR/cryptsetup-luks-1.0.6-6.fc10.x86_64.rpm \
    $PACKAGE_DIR/curl-7.18.2-7.fc10.x86_64.rpm \
    $PACKAGE_DIR/cyrus-sasl-2.1.22-19.fc10.x86_64.rpm \
    $PACKAGE_DIR/cyrus-sasl-lib-2.1.22-19.fc10.x86_64.rpm \
    $PACKAGE_DIR/cyrus-sasl-plain-2.1.22-19.fc10.x86_64.rpm \
    $PACKAGE_DIR/db4-4.7.25-5.fc10.x86_64.rpm \
    $PACKAGE_DIR/dbus-1.2.4-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/dbus-glib-0.76-3.fc10.x86_64.rpm \
    $PACKAGE_DIR/dbus-libs-1.2.4-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/device-mapper-1.02.27-6.fc10.x86_64.rpm \
    $PACKAGE_DIR/device-mapper-libs-1.02.27-6.fc10.x86_64.rpm \
    $PACKAGE_DIR/dhclient-4.0.0-30.fc10.x86_64.rpm \
    $PACKAGE_DIR/dirmngr-1.0.2-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/dmidecode-2.9-1.31.fc10.x86_64.rpm \
    $PACKAGE_DIR/dnsmasq-2.45-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/e2fsprogs-1.41.3-2.fc10.x86_64.rpm \
    $PACKAGE_DIR/e2fsprogs-libs-1.41.3-2.fc10.x86_64.rpm \
    $PACKAGE_DIR/elfutils-libelf-0.137-3.fc10.x86_64.rpm \
    $PACKAGE_DIR/ethtool-6-1.fc9.x86_64.rpm \
    $PACKAGE_DIR/exim-4.69-7.fc10.x86_64.rpm \
    $PACKAGE_DIR/expat-2.0.1-5.x86_64.rpm \
    $PACKAGE_DIR/fedora-release-10-1.noarch.rpm \
    $PACKAGE_DIR/fedora-release-notes-10.0.0-1.noarch.rpm \
    $PACKAGE_DIR/file-libs-4.26-3.fc10.x86_64.rpm \
    $PACKAGE_DIR/filesystem-2.4.19-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/findutils-4.4.0-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/gamin-0.1.9-6.fc10.x86_64.rpm \
    $PACKAGE_DIR/gawk-3.1.5-18.fc10.x86_64.rpm \
    $PACKAGE_DIR/gdbm-1.8.0-29.fc10.x86_64.rpm \
    $PACKAGE_DIR/glib2-2.18.2-3.fc10.x86_64.rpm \
    $PACKAGE_DIR/glibc-2.9-2.x86_64.rpm \
    $PACKAGE_DIR/glibc-common-2.9-2.x86_64.rpm \
    $PACKAGE_DIR/gnupg2-2.0.9-3.fc10.x86_64.rpm \
    $PACKAGE_DIR/gpgme-1.1.7-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/grep-2.5.1a-61.fc10.x86_64.rpm \
    $PACKAGE_DIR/groff-1.18.1.4-16.fc10.x86_64.rpm \
    $PACKAGE_DIR/gzip-1.3.12-7.fc10.x86_64.rpm \
    $PACKAGE_DIR/hal-0.5.12-12.20081027git.fc10.x86_64.rpm \
    $PACKAGE_DIR/hal-info-20081022-1.fc10.noarch.rpm \
    $PACKAGE_DIR/hal-libs-0.5.12-12.20081027git.fc10.x86_64.rpm \
    $PACKAGE_DIR/hdparm-8.6-1.fc9.x86_64.rpm \
    $PACKAGE_DIR/hesiod-3.1.0-13.x86_64.rpm \
    $PACKAGE_DIR/hwdata-0.220-1.fc10.noarch.rpm \
    $PACKAGE_DIR/info-4.12-4.fc10.x86_64.rpm \
    $PACKAGE_DIR/initscripts-8.86-1.x86_64.rpm \
    $PACKAGE_DIR/iproute-2.6.26-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/iputils-20071127-6.fc10.x86_64.rpm \
    $PACKAGE_DIR/irda-utils-0.9.18-5.fc10.x86_64.rpm \
    $PACKAGE_DIR/kbd-1.12-31.fc9.x86_64.rpm \
    $PACKAGE_DIR/keyutils-libs-1.2-3.fc9.x86_64.rpm \
    $PACKAGE_DIR/krb5-libs-1.6.3-16.fc10.x86_64.rpm \
    $PACKAGE_DIR/less-424-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/libacl-2.2.47-3.fc10.x86_64.rpm \
    $PACKAGE_DIR/libattr-2.4.43-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/libcap-2.10-2.fc10.x86_64.rpm \
    $PACKAGE_DIR/libcurl-7.18.2-7.fc10.x86_64.rpm \
    $PACKAGE_DIR/libdaemon-0.13-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/libedit-2.11-1.20080712cvs.fc10.x86_64.rpm \
    $PACKAGE_DIR/libgcc-4.3.2-7.x86_64.rpm \
    $PACKAGE_DIR/libgcrypt-1.4.3-2.fc10.x86_64.rpm \
    $PACKAGE_DIR/libgpg-error-1.6-2.x86_64.rpm \
    $PACKAGE_DIR/libidn-0.6.14-8.x86_64.rpm \
    $PACKAGE_DIR/libksba-1.0.4-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/libnl-1.1-5.fc10.x86_64.rpm \
    $PACKAGE_DIR/libpcap-0.9.8-3.fc10.x86_64.rpm \
    $PACKAGE_DIR/libselinux-2.0.73-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/libsepol-2.0.33-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/libsmbios-2.0.1-2.fc10.1.x86_64.rpm \
    $PACKAGE_DIR/libssh2-0.18-7.fc9.x86_64.rpm \
    $PACKAGE_DIR/libstdc++-4.3.2-7.x86_64.rpm \
    $PACKAGE_DIR/libusb-0.1.12-20.fc10.x86_64.rpm \
    $PACKAGE_DIR/libuser-0.56.9-1.x86_64.rpm \
    $PACKAGE_DIR/libvolume_id-127-3.fc10.x86_64.rpm \
    $PACKAGE_DIR/libx86-1.1-3.fc10.x86_64.rpm \
    $PACKAGE_DIR/libxml2-2.7.2-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/linux-atm-libs-2.5.0-5.x86_64.rpm \
    $PACKAGE_DIR/logrotate-3.7.7-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/lua-5.1.4-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/lzma-4.32.7-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/lzma-libs-4.32.7-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/make-3.81-14.fc10.x86_64.rpm \
    $PACKAGE_DIR/man-1.6f-11.fc10.x86_64.rpm \
    $PACKAGE_DIR/mingetty-1.08-2.fc9.x86_64.rpm \
    $PACKAGE_DIR/module-init-tools-3.5-3.fc10.x86_64.rpm \
    $PACKAGE_DIR/mysql-libs-5.0.67-2.fc10.x86_64.rpm \
    $PACKAGE_DIR/ncurses-5.6-20.20080927.fc10.x86_64.rpm \
    $PACKAGE_DIR/ncurses-base-5.6-20.20080927.fc10.x86_64.rpm \
    $PACKAGE_DIR/ncurses-libs-5.6-20.20080927.fc10.x86_64.rpm \
    $PACKAGE_DIR/net-tools-1.60-91.fc10.x86_64.rpm \
    $PACKAGE_DIR/newt-0.52.10-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/newt-python-0.52.10-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/nspr-4.7.2-2.fc10.x86_64.rpm \
    $PACKAGE_DIR/nss-3.12.2.0-3.fc10.x86_64.rpm \
    $PACKAGE_DIR/nss_db-2.2-43.fc10.x86_64.rpm \
    $PACKAGE_DIR/numactl-2.0.2-2.fc10.x86_64.rpm \
    $PACKAGE_DIR/openldap-2.4.12-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/openssh-5.1p1-3.fc10.x86_64.rpm \
    $PACKAGE_DIR/openssh-clients-5.1p1-3.fc10.x86_64.rpm \
    $PACKAGE_DIR/openssl-0.9.8g-11.fc10.x86_64.rpm \
    $PACKAGE_DIR/pam-1.0.2-2.fc10.x86_64.rpm \
    $PACKAGE_DIR/pam_ccreds-7-3.fc10.x86_64.rpm \
    $PACKAGE_DIR/pam_krb5-2.3.2-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/pam_pkcs11-0.5.3-26.x86_64.rpm \
    $PACKAGE_DIR/passwd-0.75-2.fc9.x86_64.rpm \
    $PACKAGE_DIR/pciutils-libs-3.0.2-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/pcre-7.8-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/perl-5.10.0-49.fc10.x86_64.rpm \
    $PACKAGE_DIR/perl-ExtUtils-MakeMaker-6.36-49.fc10.x86_64.rpm \
    $PACKAGE_DIR/perl-ExtUtils-ParseXS-2.18-49.fc10.x86_64.rpm \
    $PACKAGE_DIR/perl-Module-Pluggable-3.60-49.fc10.x86_64.rpm \
    $PACKAGE_DIR/perl-Pod-Escapes-1.04-49.fc10.x86_64.rpm \
    $PACKAGE_DIR/perl-Pod-Simple-3.07-49.fc10.x86_64.rpm \
    $PACKAGE_DIR/perl-TAP-Harness-3.10-1.fc9.noarch.rpm \
    $PACKAGE_DIR/perl-Test-Harness-3.12-49.fc10.x86_64.rpm \
    $PACKAGE_DIR/perl-devel-5.10.0-49.fc10.x86_64.rpm \
    $PACKAGE_DIR/perl-libs-5.10.0-49.fc10.x86_64.rpm \
    $PACKAGE_DIR/perl-libs-5.10.0-49.fc10.x86_64.rpm \
    $PACKAGE_DIR/perl-version-0.74-49.fc10.x86_64.rpm \
    $PACKAGE_DIR/pinentry-0.7.4-5.fc9.x86_64.rpm \
    $PACKAGE_DIR/pm-utils-1.2.2.1-2.fc10.x86_64.rpm \
    $PACKAGE_DIR/popt-1.13-4.fc10.x86_64.rpm \
    $PACKAGE_DIR/ppp-2.4.4-8.fc10.x86_64.rpm \
    $PACKAGE_DIR/procps-3.2.7-21.fc10.x86_64.rpm \
    $PACKAGE_DIR/psacct-6.3.2-51.fc10.x86_64.rpm \
    $PACKAGE_DIR/psmisc-22.6-8.fc10.x86_64.rpm \
    $PACKAGE_DIR/pth-2.0.7-7.x86_64.rpm \
    $PACKAGE_DIR/pygpgme-0.1-8.fc9.x86_64.rpm \
    $PACKAGE_DIR/python-2.5.2-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/python-iniparse-0.2.3-3.fc9.noarch.rpm \
    $PACKAGE_DIR/python-libs-2.5.2-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/python-urlgrabber-3.0.0-10.fc10.noarch.rpm \
    $PACKAGE_DIR/quota-3.16-5.fc10.x86_64.rpm \
    $PACKAGE_DIR/radeontool-1.5-3.fc9.x86_64.rpm \
    $PACKAGE_DIR/readline-5.2-13.fc9.x86_64.rpm \
    $PACKAGE_DIR/rpm-4.6.0-0.rc1.7.x86_64.rpm \
    $PACKAGE_DIR/rpm-libs-4.6.0-0.rc1.7.x86_64.rpm \
    $PACKAGE_DIR/rpm-python-4.6.0-0.rc1.7.x86_64.rpm \
    $PACKAGE_DIR/rsyslog-3.21.3-4.fc10.x86_64.rpm \
    $PACKAGE_DIR/sed-4.1.5-10.fc9.x86_64.rpm \
    $PACKAGE_DIR/setup-2.7.4-1.fc10.noarch.rpm \
    $PACKAGE_DIR/shadow-utils-4.1.2-8.fc10.x86_64.rpm \
    $PACKAGE_DIR/slang-2.1.4-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/smbios-utils-2.0.1-2.fc10.1.x86_64.rpm \
    $PACKAGE_DIR/specspo-16-1.noarch.rpm \
    $PACKAGE_DIR/sqlite-3.5.9-2.fc10.x86_64.rpm \
    $PACKAGE_DIR/sudo-1.6.9p17-2.fc10.x86_64.rpm \
    $PACKAGE_DIR/sysvinit-tools-2.86-24.x86_64.rpm \
    $PACKAGE_DIR/tcp_wrappers-7.6-53.fc10.x86_64.rpm \
    $PACKAGE_DIR/tcp_wrappers-libs-7.6-53.fc10.x86_64.rpm \
    $PACKAGE_DIR/tmpwatch-2.9.13-2.x86_64.rpm \
    $PACKAGE_DIR/tzdata-2008h-1.fc10.noarch.rpm \
    $PACKAGE_DIR/udev-127-3.fc10.x86_64.rpm \
    $PACKAGE_DIR/upstart-0.3.9-19.fc10.x86_64.rpm \
    $PACKAGE_DIR/usermode-1.98.1-1.x86_64.rpm \
    $PACKAGE_DIR/util-linux-ng-2.14.1-3.fc10.x86_64.rpm \
    $PACKAGE_DIR/vbetool-1.1-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/vim-minimal-7.2.025-2.fc10.x86_64.rpm \
    $PACKAGE_DIR/wget-1.11.4-1.fc10.x86_64.rpm \
    $PACKAGE_DIR/wpa_supplicant-0.6.4-2.fc10.x86_64.rpm \
    $PACKAGE_DIR/yum-3.2.20-3.fc10.noarch.rpm \
    $PACKAGE_DIR/yum-metadata-parser-1.1.2-10.fc10.x86_64.rpm \
    $PACKAGE_DIR/zlib-1.2.3-18.fc9.x86_64.rpm

# Add network config
cat > $WORK_DIR/etc/sysconfig/network <<EOF
NETWORKING=yes
HOSTNAME=localhost.localdomain
EOF

# Delete docs and man pages
rm -rf $WORK_DIR/usr/share/{man,doc,info,gnome/help}
# Delete weak password checker
rm -rf $WORK_DIR/usr/share/cracklib
# Clean ldconfig
rm -rf $WORK_DIR/etc/ld.so.cache $WORK_DIR/var/cache/ldconfig/*

tar --numeric-owner -C $WORK_DIR -f fedora10.tar -c .

rm -rf $WORK_DIR

echo 'Now transfer fedora10.tar to the machine running docker and create the'
echo 'image as follows:'
echo ''
echo 'docker import fedora10.tar fedora:10'

