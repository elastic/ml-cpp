#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#
# Temporary mitigation for the eMudhra intermediate CA compromise (InfoSec
# incident "Intrepid Falcon", https://emudhra.we-have-a-problem.com/).
#
# Distrusts the two roots the compromised intermediates chain to in the system
# CA store.  Deliberately broad (roots, not the eight intermediates): remove once
# the images carry fixed ca-certificates packages.
#
# Idempotent and best-effort (never fails the job).  Needs root or passwordless
# sudo; does nothing except on Linux (the macOS images have the roots distrusted
# in the image).  Verify with check_ca_trust.sh.

set -u

ROOT_LABELS=("Comodo AAA Services root" "emSign ECC Root CA - G3")
# SHA-256 of the two roots' DER (lowercase hex).
ROOT_SHA256=(d7a7a0fb5d7e2731d771e9484ebcdef71d5f0c3e0a2948782bc83ee0ea699ef4
             86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b)
ROOT_FILES=("Comodo_AAA_Services_root.crt" "emSign_ECC_Root_CA_-_G3.crt")

case "$(uname -s)" in
    Linux) ;;
    *) echo "distrust_emudhra_cas: not Linux, nothing to do"; exit 0 ;;
esac

if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    SUDO="sudo -n"
else
    echo "distrust_emudhra_cas: WARNING: not root and no passwordless sudo, CA store unchanged"
    exit 0
fi

if [ -d /etc/pki/ca-trust/source ]; then
    # RHEL family: blocklist the roots (directory is "blacklist" up to RHEL 8).
    BUNDLE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
    if [ -d /etc/pki/ca-trust/source/blocklist ] || [ -e /usr/share/doc/ca-certificates/blocklist ]; then
        DIR=/etc/pki/ca-trust/source/blocklist
    else
        DIR=/etc/pki/ca-trust/source/blacklist
    fi
    $SUDO mkdir -p "$DIR" || exit 0
    # Match by SHA-256 fingerprint, not the bundle's label comments.
    tmp=$(mktemp -d)
    awk -v d="$tmp" '/BEGIN CERT/{n++; p=1} p{print > d"/c" n ".pem"} /END CERT/{p=0}' "$BUNDLE" 2>/dev/null
    for pem in "$tmp"/c*.pem; do
        [ -e "$pem" ] || continue
        fp=$(awk '/BEGIN CERT/{p=1;next} /END CERT/{exit} p' "$pem" | base64 -d 2>/dev/null | sha256sum | cut -d' ' -f1)
        case " ${ROOT_SHA256[*]} " in
            *" $fp "*) $SUDO cp "$pem" "$DIR/emudhra-${fp:0:12}.pem" ;;
        esac
    done
    rm -rf "$tmp"
    $SUDO update-ca-trust extract && echo "distrust_emudhra_cas: RHEL-family CA store updated"
elif [ -f /etc/ca-certificates.conf ]; then
    # Debian, Ubuntu, Alpine: a leading "!" disables an entry.
    for f in "${ROOT_FILES[@]}"; do
        $SUDO sed -i "s#^mozilla/${f}\$#!mozilla/${f}#" /etc/ca-certificates.conf
    done
    $SUDO update-ca-certificates >/dev/null 2>&1 && echo "distrust_emudhra_cas: Debian-family CA store updated"
else
    echo "distrust_emudhra_cas: WARNING: unrecognised CA store layout, CA store unchanged"
fi
exit 0
