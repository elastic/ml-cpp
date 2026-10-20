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
# Runtime mitigation for the eMudhra intermediate CA compromise (InfoSec
# incident "Intrepid Falcon", https://emudhra.we-have-a-problem.com/).
#
# Distrusts the two roots the compromised intermediates chain to ("Comodo AAA
# Services root" and "emSign ECC Root CA - G3") in the system CA store of the
# environment it runs in, for images whose ca-certificates package has not yet
# been updated.  This is deliberately broad - it removes the roots, not just
# the eight intermediates - so it is a stop-gap: remove it once the images
# carry fixed ca-certificates packages.
#
# Idempotent, and best-effort: it never fails the job.  Run
# check_ca_trust.sh afterwards to confirm it took effect.  Needs root (or
# passwordless sudo); on non-Linux hosts it does nothing.

set -u

ROOT_LABELS=("Comodo AAA Services root" "emSign ECC Root CA - G3")
ROOT_FILES=("Comodo_AAA_Services_root.crt" "emSign_ECC_Root_CA_-_G3.crt")

if [ "$(uname -s)" != Linux ]; then
    echo "distrust_emudhra_cas: not Linux, nothing to do"
    exit 0
fi

if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    SUDO="sudo -n"
else
    echo "distrust_emudhra_cas: WARNING: not root and no passwordless sudo, CA store unchanged"
    exit 0
fi

if [ -d /etc/pki/ca-trust/source ]; then
    # RHEL family.  The directory is "blacklist" up to RHEL 8 and "blocklist"
    # from RHEL 9.  The root PEMs are taken from the installed bundle.
    BUNDLE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
    if [ -d /etc/pki/ca-trust/source/blocklist ] || [ -e /usr/share/doc/ca-certificates/blocklist ]; then
        DIR=/etc/pki/ca-trust/source/blocklist
    else
        DIR=/etc/pki/ca-trust/source/blacklist
    fi
    $SUDO mkdir -p "$DIR" || exit 0
    for label in "${ROOT_LABELS[@]}"; do
        out="$DIR/$(echo "$label" | tr ' ' '_').pem"
        # Already blocklisted by an earlier run (the root is then no longer in
        # the extracted bundle, so it can't be extracted again).
        [ -s "$out" ] && continue
        awk -v l="# $label" '$0==l{f=1;next} f&&/BEGIN CERT/{p=1} p{print} p&&/END CERT/{exit}' \
            "$BUNDLE" 2>/dev/null | $SUDO tee "$out" >/dev/null
        if ! grep -q "BEGIN CERT" "$out" 2>/dev/null; then
            # Not in the bundle (e.g. a ca-certificates update already removed it).
            $SUDO rm -f "$out"
        fi
    done
    $SUDO update-ca-trust extract && echo "distrust_emudhra_cas: RHEL-family CA store updated"
elif [ -f /etc/ca-certificates.conf ]; then
    # Debian, Ubuntu and Alpine: a leading "!" disables an entry.
    for f in "${ROOT_FILES[@]}"; do
        $SUDO sed -i "s#^mozilla/${f}\$#!mozilla/${f}#" /etc/ca-certificates.conf
    done
    $SUDO update-ca-certificates >/dev/null 2>&1 && echo "distrust_emudhra_cas: Debian-family CA store updated"
else
    echo "distrust_emudhra_cas: WARNING: unrecognised CA store layout, CA store unchanged"
fi
exit 0
