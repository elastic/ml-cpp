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
# sudo; does nothing except on Linux and macOS.  Verify with check_ca_trust.sh.

set -u

ROOT_LABELS=("Comodo AAA Services root" "emSign ECC Root CA - G3")
ROOT_FILES=("Comodo_AAA_Services_root.crt" "emSign_ECC_Root_CA_-_G3.crt")

case "$(uname -s)" in
    Linux|Darwin) ;;
    *) echo "distrust_emudhra_cas: not Linux or macOS, nothing to do"; exit 0 ;;
esac

if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    SUDO="sudo -n"
else
    echo "distrust_emudhra_cas: WARNING: not root and no passwordless sudo, CA store unchanged"
    exit 0
fi

if [ "$(uname -s)" = Darwin ]; then
    # Deny the roots in the admin trust settings (the Keychain).  A headless CI
    # session can't answer the authorization prompt, even as root, so on that
    # error temporarily allow the trust-settings right, retry, then restore it.
    TRUST_RIGHT=com.apple.trust-settings.admin
    right_backup=""
    # Restore the right on any exit or signal, so it can't be left open.
    restore_right() {
        [ -n "$right_backup" ] || return 0
        if [ -s "$right_backup" ]; then
            $SUDO security authorizationdb write "$TRUST_RIGHT" < "$right_backup" >/dev/null 2>&1
        fi
        rm -f "$right_backup"
        right_backup=""
    }
    trap restore_right EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    deny_trust() {
        $SUDO security add-trusted-cert -d -r deny -k /Library/Keychains/System.keychain "$1" 2>&1
    }
    for name in "AAA Certificate Services" "emSign ECC Root CA - G3"; do
        pem=$(mktemp -t distrust_emudhra)
        security find-certificate -a -c "$name" -p /System/Library/Keychains/SystemRootCertificates.keychain > "$pem" 2>/dev/null
        if [ -s "$pem" ]; then
            out=$(deny_trust "$pem"); rc=$?
            if [ $rc -ne 0 ] && echo "$out" | grep -q "no user interaction"; then
                if [ -z "$right_backup" ]; then
                    right_backup=$(mktemp -t distrust_right)
                    $SUDO security authorizationdb read "$TRUST_RIGHT" > "$right_backup" 2>/dev/null
                    $SUDO security authorizationdb write "$TRUST_RIGHT" allow >/dev/null 2>&1
                fi
                out=$(deny_trust "$pem"); rc=$?
            fi
            if [ $rc -eq 0 ]; then
                echo "distrust_emudhra_cas: macOS denied trust for '$name'"
            else
                echo "distrust_emudhra_cas: WARNING: could not deny trust for '$name': $out"
            fi
        else
            echo "distrust_emudhra_cas: '$name' not in the system roots, skipping"
        fi
        rm -f "$pem"
    done
    restore_right

    # The built-in /usr/bin/curl reads /etc/ssl/cert.pem, not the Keychain, so
    # drop the roots from it too (it is a series of "=== <subject>" blocks).
    BUNDLE=/etc/ssl/cert.pem
    PATTERN='^=== .*CN=(AAA Certificate Services|emSign ECC Root CA - G3)$'
    if [ -f "$BUNDLE" ] && grep -Eq "$PATTERN" "$BUNDLE"; then
        filtered=$(mktemp -t distrust_bundle)
        awk -v pat="$PATTERN" '
            $0 ~ pat { skip = 1 }
            skip { if ($0 ~ /^-----END CERTIFICATE-----$/) skip = 0; next }
            { print }' "$BUNDLE" > "$filtered"
        before=$(grep -c "BEGIN CERTIFICATE" "$BUNDLE")
        after=$(grep -c "BEGIN CERTIFICATE" "$filtered")
        # Only replace the bundle if just the expected certificates were removed.
        if [ "$after" -gt 0 ] && [ "$after" -lt "$before" ] && [ $((before - after)) -le 2 ]; then
            [ -e "$BUNDLE.pre-emudhra" ] || $SUDO cp -p "$BUNDLE" "$BUNDLE.pre-emudhra"
            if $SUDO sh -c "cat '$filtered' > '$BUNDLE'"; then
                echo "distrust_emudhra_cas: removed $((before - after)) root(s) from $BUNDLE"
            else
                echo "distrust_emudhra_cas: WARNING: could not update $BUNDLE"
            fi
        else
            echo "distrust_emudhra_cas: WARNING: unexpected result filtering $BUNDLE ($before -> $after certificates), left unchanged"
        fi
        rm -f "$filtered"
    fi
elif [ -d /etc/pki/ca-trust/source ]; then
    # RHEL family: blocklist the roots (directory is "blacklist" up to RHEL 8).
    BUNDLE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
    if [ -d /etc/pki/ca-trust/source/blocklist ] || [ -e /usr/share/doc/ca-certificates/blocklist ]; then
        DIR=/etc/pki/ca-trust/source/blocklist
    else
        DIR=/etc/pki/ca-trust/source/blacklist
    fi
    $SUDO mkdir -p "$DIR" || exit 0
    for label in "${ROOT_LABELS[@]}"; do
        out="$DIR/$(echo "$label" | tr ' ' '_').pem"
        # Already blocklisted by an earlier run.
        [ -s "$out" ] && continue
        awk -v l="# $label" '$0==l{f=1;next} f&&/BEGIN CERT/{p=1} p{print} p&&/END CERT/{exit}' \
            "$BUNDLE" 2>/dev/null | $SUDO tee "$out" >/dev/null
        if ! grep -q "BEGIN CERT" "$out" 2>/dev/null; then
            # Not in the bundle (already removed by an update).
            $SUDO rm -f "$out"
        fi
    done
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
