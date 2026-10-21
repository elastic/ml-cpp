#!/bin/bash
# Distrusts the compromised eMudhra roots (InfoSec incident "Intrepid Falcon",
# https://emudhra.we-have-a-problem.com/) in the image.  Remove once the base
# image and Homebrew carry the fix.
#
#   early   before install.sh: Keychain and /etc/ssl/cert.pem, so everything the
#           image build downloads is fetched with the roots distrusted
#   late    after the installs: JDK cacerts, Homebrew and certifi bundles
#   verify  fail unless every store is clean
#
# The two roots are matched by SHA-256 fingerprint.
set -e
export PATH=/opt/homebrew/bin:/usr/local/bin:$PATH

SUDO="${SUDO-sudo}"
ROOT_NAMES=("AAA Certificate Services" "emSign ECC Root CA - G3")
ROOT_SHA256="d7a7a0fb5d7e2731d771e9484ebcdef71d5f0c3e0a2948782bc83ee0ea699ef4 86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b"
TEST_HOSTS="dv-g3b-emudhra.we-have-a-problem.com dv-g3b-aaa-emudhra.we-have-a-problem.com"
SYSTEM_BUNDLE=/etc/ssl/cert.pem

# pem_bundle count|remove FILE: print how many compromised roots a PEM bundle
# holds; "remove" also rewrites it without them and their header/comment lines.
pem_bundle() {
    local mode="$1" file="$2" sudo=""
    [ "$mode" = remove ] && sudo="$SUDO"
    $sudo python3 - "$mode" "$file" "$ROOT_SHA256" <<'PY'
import base64, hashlib, os, re, sys
mode, path, bad = sys.argv[1], os.path.realpath(sys.argv[2]), set(sys.argv[3].split())
text = open(path, encoding="utf-8", errors="surrogateescape").read()
pem = re.compile(r"-----BEGIN CERTIFICATE-----(.*?)-----END CERTIFICATE-----\n?", re.S)
out, pos, found = [], 0, 0
for m in pem.finditer(text):
    if hashlib.sha256(base64.b64decode("".join(m.group(1).split()))).hexdigest() in bad:
        found += 1
    else:
        out.append(text[pos:m.end()])
    pos = m.end()
out.append(text[pos:])
if mode == "remove" and found:
    with open(path, "w", encoding="utf-8", errors="surrogateescape") as f:
        f.write("".join(out))
print(found)
PY
}

# keystore_aliases KEYTOOL KEYSTORE: aliases of the compromised roots in a keystore.
keystore_aliases() {
    "$1" -list -v -keystore "$2" -storepass changeit 2>/dev/null | awk -v bad="$ROOT_SHA256" '
        BEGIN { n = split(bad, b, " "); for (i = 1; i <= n; i++) want[b[i]] = 1 }
        /^Alias name:/ { alias = $0; sub(/^Alias name: */, "", alias) }
        /SHA256:/ { fp = tolower($2); gsub(":", "", fp); if (fp in want) print alias }'
}

# Deny the roots in the admin trust settings.  A headless session can't answer
# the authorization prompt, even as root, so on that error temporarily allow the
# trust-settings right, retry, then restore it.
deny_in_keychain() {
    local right=com.apple.trust-settings.admin backup="" name pem out rc
    for name in "${ROOT_NAMES[@]}"; do
        pem=$(mktemp -t distrust)
        security find-certificate -a -c "$name" -p /System/Library/Keychains/SystemRootCertificates.keychain > "$pem"
        if [ ! -s "$pem" ]; then echo "'$name' is not in the system roots"; rm -f "$pem"; continue; fi
        out=$($SUDO security add-trusted-cert -d -r deny -k /Library/Keychains/System.keychain "$pem" 2>&1) && rc=0 || rc=$?
        if [ $rc -ne 0 ] && echo "$out" | grep -q "no user interaction"; then
            if [ -z "$backup" ]; then
                backup=$(mktemp -t right)
                $SUDO security authorizationdb read "$right" > "$backup"
                $SUDO security authorizationdb write "$right" allow
            fi
            out=$($SUDO security add-trusted-cert -d -r deny -k /Library/Keychains/System.keychain "$pem" 2>&1) && rc=0 || rc=$?
        fi
        rm -f "$pem"
        [ $rc -eq 0 ] || { echo "Could not deny trust for '$name': $out"; return 1; }
        echo "Denied trust for '$name'"
    done
    if [ -n "$backup" ]; then
        $SUDO security authorizationdb write "$right" < "$backup"
        rm -f "$backup"
    fi
}

# Bundles that exist once the installs are done: Homebrew's, and every certifi
# copy (Python's and pip's vendored one) in the known site-packages locations.
late_bundles() {
    {
        if command -v brew >/dev/null; then
            ls "$(brew --prefix)"/etc/ca-certificates/cert.pem "$(brew --prefix)"/etc/openssl@*/cert.pem 2>/dev/null || true
        fi
        find /opt/homebrew/lib/python3*/site-packages \
             /opt/homebrew/Cellar/python@*/*/Frameworks/Python.framework/Versions/*/lib/python3*/site-packages \
             /usr/local/lib/python3*/site-packages /Library/Python/*/site-packages \
             /Library/Frameworks/Python.framework/Versions/*/lib/python3*/site-packages \
            -maxdepth 4 -name cacert.pem 2>/dev/null || true
    } | while read -r f; do python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$f"; done | sort -u
}

java_homes() { ls -d /Library/Java/JavaVirtualMachines/*/Contents/Home 2>/dev/null || true; }

early() {
    deny_in_keychain
    [ "$(pem_bundle count "$SYSTEM_BUNDLE")" = 0 ] || pem_bundle remove "$SYSTEM_BUNDLE" >/dev/null
}

late() {
    if command -v brew >/dev/null; then
        brew update --quiet && brew upgrade ca-certificates || echo "brew could not update ca-certificates, filtering the bundle instead"
    fi
    local f jh
    for f in $(late_bundles); do
        [ "$(pem_bundle count "$f")" = 0 ] || { pem_bundle remove "$f" >/dev/null; echo "Removed the roots from $f"; }
    done
    for jh in $(java_homes); do
        # Aliases can contain spaces (e.g. "comodoaaaca [jdk]"): read them line by line.
        keystore_aliases "$jh/bin/keytool" "$jh/lib/security/cacerts" | while IFS= read -r alias; do
            $SUDO "$jh/bin/keytool" -delete -alias "$alias" -keystore "$jh/lib/security/cacerts" -storepass changeit
            echo "Removed '$alias' from $jh"
        done
    done
}

verify() {
    local bad=0 f jh h pem name rc
    for f in $SYSTEM_BUNDLE $(late_bundles); do
        [ "$(pem_bundle count "$f")" = 0 ] || { echo "FAIL: compromised root in $f"; bad=1; }
    done
    for jh in $(java_homes); do
        [ -z "$(keystore_aliases "$jh/bin/keytool" "$jh/lib/security/cacerts")" ] || { echo "FAIL: compromised root in $jh"; bad=1; }
    done
    for name in "${ROOT_NAMES[@]}"; do
        pem=$(mktemp -t verify)
        security find-certificate -a -c "$name" -p /System/Library/Keychains/SystemRootCertificates.keychain > "$pem"
        if [ -s "$pem" ] && security verify-cert -c "$pem" -p basic >/dev/null 2>&1; then
            echo "FAIL: the Keychain still trusts '$name'"; bad=1
        fi
        rm -f "$pem"
    done
    for h in $TEST_HOSTS; do
        /usr/bin/curl -sS -m 30 -o /dev/null "https://$h/" >/dev/null 2>&1 && rc=0 || rc=$?
        [ $rc -eq 60 ] || { echo "FAIL: curl exit $rc for $h (expected 60: certificate rejected)"; bad=1; }
    done
    [ $bad -eq 0 ] && echo "All stores distrust the compromised roots"
    return $bad
}

# Run only when executed, so the functions can be sourced for testing.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    case "${1:-}" in
        early|late|verify) "$1" ;;
        *) echo "Usage: $0 early|late|verify" >&2; exit 2 ;;
    esac
fi
