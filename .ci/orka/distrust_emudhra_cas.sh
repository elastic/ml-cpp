#!/bin/bash
# Distrusts the compromised eMudhra roots (InfoSec incident "Intrepid Falcon",
# https://emudhra.we-have-a-problem.com/) in the image.  Remove once the base
# image and Homebrew carry the fix.
#
#   distrust  deny the roots in the Keychain and remove them from every CA store
#             found: the system bundle, other PEM bundles (Homebrew, certifi,
#             vendored copies) and JDK cacerts.  Idempotent: run it before the
#             installs, so everything the image build downloads is fetched with
#             the roots distrusted, and again after them, for the stores they create
#   verify    fail unless every store is clean
#   report    list every CA store found and how many compromised roots it holds
#
# The two roots are matched by SHA-256 fingerprint.  Needs python3 (the Xcode
# command line tools provide /usr/bin/python3).
set -e
export PATH=/opt/homebrew/bin:/usr/local/bin:$PATH

SUDO="${SUDO-sudo}"
ROOT_NAMES=("AAA Certificate Services" "emSign ECC Root CA - G3")
# Overridable for tests.
ROOT_SHA256="${ROOT_SHA256:-d7a7a0fb5d7e2731d771e9484ebcdef71d5f0c3e0a2948782bc83ee0ea699ef4 86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b}"
TEST_HOSTS="dv-g3b-emudhra.we-have-a-problem.com dv-g3b-aaa-emudhra.we-have-a-problem.com"
SYSTEM_BUNDLE="${SYSTEM_BUNDLE:-/etc/ssl/cert.pem}"
# Where to look for other CA stores.  /Users covers the admin user's home, e.g.
# ~/google-cloud-sdk, which ships its own certifi.
SWEEP_DIRS="${SWEEP_DIRS:-/opt/homebrew /usr/local /Library /Applications /Users}"
BUNDLE_NAMES="cacert.pem cert.pem ca-bundle.crt ca-bundle.pem ca-certificates.crt tls-ca-bundle.pem cacerts.txt"

require_python() {
    command -v python3 >/dev/null && python3 -c 'import sys' ||
        { echo "python3 is not usable (the Xcode command line tools should provide /usr/bin/python3)" >&2; return 1; }
}

# pem_bundle count|remove FILE: print how many compromised roots a PEM bundle
# holds; "remove" also rewrites it without them and their header/comment lines.
pem_bundle() {
    local mode="$1" file="$2" sudo="" py
    [ "$mode" = remove ] && sudo="$SUDO"
    py=$(command -v python3)
    $sudo "$py" - "$mode" "$file" "$ROOT_SHA256" <<'PY'
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

# bundle_files: the real path of every PEM bundle: the system bundle, plus any
# file under SWEEP_DIRS with a bundle name (symlinks resolved, duplicates dropped).
bundle_files() {
    local args=() n
    for n in $BUNDLE_NAMES; do args+=(-o -name "$n"); done
    {
        [ ! -f "$SYSTEM_BUNDLE" ] || echo "$SYSTEM_BUNDLE"
        # shellcheck disable=SC2086
        find $SWEEP_DIRS \( -type f -o -type l \) \( "${args[@]:1}" \) 2>/dev/null || true
    } | python3 -c 'import os, sys
for line in sys.stdin:
    p = os.path.realpath(line.rstrip("\n"))
    if os.path.isfile(p):
        print(p)' | sort -u
}

# keystore_files: the real path of every JDK cacerts under SWEEP_DIRS.
keystore_files() {
    # shellcheck disable=SC2086
    { find $SWEEP_DIRS -path '*/lib/security/cacerts' \( -type f -o -type l \) 2>/dev/null || true; } |
        python3 -c 'import os, sys
for line in sys.stdin:
    p = os.path.realpath(line.rstrip("\n"))
    if os.path.isfile(p):
        print(p)' | sort -u
}

# keytool_for KEYSTORE: the keytool of the JDK a cacerts belongs to, if any.
keytool_for() {
    local jh="${1%/lib/security/cacerts}" kt
    for kt in "$jh/bin/keytool" "$jh/../bin/keytool"; do
        [ -x "$kt" ] && { echo "$kt"; return 0; }
    done
    return 1
}

# scan_stores: print "KIND COUNT PATH" for every CA store (KIND is bundle or
# keystore).  COUNT is how many compromised roots it holds, or ? if it can't be read.
scan_stores() {
    local f ks kt n
    while IFS= read -r f; do
        n=$(pem_bundle count "$f" 2>/dev/null) || n='?'
        echo "bundle $n $f"
    done < <(bundle_files)
    while IFS= read -r ks; do
        if kt=$(keytool_for "$ks"); then
            n=$(keystore_aliases "$kt" "$ks" | wc -l | tr -d ' ')
        else
            n='?'
        fi
        echo "keystore $n $ks"
    done < <(keystore_files)
}

# remove_from_stores: remove the roots from every store scan_stores finds them in.
remove_from_stores() {
    local kind n path kt alias
    while read -r -u 3 kind n path; do
        # 0: clean; ?: unreadable, which verify reports.
        case "$n" in 0|'?') continue ;; esac
        case "$kind" in
            bundle)
                pem_bundle remove "$path" >/dev/null
                echo "Removed the roots from $path" ;;
            keystore)
                kt=$(keytool_for "$path") || { echo "No keytool for $path"; return 1; }
                # Aliases can contain spaces (e.g. "comodoaaaca [jdk]"): read them line by line.
                while IFS= read -r alias; do
                    $SUDO "$kt" -delete -alias "$alias" -keystore "$path" -storepass changeit
                    echo "Removed '$alias' from $path"
                done < <(keystore_aliases "$kt" "$path") ;;
        esac
    done 3< <(scan_stores)
}

distrust() {
    require_python
    deny_in_keychain
    remove_from_stores
}

verify() {
    local bad=0 kind n path h pem name rc
    require_python
    while read -r -u 3 kind n path; do
        [ "$n" = 0 ] || { echo "FAIL: compromised root in $path (count: $n)"; bad=1; }
    done 3< <(scan_stores)
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

report() {
    require_python
    scan_stores | sort -k3
}

# Run only when executed, so the functions can be sourced for testing.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    case "${1:-}" in
        distrust|verify|report) "$1" ;;
        *) echo "Usage: $0 distrust|verify|report" >&2; exit 2 ;;
    esac
fi
