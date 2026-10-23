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
# Checks whether this environment's CA store still trusts the compromised
# eMudhra intermediate CAs (InfoSec incident "Intrepid Falcon",
# https://emudhra.we-have-a-problem.com/).  The test hosts serve chains issued
# by those CAs, so a fixed client rejects them: certificate error = PASS,
# connection succeeds = FAIL, any other error = INCONCLUSIVE.
#
# Self-contained, as it runs inside every CI image.  Probes with each of curl
# and python3 found, as they can use different trust stores.  A tool that does
# not accept an ordinary host (GitHub) is inconclusive, so a broken CA bundle
# cannot look like a PASS.
# Exit status: 0 = all pass, 1 = any FAIL, 2 = inconclusive only.

set -u

HOSTS=("$@")
if [ ${#HOSTS[@]} -eq 0 ]; then
    HOSTS=(dv-g3b-emudhra.we-have-a-problem.com dv-g3b-aaa-emudhra.we-have-a-problem.com)
fi

echo "--- Environment"
uname -a
id
cat /etc/os-release 2>/dev/null | grep -E '^(PRETTY_NAME|VERSION_ID)=' || true
# CA bundle package version.
(rpm -q ca-certificates 2>/dev/null ||
 dpkg-query -W -f='ca-certificates ${Version}\n' ca-certificates 2>/dev/null ||
 apk info -v ca-certificates 2>/dev/null) | head -1 || true
python3 -c 'import certifi; print("certifi", certifi.__version__)' 2>/dev/null || true
command -v curl >/dev/null 2>&1 && curl --version | head -1

# Probes return 0 = PASS, 1 = FAIL, 2 = INCONCLUSIVE.
probe_with_curl() {
    local out rc
    out=$(curl -sS -m 30 -o /dev/null "https://$1/" 2>&1)
    rc=$?
    case $rc in
        0)  return 1 ;;
        # 60: peer certificate not trusted.  Other TLS errors (e.g. 77: CA bundle
        # unreadable, 51: hostname mismatch on curl < 7.62) say nothing about trust.
        60) return 0 ;;
        *)  echo "    curl exit $rc: $out"; return 2 ;;
    esac
}

probe_with_python() {
    python3 - "$1" <<'PYEOF'
import socket, ssl, sys
host = sys.argv[1]
try:
    with socket.create_connection((host, 443), timeout=30) as sock:
        with ssl.create_default_context().wrap_socket(sock, server_hostname=host):
            pass
except ssl.SSLCertVerificationError:
    sys.exit(0)
except Exception as e:
    print("    python error: %r" % (e,))
    sys.exit(2)
sys.exit(1)
PYEOF
}

echo "--- Probing for trust of the compromised eMudhra intermediates"
CONTROL_HOST=github.com

probe() {  # probe TOOL HOST
    case "$1" in
        curl) probe_with_curl "$2" ;;
        python3) probe_with_python "$2" ;;
    esac
}

tools=""
command -v curl >/dev/null 2>&1 && tools="$tools curl"
command -v python3 >/dev/null 2>&1 && tools="$tools python3"
if [ -z "$tools" ]; then
    echo "Neither curl nor python3 is available: cannot run the check"
    exit 2
fi

usable=""
unusable=""
for tool in $tools; do
    probe "$tool" "$CONTROL_HOST"
    if [ $? -eq 1 ]; then
        usable="$usable $tool"
    else
        unusable="$unusable $tool"
        echo "    $tool does not accept $CONTROL_HOST, so it cannot vouch for the result"
    fi
done

status=0
for host in "${HOSTS[@]}"; do
    accepted=""
    incomplete="$unusable"
    for tool in $usable; do
        probe "$tool" "$host"
        case $? in
            0) ;;
            1) accepted="$accepted $tool" ;;
            *) incomplete=yes ;;
        esac
    done
    if [ -n "$accepted" ]; then
        echo "FAIL: $host ACCEPTED by${accepted} - this environment still trusts the compromised CAs"
        status=1
    elif [ -n "$incomplete" ]; then
        echo "INCONCLUSIVE: $host (could not complete the check)"
        [ $status -eq 0 ] && status=2
    else
        echo "PASS: $host rejected by${usable} (compromised CA is not trusted)"
    fi
done
exit $status
