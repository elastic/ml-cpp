#!/bin/bash
# THROWAWAY (do not merge): exercise .ci/orka/distrust_emudhra_cas.sh on a real
# ml-macos-14-arm-004.orkasi VM.
#
#  1. is /usr/bin/python3 (from the Xcode command line tools) usable?
#  2. which CA stores does the sweep find, and which of them did the old static
#     globs miss?  Any of those that still hold the roots is a gap.
#  3. does "distrust" then "verify" pass, and is "distrust" idempotent?
#
# Exit status: 0 = all fine, 1 = a problem was found.
set -u
cd "$(dirname "$0")/../../.."

SCRIPT=.ci/orka/distrust_emudhra_cas.sh
OUT=${TMPDIR:-/tmp}/orka_vm_check.$$
mkdir -p "$OUT"
rc=0
strip() { sed -E 's/^[a-z]+ [0-9?]+ //'; }   # "KIND COUNT PATH" -> PATH

echo "--- Environment"
sw_vers; uname -m; id -un
xcode-select -p || true
ls -l /usr/bin/python3 || true
if /usr/bin/python3 --version; then :; else echo "PROBLEM: /usr/bin/python3 is not usable"; rc=1; fi
echo "python3 on PATH (in order):"; command -v -a python3 || true

echo "--- CA stores found by the sweep (before)"
bash $SCRIPT report | tee "$OUT/before.txt"

echo "--- The old static globs (late_bundles, java_homes in ee976a787)"
{
    if command -v brew >/dev/null; then
        ls "$(brew --prefix)"/etc/ca-certificates/cert.pem "$(brew --prefix)"/etc/openssl@*/cert.pem 2>/dev/null || true
    fi
    find /opt/homebrew/lib/python3*/site-packages \
         /opt/homebrew/Cellar/python@*/*/Frameworks/Python.framework/Versions/*/lib/python3*/site-packages \
         /usr/local/lib/python3*/site-packages /Library/Python/*/site-packages \
         /Library/Frameworks/Python.framework/Versions/*/lib/python3*/site-packages \
         -maxdepth 4 -name cacert.pem 2>/dev/null || true
    echo /etc/ssl/cert.pem
    ls /Library/Java/JavaVirtualMachines/*/Contents/Home/lib/security/cacerts 2>/dev/null || true
} | python3 -c 'import os, sys
for line in sys.stdin:
    p = os.path.realpath(line.rstrip("\n"))
    if os.path.isfile(p):
        print(p)' | sort -u > "$OUT/old.txt"
cat "$OUT/old.txt"

strip < "$OUT/before.txt" | sort -u > "$OUT/sweep.txt"
echo "--- Stores found by the sweep but not by the old globs"
comm -13 "$OUT/old.txt" "$OUT/sweep.txt" | tee "$OUT/extra.txt"
echo "($(wc -l < "$OUT/extra.txt" | tr -d ' ') extra, $(wc -l < "$OUT/sweep.txt" | tr -d ' ') in total)"

echo "--- Extra stores that hold the roots (or can't be read)"
awk 'NR==FNR { extra[$0] = 1; next }
     { p = $0; sub(/^[a-z]+ [0-9?]+ /, "", p); split($0, f, " ");
       if ((p in extra) && f[2] != "0") print }' "$OUT/extra.txt" "$OUT/before.txt" | tee "$OUT/gap.txt"
if [ -s "$OUT/gap.txt" ]; then echo "PROBLEM: stores outside the old globs still hold the roots"; rc=1; fi

echo "--- distrust"
bash $SCRIPT distrust || { echo "PROBLEM: distrust failed"; rc=1; }
echo "--- verify"
bash $SCRIPT verify || { echo "PROBLEM: verify failed"; rc=1; }
echo "--- distrust again (should change nothing)"
bash $SCRIPT distrust | grep -E "^Removed" && { echo "PROBLEM: second distrust still removed something"; rc=1; }
echo "--- CA stores found by the sweep (after)"
bash $SCRIPT report

[ $rc -eq 0 ] && echo "OK: python3 usable, sweep found no extra stores holding the roots, distrust and verify pass"
exit $rc
