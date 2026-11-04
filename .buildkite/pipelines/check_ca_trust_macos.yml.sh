#!/bin/bash
# TEMPORARY throwaway: run the CA trust check on the 8.19 macOS image, without
# and with the runtime mitigation.

for mode in raw mitigated; do
  if [ $mode = mitigated ]; then
    ENVBLOCK='    env:
      EMUDHRA_MITIGATE: "true"'
  else
    ENVBLOCK=''
  fi
cat <<EOL
  - label: "CA trust check (${mode}): macOS 12 aarch64 agent"
    key: "check_ca_trust_macos12_${mode}"
    command: ".buildkite/scripts/steps/check_ca_trust.sh"
    soft_fail: true
    timeout_in_minutes: 10
    agents:
      provider: "orka"
      image: "ml-macos-12-arm-001.orkasi"
${ENVBLOCK}
EOL
done
