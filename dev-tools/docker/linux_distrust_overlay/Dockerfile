#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#

# Adds the distrust of the compromised eMudhra roots (InfoSec incident
# "Intrepid Falcon") to an existing CentOS 7 build image.  CentOS 7 is end of
# life and has no fixed ca-certificates package, so this blacklists the roots
# with the script the Buildkite post-checkout hook runs, and fails the build if
# either is still trusted.  Remove it once these images are no longer used.
#
# Build it with dev-tools/docker/build_linux_distrust_overlay_images.sh, which
# copies the script into this directory first.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY distrust_emudhra_cas.sh /tmp/distrust_emudhra_cas.sh
RUN bash /tmp/distrust_emudhra_cas.sh && rm -f /tmp/distrust_emudhra_cas.sh

# Fail the build if the compromised eMudhra roots are still trusted.
RUN anchors=$(trust list --filter=ca-anchors) && \
  [ "$(echo "$anchors" | grep -c '^pkcs11:')" -gt 50 ] && \
  ! echo "$anchors" | grep -q -E 'label: (Comodo AAA Services root|AAA Certificate Services|emSign ECC Root CA - G3)$'
