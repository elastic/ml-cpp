FROM ubuntu:22.04

RUN apt-get update && apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install aptitude ssh git openjdk-17-jre openjdk-17-jdk jq zip -y

# Guard for the eMudhra intermediate CA compromise (InfoSec incident "Intrepid
# Falcon", https://emudhra.we-have-a-problem.com/).  Ubuntu 22.04's
# ca-certificates package (20260816ubuntu0.22.04.1 and later) removes the
# affected roots, so no workaround is needed, but fail the build if an image
# built from a stale base still has either root (by SHA-256 fingerprint) in
# /etc/ssl/certs or in the Java keystore, or if either store can't be read.
RUN bad_fps='D7:A7:A0:FB:5D:7E:27:31:D7:71:E9:48:4E:BC:DE:F7:1D:5F:0C:3E:0A:29:48:78:2B:C8:3E:E0:EA:69:9E:F4|86:A1:EC:BA:08:9C:4A:8D:3B:BE:27:34:C6:12:BA:34:1D:81:3E:04:3C:F9:E8:A8:62:CD:5C:57:A3:6B:BE:6B' && \
    fps=$(for f in /etc/ssl/certs/*.pem; do openssl x509 -noout -fingerprint -sha256 -in "$f"; done) && \
    [ "$(echo "$fps" | wc -l)" -gt 50 ] && \
    ! echo "$fps" | grep -q -E "$bad_fps" && \
    java_fps=$(keytool -list -cacerts -storepass changeit | grep 'SHA-256') && \
    [ "$(echo "$java_fps" | wc -l)" -gt 50 ] && \
    ! echo "$java_fps" | grep -q -E "$bad_fps"

RUN useradd -ms /bin/bash -u 1000 elasticsearch

USER elasticsearch
WORKDIR /home/elasticsearch
