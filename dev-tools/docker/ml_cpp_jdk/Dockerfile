FROM ubuntu:20.04

RUN apt-get update && apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install aptitude ssh git openjdk-17-jre openjdk-17-jdk jq zip -y

# Temporary mitigation for the eMudhra intermediate CA compromise (InfoSec
# incident "Intrepid Falcon", https://emudhra.we-have-a-problem.com/).  The
# roots the compromised intermediates chain to are distrusted in the system CA
# store, and update-ca-certificates also rebuilds the Java keystore.  Ubuntu
# 20.04 has no fixed ca-certificates package, and the steps that use this image
# run as a non-root user, so this cannot be done at runtime.  Remove once the
# base image carries a fixed package.  The build fails if either root is still
# in the Java keystore.
RUN for f in Comodo_AAA_Services_root.crt emSign_ECC_Root_CA_-_G3.crt; do \
      sed -i "s#^mozilla/${f}\$#!mozilla/${f}#" /etc/ca-certificates.conf; \
    done && \
    update-ca-certificates && \
    ! keytool -list -cacerts -storepass changeit | grep -i -E 'comodo_aaa_services_root|emsign_ecc_root_ca_-_g3'

RUN useradd -ms /bin/bash -u 1000 elasticsearch

USER elasticsearch
WORKDIR /home/elasticsearch
