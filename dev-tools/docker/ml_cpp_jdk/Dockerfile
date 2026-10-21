FROM ubuntu:22.04

RUN apt-get update && apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install aptitude ssh git openjdk-17-jre openjdk-17-jdk jq zip -y

# Guard for the eMudhra intermediate CA compromise (InfoSec incident "Intrepid
# Falcon", https://emudhra.we-have-a-problem.com/).  Ubuntu 22.04's
# ca-certificates package (20260816ubuntu0.22.04.1 and later) removes the
# affected roots, so no workaround is needed here, but fail the build if an
# image built from a stale base still has either root in the Java keystore.
RUN ! keytool -list -cacerts -storepass changeit | grep -i -E 'comodo_aaa_services_root|emsign_ecc_root_ca_-_g3'

RUN useradd -ms /bin/bash -u 1000 elasticsearch

USER elasticsearch
WORKDIR /home/elasticsearch
