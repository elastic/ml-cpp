#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# Increment the version here when a new tools/3rd party components image is built
FROM docker.elastic.co/ml-dev/ml-linux-build:4

MAINTAINER David Roberts <dave.roberts@elastic.co>

# Build cppunit
RUN \
  wget --no-check-certificate --quiet -O - http://dev-www.libreoffice.org/src/cppunit-1.13.2.tar.gz | tar zxf - && \
  cd cppunit-1.13.2 && \
  ./configure --prefix=/usr/local/gcc62 && \
  make -j`grep -c '^processor' /proc/cpuinfo` && \
  make install && \
  cd .. && \
  rm -rf cppunit-1.13.2
