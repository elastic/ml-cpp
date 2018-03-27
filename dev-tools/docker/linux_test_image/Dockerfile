#
# ELASTICSEARCH CONFIDENTIAL
#
# Copyright (c) 2018 Elasticsearch BV. All Rights Reserved.
#
# Notice: this software, and all information contained
# therein, is the exclusive property of Elasticsearch BV
# and its licensors, if any, and is protected under applicable
# domestic and foreign law, and international treaties.
#
# Reproduction, republication or distribution without the
# express written consent of Elasticsearch BV is
# strictly prohibited.
#

# Increment the version here when a new tools/3rd party components image is built
FROM droberts195/ml-linux-build:3

MAINTAINER David Roberts <dave.roberts@elastic.co>

# Build cppunit
RUN \
  wget --quiet -O - http://dev-www.libreoffice.org/src/cppunit-1.13.2.tar.gz | tar zxf - && \
  cd cppunit-1.13.2 && \
  ./configure --prefix=/usr/local/gcc73 && \
  make -j`grep -c '^processor' /proc/cpuinfo` && \
  make install && \
  cd .. && \
  rm -rf cppunit-1.13.2
