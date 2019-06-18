#!/bin/bash

docker run --rm -v $CPP_SRC_HOME:/ml-cpp -u $(id -u):$(id -g) alpine:clang-format