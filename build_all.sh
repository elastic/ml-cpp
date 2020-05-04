#!/usr/bin/env bash

cd $CPP_SRC_HOME
compiledb -o ${CPP_SRC_HOME}/compile_commands.json -n make -j8

for module in 'api' 'core' 'maths' 'model'; do
  cd ${CPP_SRC_HOME}/lib/${module}/unittest
  compiledb -o ${CPP_SRC_HOME}/compile_commands.json -n make -j8
done
