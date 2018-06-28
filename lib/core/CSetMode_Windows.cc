/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CSetMode.h>

#include <fcntl.h>
#include <io.h>

namespace ml {
namespace core {

int CSetMode::setMode(int fd, int mode) {
    return _setmode(fd, mode);
}

int CSetMode::setBinaryMode(int fd) {
    return _setmode(fd, _O_BINARY);
}
}
}
