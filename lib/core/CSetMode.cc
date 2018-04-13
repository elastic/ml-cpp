/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CSetMode.h>

namespace ml {
namespace core {

int CSetMode::setMode(int /* fd */, int /* mode */) {
    return 0;
}

int CSetMode::setBinaryMode(int /* fd */) {
    return 0;
}
}
}
