/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <vflib/CIncrementer.h>

namespace ml {
namespace vflib {

CIncrementer::~CIncrementer() {
}

size_t CIncrementer::nonVirtualIncrement(size_t val) {
    return val + 1;
}

size_t CIncrementer::virtualIncrement(size_t val) {
    return val + 1;
}
}
}
