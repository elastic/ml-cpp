/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
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
