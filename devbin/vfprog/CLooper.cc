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
#include "CLooper.h"

#include <vflib/CIncrementer.h>

#include "CIncrementer.h"

namespace ml {
namespace vfprog {

size_t CLooper::inlinedProgramCallLoop(CIncrementer& incrementer, size_t count, size_t val) {
    for (size_t i = 0; i < count; ++i) {
        val = incrementer.inlinedIncrement(val);
    }
    return val;
}

size_t CLooper::nonVirtualProgramCallLoop(CIncrementer& incrementer, size_t count, size_t val) {
    for (size_t i = 0; i < count; ++i) {
        val = incrementer.nonVirtualIncrement(val);
    }
    return val;
}

size_t CLooper::virtualProgramCallLoop(CIncrementer& incrementer, size_t count, size_t val) {
    for (size_t i = 0; i < count; ++i) {
        val = incrementer.virtualIncrement(val);
    }
    return val;
}

size_t CLooper::inlinedLibraryCallLoop(vflib::CIncrementer& incrementer, size_t count, size_t val) {
    for (size_t i = 0; i < count; ++i) {
        val = incrementer.inlinedIncrement(val);
    }
    return val;
}

size_t CLooper::nonVirtualLibraryCallLoop(vflib::CIncrementer& incrementer,
                                          size_t count,
                                          size_t val) {
    for (size_t i = 0; i < count; ++i) {
        val = incrementer.nonVirtualIncrement(val);
    }
    return val;
}

size_t CLooper::virtualLibraryCallLoop(vflib::CIncrementer& incrementer, size_t count, size_t val) {
    for (size_t i = 0; i < count; ++i) {
        val = incrementer.virtualIncrement(val);
    }
    return val;
}
}
}
