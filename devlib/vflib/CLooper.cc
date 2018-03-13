/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#include <vflib/CLooper.h>

#include <vflib/CIncrementer.h>

namespace ml {
namespace vflib {

size_t CLooper::inlinedLibraryCallLoop(CIncrementer &incrementer, size_t count, size_t val) {
    for (size_t i = 0; i < count; ++i) {
        val = incrementer.inlinedIncrement(val);
    }
    return val;
}

size_t CLooper::nonVirtualLibraryCallLoop(CIncrementer &incrementer, size_t count, size_t val) {
    for (size_t i = 0; i < count; ++i) {
        val = incrementer.nonVirtualIncrement(val);
    }
    return val;
}

size_t CLooper::virtualLibraryCallLoop(CIncrementer &incrementer, size_t count, size_t val) {
    for (size_t i = 0; i < count; ++i) {
        val = incrementer.virtualIncrement(val);
    }
    return val;
}
}
}
