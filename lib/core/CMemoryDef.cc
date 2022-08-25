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

#include <core/CMemoryDec.h>
#include <core/CMemoryDef.h>

namespace ml {
namespace core {
namespace memory {
std::size_t dynamicSize(const std::string& t) {
    std::size_t capacity = t.capacity();
    // The different STLs we use on various platforms all have different
    // allocation strategies for strings
    // These are hard-coded here, on the assumption that they will not
    // change frequently - but checked by unittests that do runtime
    // verification
#ifdef MacOSX
    if (capacity <= 22) {
        // For lengths up to 22 bytes there is no allocation
        return 0;
    }
#else // Linux (with C++11 ABI) and Windows
    if (capacity <= 15) {
        // For lengths up to 15 bytes there is no allocation
        return 0;
    }
#endif
    return capacity + 1;
}

CAnyVisitor& anyVisitor() {
    return CAnyVisitor::instance();
}

std::size_t dynamicSize(const std::any& t) {
    // std::any holds a pointer to a new'd item.
    return anyVisitor().dynamicSize(t);
}
}

namespace memory_debug {

void dynamicSize(const char* name, const std::string& t, const CMemoryUsage::TMemoryUsagePtr& mem) {
    std::string componentName(name);
    componentName += "_string";
    std::size_t length = t.size();
    std::size_t capacity = t.capacity();
    std::size_t unused = 0;
#ifdef MacOSX
    if (capacity > 22) {
        unused = capacity - length;
        ++capacity;
    } else {
        // For lengths up to 22 bytes there is no allocation
        capacity = 0;
    }
#else // Linux (with C++11 ABI) and Windows
    if (capacity > 15) {
        unused = capacity - length;
        ++capacity;
    } else {
        // For lengths up to 15 bytes there is no allocation
        capacity = 0;
    }
#endif
    CMemoryUsage::SMemoryUsage usage(componentName, capacity, unused);
    CMemoryUsage::TMemoryUsagePtr ptr = mem->addChild();
    ptr->setName(usage);
}

CAnyVisitor& anyVisitor() {
    return CAnyVisitor::instance();
}

void dynamicSize(const char* name, const std::any& t, const CMemoryUsage::TMemoryUsagePtr& mem) {
    // std::any holds a pointer to a new'd item.
    anyVisitor().dynamicSize(name, t, mem);
}
}
}
}
